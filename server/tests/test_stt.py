"""Tests for Groq Whisper transcription.

The upload is attacker-controlled and every accepted clip is a paid API call, so
the guards and the error mapping are the parts worth pinning down.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.services.stt import (
    STTError,
    STTRateLimited,
    GroqWhisper,
    build_vocabulary_prompt,
    is_supported_content_type,
)


def _stt(handler, **kwargs) -> GroqWhisper:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GroqWhisper(api_key="groq-key", client=client, **kwargs)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_transcription_posts_openai_compatible_multipart() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"text": "  who built the column?  "})

    transcript = run(
        _stt(handler).transcribe(
            audio=b"clip-bytes",
            filename="question.m4a",
            content_type="audio/mp4",
            vocabulary=["Nelson's Column", "Hyde Park"],
        )
    )

    body = seen["body"]
    assert isinstance(body, bytes)
    assert transcript == "who built the column?"  # stripped
    assert str(seen["url"]).endswith("/audio/transcriptions")
    assert seen["auth"] == "Bearer groq-key"
    assert b"whisper-large-v3" in body
    assert b'name="language"' in body and b"en" in body
    assert b'name="temperature"' in body and b"0" in body
    assert b"question.m4a" in body
    assert b"clip-bytes" in body
    # The nearby place names are seeded so a proper noun is spelled correctly.
    assert b"Nelson's Column" in body


def test_model_is_configurable_for_the_cheaper_turbo_variant() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "hi"})

    run(_stt(handler, model="whisper-large-v3-turbo").transcribe(audio=b"x"))
    assert b"whisper-large-v3-turbo" in seen["body"]


def test_no_prompt_field_when_there_is_no_vocabulary() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert b'name="prompt"' not in request.content
        return httpx.Response(200, json={"text": "hi"})

    run(_stt(handler).transcribe(audio=b"x"))


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_429_becomes_rate_limited() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow down"})

    with pytest.raises(STTRateLimited):
        run(_stt(handler).transcribe(audio=b"x"))


def test_error_status_becomes_stt_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream exploded")

    with pytest.raises(STTError, match="500"):
        run(_stt(handler).transcribe(audio=b"x"))


def test_error_body_inside_a_200_becomes_stt_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "bad audio"}})

    with pytest.raises(STTError, match="bad audio"):
        run(_stt(handler).transcribe(audio=b"x"))


def test_non_json_response_becomes_stt_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>oops</html>")

    with pytest.raises(STTError, match="non-JSON"):
        run(_stt(handler).transcribe(audio=b"x"))


def test_transport_failure_becomes_stt_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(STTError, match="transport failure"):
        run(_stt(handler).transcribe(audio=b"x"))


def test_missing_api_key_is_a_construction_error() -> None:
    with pytest.raises(ValueError):
        GroqWhisper(api_key="")


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content_type",
    ["audio/mp4", "audio/m4a", "audio/x-m4a", "audio/webm", "audio/mpeg", "video/mp4"],
)
def test_supported_content_types(content_type: str) -> None:
    assert is_supported_content_type(content_type)


@pytest.mark.parametrize(
    "content_type",
    [None, "", "application/json", "text/plain", "audio/mp3;codecs=opus"],
)
def test_unsupported_content_types(content_type: str | None) -> None:
    assert not is_supported_content_type(content_type)


def test_content_type_parameters_are_stripped_before_matching() -> None:
    assert is_supported_content_type("audio/webm; codecs=opus")


# ---------------------------------------------------------------------------
# Vocabulary seeding
# ---------------------------------------------------------------------------


def test_vocabulary_prompt_preserves_order_and_deduplicates() -> None:
    prompt = build_vocabulary_prompt(["Hyde Park", "Hyde Park", " ", "Trafalgar Square"])
    assert prompt == "Hyde Park, Trafalgar Square"


def test_vocabulary_prompt_is_truncated_to_the_documented_limit() -> None:
    prompt = build_vocabulary_prompt(["A" * 500], limit=224)
    assert len(prompt) == 224


def test_vocabulary_prompt_is_empty_without_titles() -> None:
    assert build_vocabulary_prompt([]) == ""
    assert build_vocabulary_prompt(["", "   "]) == ""
