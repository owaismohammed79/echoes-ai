"""Tests for ``POST /api/qa`` -- the Push-To-Talk pipeline.

The headline requirement: on TTS failure the response must set
``use_native_tts: true`` in a header, so the client falls back to
``window.speechSynthesis`` with text it already holds. Both directions of that
flag are asserted here, plus the guard rails that must fire *before* any paid
call, and the rule that only validated text ever reaches synthesis.
"""

from __future__ import annotations

from urllib.parse import unquote

import pytest

from app.api import qa as qa_route
from app.services.prompts import CANNED_REFUSAL
from app.services.sessions import SessionStore
from app.services.stt import STTError, STTRateLimited
from app.services.tts import TTSRateLimited
from tests.helpers import (
    FAKE_AUDIO,
    FID,
    FID_SOURCE,
    OTHER,
    ScriptedLLM,
    StubSTT,
    StubTTS,
    admission,
    build_container,
    client_for,
    fact,
)


def _post_qa(client, *, session_id=None, clip=b"clip-bytes", content_type="audio/mp4", **form):
    data = {k: v for k, v in form.items() if v is not None}
    if session_id is not None:
        data["session_id"] = session_id
    return client.post(
        "/api/qa",
        files={"audio": ("question.m4a", clip, content_type)},
        data=data,
    )


def _session_store(*facts, anchor=(51.507, -0.127)) -> tuple[SessionStore, str]:
    """A walk session anchored where the facts were fetched."""
    store = SessionStore()
    session = store.create(facts=list(facts) or [fact()], anchor=anchor)
    return store, session.session_id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_qa_returns_audio_with_metadata_headers() -> None:
    store, session_id = _session_store()
    tts, stt = StubTTS(), StubSTT(transcript="who built the column?")
    llm = ScriptedLLM([f"Nelson's Column was completed in 1843.[cite:{FID}]"])
    container = build_container(llm=llm, stt=stt, tts=tts, sessions=store)

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == FAKE_AUDIO

    # The flag is present in both spellings and is false: audio is real.
    assert resp.headers["x-use-native-tts"] == "false"
    assert resp.headers["use_native_tts"] == "false"

    # Text metadata travels in headers, because the body is audio.
    assert unquote(resp.headers["x-transcript"]) == "who built the column?"
    assert unquote(resp.headers["x-answer-text"]) == "Nelson's Column was completed in 1843."
    assert resp.headers["x-answer-citations"] == FID
    assert unquote(resp.headers["x-answer-sources"]) == FID_SOURCE
    assert resp.headers["x-answer-refused"] == "false"

    # Per-stage timings, so a slow dependency can be identified.
    timings = resp.headers["x-timings-ms"]
    assert "transcribe=" in timings and "answer=" in timings and "tts=" in timings

    # The tag never reaches the synthesis layer.
    assert tts.spoken == ["Nelson's Column was completed in 1843."]
    assert "[cite:" not in tts.spoken[0]

    # The nearby place name was seeded into the transcription prompt.
    assert stt.uploads[0]["vocabulary"] == ["Nelson's Column"]
    assert stt.uploads[0]["bytes"] == len(b"clip-bytes")


def test_qa_streams_audio_rather_than_buffering_the_whole_clip() -> None:
    """The handler must return once the first chunk exists, holding the rest.

    This calls the handler directly rather than going through ``TestClient``:
    httpx's ASGI transport buffers a streamed body before handing back the
    response, so a client-side assertion could not distinguish streaming from
    buffering. Inspecting the returned object can: if the handler had buffered
    the clip, the provider would already have produced every chunk by the time
    it returned.
    """
    import asyncio
    import io

    from fastapi.responses import StreamingResponse
    from starlette.datastructures import Headers
    from starlette.datastructures import UploadFile as StarletteUploadFile

    from app.api.qa import api_qa

    store, session_id = _session_store()

    class CountingTTS(StubTTS):
        def __init__(self) -> None:
            super().__init__(chunks=(b"FIRST", b"SECOND"))
            self.produced: list[bytes] = []

        async def stream_audio(self, text: str):
            self.spoken.append(text)
            for chunk in self._chunks:
                self.produced.append(chunk)
                yield chunk

    tts = CountingTTS()
    container = build_container(
        llm=ScriptedLLM([f"A claim.[cite:{FID}]"]),
        stt=StubSTT(),
        tts=tts,
        sessions=store,
    )

    async def scenario() -> tuple[StreamingResponse, bytes]:
        upload = StarletteUploadFile(
            file=io.BytesIO(b"clip-bytes"),
            filename="q.m4a",
            headers=Headers({"content-type": "audio/mp4"}),
        )
        response = await api_qa(
            container=container,
            # The handler takes its admission from a dependency in production;
            # called directly, it has to be handed one.
            admission=admission(),
            audio=upload,
            session_id=session_id,
            lat=None,
            lon=None,
        )
        assert isinstance(response, StreamingResponse)
        # Exactly one chunk has been pulled: the clip is not buffered yet, and
        # the flag is already decided.
        assert tts.produced == [b"FIRST"]
        body = b"".join([chunk async for chunk in response.body_iterator])
        return response, body

    response, body = asyncio.run(scenario())
    assert response.headers["x-use-native-tts"] == "false"
    assert response.headers["x-answer-citations"] == FID
    assert response.media_type == "audio/mpeg"
    assert body == b"FIRSTSECOND"


def test_history_records_the_validated_answer() -> None:
    """Storing raw model output would prime the model with its own claim."""
    store, session_id = _session_store()
    tts, stt = StubTTS(), StubSTT(transcript="who built the column?")
    llm = ScriptedLLM([f"Nelson's Column was completed in 1843.[cite:{FID}]"])
    container = build_container(llm=llm, stt=stt, tts=tts, sessions=store)

    with client_for(container) as client:
        _post_qa(client, session_id=session_id)

    session = store.get(session_id)
    assert session is not None
    assert session.history == [
        ("user", "who built the column?"),
        ("assistant", "Nelson's Column was completed in 1843."),
    ]


def test_prompt_carries_the_question_and_the_cached_facts_to_the_model() -> None:
    """The Gemma call must receive both the transcript and the walk's facts.

    Without them the citation the validator later checks could not have been
    produced, so this pins the input side of the contract rather than trusting
    the output side.
    """
    store, session_id = _session_store(fact(FID, "Nelson's Column"))
    llm = ScriptedLLM([f"A claim.[cite:{FID}]"])
    container = build_container(
        llm=llm, stt=StubSTT(transcript="who built the column?"), tts=StubTTS(), sessions=store
    )

    with client_for(container) as client:
        _post_qa(client, session_id=session_id)

    prompt = llm.calls[0]["user"]
    assert "who built the column?" in prompt
    assert FID in prompt
    assert "Nelson's Column" in prompt
    # The source URL stays withheld: the client resolves it from the fact_id.
    assert FID_SOURCE not in prompt


def test_second_question_carries_the_first_turn_back_to_the_model() -> None:
    store, session_id = _session_store()
    llm = ScriptedLLM(
        [
            f"Nelson's Column was completed in 1843.[cite:{FID}]",
            f"It stands in the square.[cite:{FID}]",
        ]
    )
    container = build_container(
        llm=llm, stt=StubSTT(transcript="who built it?"), tts=StubTTS(), sessions=store
    )

    with client_for(container) as client:
        _post_qa(client, session_id=session_id)
        _post_qa(client, session_id=session_id)

    assert "RECENT CONVERSATION" in llm.calls[1]["user"]
    assert "completed in 1843" in llm.calls[1]["user"]


def test_question_carries_the_narration_to_the_model() -> None:
    """A follow-up like "what was that about?" must see what the walk narrated."""
    store, session_id = _session_store(fact(FID, "Nelson's Column"))
    store.get(session_id).narration_text = "Nelson's Column rises over the square."
    llm = ScriptedLLM([f"A claim.[cite:{FID}]"])
    container = build_container(
        llm=llm,
        stt=StubSTT(transcript="what was that about?"),
        tts=StubTTS(),
        sessions=store,
    )

    with client_for(container) as client:
        _post_qa(client, session_id=session_id)

    prompt = llm.calls[0]["user"]
    assert "WHAT YOU JUST NARRATED" in prompt
    assert "Nelson's Column rises over the square." in prompt


# ---------------------------------------------------------------------------
# TTS failure -> use_native_tts in a header
# ---------------------------------------------------------------------------


def test_tts_rate_limit_sets_the_native_tts_header() -> None:
    store, session_id = _session_store()
    tts = StubTTS(error=TTSRateLimited("429 from ElevenLabs"))
    llm = ScriptedLLM([f"Nelson's Column was completed in 1843.[cite:{FID}]"])
    container = build_container(
        llm=llm, stt=StubSTT(), tts=tts, sessions=store
    )

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.status_code == 200
    assert resp.headers["x-use-native-tts"] == "true"
    assert resp.headers["use_native_tts"] == "true"
    assert resp.headers["x-fallback-reason"] == "elevenlabs_rate_limited"

    # The grounded answer is still delivered, as text, for speechSynthesis.
    body = resp.json()
    assert body["use_native_tts"] is True
    assert body["reason"] == "elevenlabs_rate_limited"
    assert body["answer"] == "Nelson's Column was completed in 1843."
    assert body["citations"] == [FID]
    # ...and it is in the header too, so a client reading headers is not blind.
    assert unquote(resp.headers["x-answer-text"]) == body["answer"]


def test_tts_provider_error_sets_the_native_tts_header() -> None:
    store, session_id = _session_store()
    container = build_container(
        llm=ScriptedLLM([f"A claim.[cite:{FID}]"]),
        stt=StubSTT(),
        tts=StubTTS(error=RuntimeError("socket closed")),
        sessions=store,
    )

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.headers["x-use-native-tts"] == "true"
    assert resp.headers["x-fallback-reason"] == "elevenlabs_error"


def test_unconfigured_tts_falls_back_proactively() -> None:
    """No provider at all must flag the client rather than fail the request."""
    store, session_id = _session_store()
    container = build_container(
        llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), stt=StubSTT(), tts=None, sessions=store
    )

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.status_code == 200
    assert resp.headers["x-fallback-reason"] == "tts_not_configured"
    assert resp.headers["x-use-native-tts"] == "true"


def test_open_breaker_falls_back_without_calling_the_provider() -> None:
    store, session_id = _session_store()
    tts = StubTTS()
    container = build_container(
        llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), stt=StubSTT(), tts=tts, sessions=store
    )
    # Trip the breaker as three prior failures would.
    for _ in range(3):
        container.tts_queue.breaker.record_failure()

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.headers["x-fallback-reason"] == "circuit_open"
    assert tts.spoken == []


# ---------------------------------------------------------------------------
# Refusal path
# ---------------------------------------------------------------------------


def test_uncited_answer_is_replaced_by_the_canned_refusal() -> None:
    store, session_id = _session_store()
    tts = StubTTS()
    llm = ScriptedLLM(["Nelson's Column is the tallest thing around."])
    container = build_container(llm=llm, stt=StubSTT(), tts=tts, sessions=store)

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.status_code == 200
    assert resp.headers["x-answer-refused"] == "true"
    assert resp.headers["x-refusal-reason"] == "ungrounded_answer"
    # Only the constant is ever spoken -- never model output.
    assert tts.spoken == [CANNED_REFUSAL]
    assert unquote(resp.headers["x-answer-text"]) == CANNED_REFUSAL


def test_fabricated_fact_id_is_refused() -> None:
    store, session_id = _session_store()
    llm = ScriptedLLM([f"Invented detail.[cite:{OTHER}]"])  # valid shape, unknown id
    container = build_container(llm=llm, stt=StubSTT(), tts=StubTTS(), sessions=store)

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.headers["x-refusal-reason"] == "ungrounded_answer"


def test_insufficient_sentinel_refuses_without_speaking_it() -> None:
    store, session_id = _session_store()
    tts = StubTTS()
    container = build_container(
        llm=ScriptedLLM(["INSUFFICIENT"]), stt=StubSTT(), tts=tts, sessions=store
    )

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.headers["x-refusal-reason"] == "insufficient_facts"
    assert tts.spoken == [CANNED_REFUSAL]
    assert "INSUFFICIENT" not in tts.spoken[0]


def test_refusal_history_stores_the_refusal_not_the_hallucination() -> None:
    store, session_id = _session_store()
    llm = ScriptedLLM(["Nelson's Column is the tallest thing around."])
    container = build_container(llm=llm, stt=StubSTT(), tts=StubTTS(), sessions=store)

    with client_for(container) as client:
        _post_qa(client, session_id=session_id)

    session = store.get(session_id)
    assert session is not None
    assert ("assistant", CANNED_REFUSAL) in session.history
    assert "tallest" not in str(session.history)


def test_model_failure_refuses_instead_of_erroring() -> None:
    store, session_id = _session_store()
    container = build_container(
        llm=ScriptedLLM(RuntimeError("model down")),
        stt=StubSTT(),
        tts=StubTTS(),
        sessions=store,
    )

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.status_code == 200
    assert resp.headers["x-refusal-reason"] == "empty_answer"


# ---------------------------------------------------------------------------
# Session guards
# ---------------------------------------------------------------------------


def test_unknown_session_refuses() -> None:
    container = build_container(
        llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), stt=StubSTT(), tts=StubTTS()
    )

    with client_for(container) as client:
        resp = _post_qa(client, session_id="never-issued")

    assert resp.status_code == 200
    assert resp.headers["x-refusal-reason"] == "no_session"
    assert resp.headers["x-use-native-tts"] == "true"


def test_missing_session_field_refuses() -> None:
    container = build_container(llm=ScriptedLLM(["x"]), stt=StubSTT(), tts=StubTTS())

    with client_for(container) as client:
        resp = _post_qa(client)

    assert resp.headers["x-refusal-reason"] == "no_session"


def test_stale_context_refuses_rather_than_answering_about_the_old_place() -> None:
    store, session_id = _session_store()
    tts = StubTTS()
    container = build_container(
        llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), stt=StubSTT(), tts=tts, sessions=store
    )

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id, lat=53.0, lon=-2.0)

    assert resp.headers["x-refusal-reason"] == "context_stale"
    assert tts.spoken == []


def test_walker_inside_the_radius_is_not_treated_as_stale() -> None:
    store, session_id = _session_store()
    container = build_container(
        llm=ScriptedLLM([f"A claim.[cite:{FID}]"]),
        stt=StubSTT(),
        tts=StubTTS(),
        sessions=store,
    )

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id, lat=51.5080, lon=-0.1280)

    assert resp.headers["x-answer-refused"] == "false"


# ---------------------------------------------------------------------------
# Pre-paid-call guards
# ---------------------------------------------------------------------------


def test_unsupported_content_type_is_rejected_before_transcription() -> None:
    stt = StubSTT()
    container = build_container(
        llm=ScriptedLLM(["x"]), stt=stt, tts=StubTTS()
    )

    with client_for(container) as client:
        resp = _post_qa(client, content_type="application/json")

    assert resp.status_code == 415
    assert stt.uploads == []


def test_oversize_upload_is_rejected_before_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qa_route, "MAX_AUDIO_BYTES", 8)
    stt = StubSTT()
    container = build_container(llm=ScriptedLLM(["x"]), stt=stt, tts=StubTTS())

    with client_for(container) as client:
        resp = _post_qa(client, clip=b"123456789")

    assert resp.status_code == 413
    assert stt.uploads == []


def test_empty_upload_is_rejected() -> None:
    stt = StubSTT()
    container = build_container(llm=ScriptedLLM(["x"]), stt=stt, tts=StubTTS())

    with client_for(container) as client:
        resp = _post_qa(client, clip=b"")

    assert resp.status_code == 422
    assert stt.uploads == []


def test_unconfigured_transcriber_returns_503() -> None:
    container = build_container(llm=ScriptedLLM(["x"]), stt=None, tts=StubTTS())

    with client_for(container) as client:
        resp = _post_qa(client, session_id="whatever")

    assert resp.status_code == 503


def test_groq_rate_limit_surfaces_as_429() -> None:
    store, session_id = _session_store()
    container = build_container(
        llm=ScriptedLLM(["x"]),
        stt=StubSTT(error=STTRateLimited("429")),
        tts=StubTTS(),
        sessions=store,
    )

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.status_code == 429


def test_transcription_failure_surfaces_as_502() -> None:
    store, session_id = _session_store()
    container = build_container(
        llm=ScriptedLLM(["x"]),
        stt=StubSTT(error=STTError("upstream down")),
        tts=StubTTS(),
        sessions=store,
    )

    with client_for(container) as client:
        resp = _post_qa(client, session_id=session_id)

    assert resp.status_code == 502
