"""Tests for the narration provider cascade: Gemma -> Groq, under one budget.

The cascade is the difference between a walk that continues through an AI Studio
outage and one that drops to static copy (or, worse, 500s). These tests exercise
every rung without a key or a network: ``httpx.MockTransport`` stands in for both
providers, and the failover loop is driven by an injectable clock so the 15-second
budget is spent in microseconds rather than in real time.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Sequence

import httpx
import pytest

from app.services import factories
from app.services.gemma import (
    FailoverLLMProvider,
    GemmaProvider,
    GroqGemmaProvider,
)
from app.services.providers import LLMError, LLMOutageError, LLMTemporaryOutage
from app.services.narration import generate_narration
from tests.helpers import FID, build_container, client_for, fact

from app.services import parks, weather, wikipedia


def collect(agen: AsyncIterator[str]) -> list[str]:
    async def run() -> list[str]:
        return [chunk async for chunk in agen]

    return asyncio.run(run())


class ScriptedProvider:
    """Yields a scripted response per call; raises when scripted to."""

    def __init__(self, *responses: Sequence[str] | Exception) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
        self.calls += 1
        response = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        for chunk in response:
            yield chunk


class AlwaysDown:
    """A provider that fails before producing any text."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or LLMOutageError("HTTP 503")

    async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
        raise self._exc
        yield ""  # pragma: no cover - makes this an async generator


# ---------------------------------------------------------------------------
# Provider error classification
# ---------------------------------------------------------------------------


def _gemma(handler: Any) -> GemmaProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GemmaProvider(api_key="test-key", client=client)


def test_gemma_maps_5xx_to_an_outage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    with pytest.raises(LLMOutageError, match="HTTP 503"):
        collect(_gemma(handler).stream(system="s", user="u"))


def test_gemma_maps_a_4xx_to_a_terminal_error() -> None:
    """A bad request will fail the same way on every attempt: do not retry it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad prompt")

    with pytest.raises(LLMError) as excinfo:
        collect(_gemma(handler).stream(system="s", user="u"))
    assert not isinstance(excinfo.value, LLMOutageError)


def test_gemma_maps_a_timeout_to_an_outage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(LLMOutageError, match="timed out"):
        collect(_gemma(handler).stream(system="s", user="u"))


# ---------------------------------------------------------------------------
# The Groq rung
# ---------------------------------------------------------------------------


def _groq(handler: Any) -> GroqGemmaProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GroqGemmaProvider(api_key="groq-key", client=client)


def _groq_sse(*deltas: str) -> bytes:
    body = "".join(
        f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}\n\n"
        for d in deltas
    )
    return (body + "data: [DONE]\n\n").encode("utf-8")


def test_groq_streams_openai_style_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_groq_sse("Nelson's Column ", "rises."))

    assert collect(_groq(handler).stream(system="s", user="u")) == [
        "Nelson's Column ",
        "rises.",
    ]


def test_groq_posts_the_bearer_key_and_streams() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, content=_groq_sse("ok."))

    collect(_groq(handler).stream(system="SYSTEM", user="USER"))
    assert seen["url"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer groq-key"
    assert seen["payload"]["model"] == "gemma2-9b-it"
    assert seen["payload"]["stream"] is True
    assert seen["payload"]["messages"][0] == {"role": "system", "content": "SYSTEM"}


def test_groq_maps_5xx_to_an_outage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="groq down")

    with pytest.raises(LLMOutageError, match="HTTP 500"):
        collect(_groq(handler).stream(system="s", user="u"))


# ---------------------------------------------------------------------------
# The failover loop
# ---------------------------------------------------------------------------


def test_cascade_hands_over_to_the_secondary_provider() -> None:
    secondary = ScriptedProvider(["From Groq."])
    provider = FailoverLLMProvider([AlwaysDown(), secondary])

    assert collect(provider.stream(system="s", user="u")) == ["From Groq."]
    assert secondary.calls == 1


def test_cascade_does_not_replay_a_stream_that_already_spoke() -> None:
    """Text already yielded is already heard: fail over before it, never after."""
    forbidden = AlwaysDown(AssertionError("secondary must not be called"))

    class PartialThenFail:
        async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
            yield "already said."
            raise LLMOutageError("died mid-stream")

    provider = FailoverLLMProvider([PartialThenFail(), forbidden])
    with pytest.raises(LLMOutageError, match="mid-stream"):
        collect(provider.stream(system="s", user="u"))


def test_cascade_exhausts_its_budget_then_reports_a_temporary_outage() -> None:
    now = [0.0]
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    provider = FailoverLLMProvider(
        [AlwaysDown(), AlwaysDown()],
        budget=15.0,
        clock=lambda: now[0],
        sleep=fake_sleep,
    )

    with pytest.raises(LLMTemporaryOutage) as excinfo:
        collect(provider.stream(system="s", user="u"))

    assert excinfo.value.code == "LLM_TEMPORARY_OUTAGE"
    assert "high global traffic" in excinfo.value.message
    # The budget is a ceiling, not a suggestion.
    assert now[0] <= 15.0
    # First hand-off is immediate; the ladder that follows is 1s then 2s.
    assert sleeps[:2] == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Wiring and the orchestrator
# ---------------------------------------------------------------------------


def test_factory_builds_both_rungs_from_the_two_keys(monkeypatch) -> None:
    monkeypatch.setenv("GEMMA_API_KEY", "google-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provider = factories.build_llm_provider()
    assert isinstance(provider, FailoverLLMProvider)
    # Wrap both rungs: Gemini first, Groq second.
    assert len(provider._providers) == 2  # noqa: SLF001 - the wiring is the test
    assert isinstance(provider._providers[0], GemmaProvider)  # noqa: SLF001
    assert isinstance(provider._providers[1], GroqGemmaProvider)  # noqa: SLF001


def test_factory_returns_none_with_no_keys(monkeypatch) -> None:
    for name in ("GEMMA_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert factories.build_llm_provider() is None


def test_narration_propagates_a_temporary_outage() -> None:
    """An outage is not a bad sentence: it must not be swallowed as static copy."""

    class OutageProvider:
        async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
            raise LLMTemporaryOutage("all rungs down")
            yield ""  # pragma: no cover

    with pytest.raises(LLMTemporaryOutage):
        asyncio.run(
            generate_narration(
                mode="STORY", provider=OutageProvider(), facts=[fact()]
            )
        )


# ---------------------------------------------------------------------------
# The API's structured error
# ---------------------------------------------------------------------------


def _stub_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    async def wiki_facts(client, lat, lon, *, radius_m: int | None = None):
        return [fact(FID, "Nelson's Column")]

    async def park_facts(client, lat, lon):
        return []

    async def no_history(client, lat, lon, *, radius_m: int | None = None):
        return []

    async def no_weather(client, lat, lon):
        return None

    monkeypatch.setattr(wikipedia, "fetch_wikipedia_facts", wiki_facts)
    monkeypatch.setattr(wikipedia, "fetch_wikipedia_history_facts", no_history)
    monkeypatch.setattr(parks, "fetch_park_facts", park_facts)
    monkeypatch.setattr(weather, "fetch_weather", no_weather)


def test_walk_returns_a_structured_outage_when_the_cascade_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_sources(monkeypatch)
    container = build_container(llm=AlwaysDown(LLMTemporaryOutage("all down")))

    with client_for(container) as client:
        resp = client.post("/api/walk", json={"lat": 51.507, "lon": -0.127})

    assert resp.status_code == 503
    assert resp.json() == {
        "status": "error",
        "code": "LLM_TEMPORARY_OUTAGE",
        "message": "Gemma is experiencing high global traffic. Please try again shortly.",
    }
