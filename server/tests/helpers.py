"""Shared test scaffolding.

The point of these doubles is to make *what was asked of each dependency*
observable. ``StubTTS.spoken`` in particular is how the tests prove that a
``[cite:...]`` tag never reaches the synthesis layer.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta
from typing import Any, AsyncIterator, Iterator, Sequence

import httpx
from fastapi.testclient import TestClient

from app.dependencies import Container, get_container
from app.main import create_app
from app.models import FactObject
from app.services.audio_tokens import AudioTokenStore
from app.services.budgets import (
    Admission,
    Guardrails,
    PerIPWalkLimiter,
    Prices,
    SpendLedger,
)
from app.services.sessions import SessionStore
from app.services.tts_queue import TTSQueue

FID = "9c1f2ab73e0d5144"
OTHER = "4be7c0a91d2f8836"
FID_SOURCE = "https://en.wikipedia.org/wiki/Nelson%27s_Column"

FAKE_AUDIO = b"ID3\x03\x00\x00\x00FAKE-MP3-BYTES"


def fact(
    fid: str = FID,
    title: str = "Nelson's Column",
    *,
    source_url: str = FID_SOURCE,
    distance_m: float = 21.6,
    kind: str = "monument",
) -> FactObject:
    """A representative graded fact."""
    return FactObject(
        fact_id=fid,
        title=title,
        snippet=f"{title} is a monument in the square.",
        source_url=source_url,
        distance_m=distance_m,
        kind=kind,
    )


class StubTTS:
    """Records every string handed to the provider; yields scripted audio."""

    def __init__(
        self, *, error: Exception | None = None, chunks: Sequence[bytes] = (FAKE_AUDIO,)
    ) -> None:
        self.spoken: list[str] = []
        self._error = error
        self._chunks = tuple(chunks)

    @property
    def content_type(self) -> str:
        return "audio/mpeg"

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        if self._error is not None:
            raise self._error
        for chunk in self._chunks:
            yield chunk


class StubSTT:
    """Records the upload; returns a scripted transcript or raises."""

    def __init__(
        self,
        *,
        transcript: str = "who built the column?",
        error: Exception | None = None,
    ) -> None:
        self.transcript = transcript
        self._error = error
        self.uploads: list[dict[str, Any]] = []

    async def transcribe(
        self,
        *,
        audio: bytes,
        filename: str = "clip.m4a",
        content_type: str = "audio/mp4",
        vocabulary: list[str] | None = None,
    ) -> str:
        self.uploads.append(
            {
                "bytes": len(audio),
                "filename": filename,
                "content_type": content_type,
                "vocabulary": list(vocabulary or []),
            }
        )
        if self._error is not None:
            raise self._error
        return self.transcript


class ScriptedLLM:
    """Yields a scripted response per call; raises when scripted to."""

    def __init__(self, *responses: Sequence[str] | Exception) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, str]] = []

    async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
        index = len(self.calls)
        self.calls.append({"system": system, "user": user})
        response = self._responses[min(index, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        for chunk in response:
            yield chunk


def empty_wikipedia(request: httpx.Request) -> httpx.Response:
    """A well-formed GeoSearch response with no hits."""
    return httpx.Response(200, json={"query": {"geosearch": [], "pages": []}})


class FakeClock:
    """A monotonic clock the test drives, for window expiry without sleeping."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeDay:
    """A calendar the test rolls over, for the ledger's day boundary."""

    def __init__(self, day: date | None = None) -> None:
        self.day = day or date(2026, 1, 1)

    def __call__(self) -> date:
        return self.day

    def advance(self, days: int = 1) -> None:
        self.day = self.day + timedelta(days=days)


def admission(*, demo: bool = False) -> Admission:
    """A decision for tests that call a handler directly, with no request to admit."""
    return Admission(allowed=True, demo=demo, walks_remaining=1_000)


def build_guardrails(
    *,
    walk_limit: int = 1_000,
    window_s: float = 3_600.0,
    cap_usd: float = 0.0,
    force_demo: bool = False,
    clock: Any = None,
) -> Guardrails:
    """Guardrails for tests.

    Generous by default -- the existing suites post many walks and none of them
    are about the limiter -- and the cap is disabled at zero, so only the tests
    that configure a budget can trip it.
    """
    return Guardrails(
        limiter=PerIPWalkLimiter(
            limit=walk_limit, window_s=window_s, **({"clock": clock} if clock else {})
        ),
        ledger=SpendLedger(cap_usd=cap_usd, prices=Prices(configured=True)),
        force_demo=force_demo,
    )


def build_container(
    *,
    llm: Any = None,
    stt: Any = None,
    tts: Any = None,
    sessions: SessionStore | None = None,
    audio_tokens: AudioTokenStore | None = None,
    guardrails: Guardrails | None = None,
) -> Container:
    """A container whose HTTP client never touches the network."""
    http = httpx.AsyncClient(transport=httpx.MockTransport(empty_wikipedia))
    return Container(
        http=http,
        llm=llm,
        stt=stt,
        tts_queue=TTSQueue(tts),
        sessions=sessions or SessionStore(),
        audio_tokens=audio_tokens or AudioTokenStore(),
        guardrails=guardrails or build_guardrails(),
    )


@contextmanager
def client_for(container: Container) -> Iterator[TestClient]:
    """Run the real app with the container dependency swapped out.

    This is the seam the refactor buys: routes depend on a typed container, so a
    test replaces one object instead of patching module globals.
    """
    application = create_app()
    application.dependency_overrides[get_container] = lambda: container
    with TestClient(application) as client:
        yield client


def ndjson(response: httpx.Response) -> list[dict[str, Any]]:
    """Parse an NDJSON body into frames."""
    import json

    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def spoken(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Narration text frames, excluding the pedestrian reminder."""
    return [
        frame
        for frame in frames
        if frame.get("type") in {"text", "sentence"}
        and frame.get("role") != "reminder"
    ]


def frame_text(frame: dict[str, Any]) -> str:
    return str(frame.get("content") or frame.get("text") or "")
