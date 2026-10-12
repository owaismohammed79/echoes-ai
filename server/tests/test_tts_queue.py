"""Tests for synthesis dispatch: ordering, bounds, breaker, and failure mapping.

These are the tests that make ``use_native_tts`` trustworthy. Every failure mode
of the provider must become a *reason* on a :class:`SynthesisOutcome` rather
than an exception, because the only correct response to a TTS problem is a flag
telling the client to speak with ``window.speechSynthesis``.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.services.tts import ElevenLabsTTS, TTSError, TTSRateLimited
from app.services.tts_queue import CircuitBreaker, TTSQueue
from tests.helpers import FAKE_AUDIO, StubTTS


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Provider request shape
# ---------------------------------------------------------------------------


def _provider(handler, **kwargs) -> ElevenLabsTTS:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ElevenLabsTTS(
        api_key="test-key", voice_id="voice-123", client=client, **kwargs
    )


def test_provider_posts_to_the_streaming_endpoint_with_the_flash_model() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["model"] = request.read().decode()
        seen["api_key"] = request.headers.get("xi-api-key")
        return httpx.Response(200, content=FAKE_AUDIO)

    async def collect():
        return [c async for c in _provider(handler).stream_audio("Hello there.")]

    assert run(collect()) == [FAKE_AUDIO]
    assert "/v1/text-to-speech/voice-123/stream" in str(seen["url"])
    assert "output_format=mp3_44100_64" in str(seen["url"])
    assert "eleven_flash_v2_5" in str(seen["model"])
    assert seen["api_key"] == "test-key"


def test_provider_maps_429_to_rate_limited() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "too many"})

    async def collect():
        return [c async for c in _provider(handler).stream_audio("Hi.")]

    with pytest.raises(TTSRateLimited):
        run(collect())


def test_provider_maps_500_to_tts_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    async def collect():
        return [c async for c in _provider(handler).stream_audio("Hi.")]

    with pytest.raises(TTSError, match="503"):
        run(collect())


def test_provider_wraps_transport_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    async def collect():
        return [c async for c in _provider(handler).stream_audio("Hi.")]

    with pytest.raises(TTSError, match="transport failure"):
        run(collect())


def test_provider_requires_key_and_voice() -> None:
    with pytest.raises(ValueError):
        ElevenLabsTTS(api_key="", voice_id="v")
    with pytest.raises(ValueError):
        ElevenLabsTTS(api_key="k", voice_id="")


# ---------------------------------------------------------------------------
# Queue behaviour
# ---------------------------------------------------------------------------


def test_unconfigured_queue_reports_a_reason_without_raising() -> None:
    """No provider must produce a flag, not an exception."""
    queue = TTSQueue(None)
    outcome = run(queue.synthesize("Anything."))
    assert outcome.ok is False
    assert outcome.reason == "tts_not_configured"
    assert outcome.reason is not None  # closed enum for the client


def test_successful_synthesis_returns_audio_and_content_type() -> None:
    provider = StubTTS()
    outcome = run(TTSQueue(provider).synthesize("Nelson's Column rises."))
    assert outcome.ok
    assert outcome.audio == FAKE_AUDIO
    assert outcome.content_type == "audio/mpeg"


def test_rate_limit_becomes_a_mapped_reason() -> None:
    provider = StubTTS(error=TTSRateLimited("429"))
    outcome = run(TTSQueue(provider).synthesize("Hi."))
    assert outcome.ok is False
    assert outcome.reason == "elevenlabs_rate_limited"


def test_empty_audio_body_is_a_failure_not_a_silent_success() -> None:
    """A 200 with an empty body would otherwise play as silence."""
    outcome = run(TTSQueue(StubTTS(chunks=())).synthesize("Hi."))
    assert outcome.ok is False
    assert outcome.reason == "elevenlabs_error"


def test_unexpected_provider_exception_never_escapes() -> None:
    outcome = run(TTSQueue(StubTTS(error=RuntimeError("boom"))).synthesize("Hi."))
    assert outcome.ok is False
    assert outcome.reason == "elevenlabs_error"


def test_synthesize_all_preserves_input_order() -> None:
    class Slowing(StubTTS):
        async def stream_audio(self, text: str):
            # First sentence is slowest: a naive gather would reorder output.
            await asyncio.sleep(0.05 if text == "first" else 0)
            async for chunk in super().stream_audio(text):
                yield chunk

    outcomes = run(TTSQueue(Slowing()).synthesize_all(["first", "second", "third"]))
    assert [o.text for o in outcomes] == ["first", "second", "third"]


def test_incremental_dispatch_starts_sentence_one_before_the_source_ends() -> None:
    """The whole point of the design: synthesize while the model still writes.

    The source only publishes its second sentence once the first is already
    being synthesised, so a batched implementation would deadlock rather than
    pass. The timeout turns that hang into a clear failure.
    """
    started = asyncio.Event()

    class Signaling(StubTTS):
        async def stream_audio(self, text: str):
            started.set()
            async for chunk in super().stream_audio(text):
                yield chunk

    async def source():
        yield "one"
        await asyncio.wait_for(started.wait(), timeout=5)
        yield "two"

    async def drain() -> list[str]:
        queue = TTSQueue(Signaling())
        return [o.text async for o in queue.synthesize_incremental(source())]

    assert run(asyncio.wait_for(drain(), timeout=5)) == ["one", "two"]


def test_incremental_dispatch_surfaces_a_source_failure() -> None:
    """A failing source must not silently truncate the stream.

    The feeder publishes its sentinel from a ``finally``, so an exception there
    would be stored on a task nobody reads -- the walker would simply get less
    narration than the model produced, with nothing logged.
    """
    provider = StubTTS()

    async def source():
        yield "one"
        raise RuntimeError("source exploded")

    async def drain() -> list[str]:
        seen: list[str] = []
        with pytest.raises(RuntimeError, match="source exploded"):
            async for outcome in TTSQueue(provider).synthesize_incremental(source()):
                seen.append(outcome.text)
        return seen

    # The sentence dispatched before the failure is still delivered.
    assert run(drain()) == ["one"]


def test_incremental_dispatch_emits_results_in_order() -> None:
    class Slowing(StubTTS):
        async def stream_audio(self, text: str):
            await asyncio.sleep(0.05 if text == "slow" else 0)
            async for chunk in super().stream_audio(text):
                yield chunk

    async def source():
        yield "slow"
        yield "fast"

    async def drain() -> list[str]:
        return [o.text async for o in TTSQueue(Slowing()).synthesize_incremental(source())]

    assert run(drain()) == ["slow", "fast"]


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_breaker_opens_after_consecutive_failures_and_stops_calling_out() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=2, reset_after=60.0, clock=lambda: now[0])
    provider = StubTTS(error=TTSRateLimited("429"))
    queue = TTSQueue(provider, breaker=breaker)

    assert run(queue.synthesize("a")).reason == "elevenlabs_rate_limited"
    assert run(queue.synthesize("b")).reason == "elevenlabs_rate_limited"
    assert breaker.is_open()

    # Third call is refused before any provider work happens.
    assert run(queue.synthesize("c")).reason == "circuit_open"
    assert provider.spoken == ["a", "b"]


def test_breaker_allows_one_probe_after_the_cool_down() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=1, reset_after=60.0, clock=lambda: now[0])
    provider = StubTTS()
    queue = TTSQueue(provider, breaker=breaker)

    breaker.record_failure()
    assert run(queue.synthesize("blocked")).reason == "circuit_open"

    now[0] = 61.0
    assert run(queue.synthesize("probe")).ok


def test_breaker_resets_after_a_success() -> None:
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure()
    breaker.record_success()
    assert breaker.failures == 0
    assert not breaker.is_open()


def test_stream_sentence_concatenates_first_chunk_then_the_rest() -> None:
    async def scenario() -> tuple[str, bytes]:
        # Everything in one loop: the returned generator belongs to the loop
        # that started the stream, exactly as it does in a request handler.
        attempt = await TTSQueue(
            StubTTS(chunks=(b"FIRST", b"SECOND"))
        ).stream_sentence("Hello.")
        assert attempt.ok
        assert attempt.chunks is not None
        body = b"".join([chunk async for chunk in attempt.chunks])
        return attempt.content_type or "", body

    content_type, body = run(scenario())
    assert content_type == "audio/mpeg"
    assert body == b"FIRSTSECOND"


def test_stream_sentence_reports_failure_before_any_bytes() -> None:
    """A failure must be known while the status code can still be changed."""
    attempt = run(TTSQueue(StubTTS(error=TTSRateLimited("429"))).stream_sentence("Hi."))
    assert attempt.ok is False
    assert attempt.reason == "elevenlabs_rate_limited"
    assert attempt.chunks is None


def test_stream_sentence_reports_unconfigured_and_open_breaker() -> None:
    assert run(TTSQueue(None).stream_sentence("Hi.")).reason == "tts_not_configured"

    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()
    attempt = run(TTSQueue(StubTTS(), breaker=breaker).stream_sentence("Hi."))
    assert attempt.reason == "circuit_open"


def test_stream_sentence_treats_an_empty_stream_as_a_failure() -> None:
    attempt = run(TTSQueue(StubTTS(chunks=())).stream_sentence("Hi."))
    assert attempt.ok is False
    assert attempt.reason == "elevenlabs_error"


def test_stream_failure_mid_sentence_is_raised_not_swallowed() -> None:
    """Truncating silently would be the same defect as a dropped sentence."""

    class DyingMidway(StubTTS):
        async def stream_audio(self, text: str):
            self.spoken.append(text)
            yield b"FIRST"
            raise TTSError("connection reset")

    async def scenario() -> bytes:
        attempt = await TTSQueue(DyingMidway()).stream_sentence("Hi.")
        assert attempt.chunks is not None
        return b"".join([chunk async for chunk in attempt.chunks])

    with pytest.raises(TTSError, match="connection reset"):
        run(scenario())


def test_stream_sentence_releases_its_concurrency_slot() -> None:
    """A leaked slot would starve every later sentence."""
    provider = StubTTS(chunks=(b"A",))
    queue = TTSQueue(provider, max_concurrency=1)

    async def scenario() -> int:
        attempt = await queue.stream_sentence("one")
        assert attempt.chunks is not None
        async for _ in attempt.chunks:
            pass
        # If the slot leaked, this second call would block forever.
        second = await queue.stream_sentence("two")
        return 1 if second.ok else 0

    assert run(asyncio.wait_for(scenario(), timeout=5)) == 1


def test_concurrency_is_bounded() -> None:
    """An unbounded fan-out fails under load rather than going faster."""
    in_flight = 0
    peak = 0

    class Counting(StubTTS):
        async def stream_audio(self, text: str):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            async for chunk in super().stream_audio(text):
                yield chunk

    queue = TTSQueue(Counting(), max_concurrency=2)
    run(queue.synthesize_all([f"s{i}" for i in range(6)]))
    assert peak <= 2
