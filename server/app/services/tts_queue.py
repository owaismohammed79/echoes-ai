"""Bounded, order-preserving synthesis dispatch.

Implements sections 5 and 7 of docs/audio-streaming.md.

Two responsibilities, both about *when* synthesis happens rather than how:

* :class:`TTSQueue` dispatches each validated sentence as soon as it exists and
  yields outcomes **in input order**, so a fast sentence 2 never overtakes
  sentence 1.
* :class:`CircuitBreaker` stops a struggling provider from being hammered by
  every walker at once, and is what makes the proactive ``use_native_tts``
  fallback possible: the server knows before the client asks.

Concurrency is bounded by a semaphore sized just below the plan limit. Over
HTTP every in-flight request counts fully against ElevenLabs' concurrency, so an
unbounded fan-out would fail under load rather than go faster.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

from app.services.tts import ElevenLabsTTS, TTSError, TTSRateLimited

logger = logging.getLogger("echoes.speech.queue")

DEFAULT_MAX_CONCURRENCY = 2
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_RESET_AFTER_SECONDS = 60.0


class CircuitBreaker:
    """Trip after N consecutive failures; allow one probe after a cool-down.

    The clock is injectable so tests do not sleep.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        reset_after: float = DEFAULT_RESET_AFTER_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._reset_after = reset_after
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def failures(self) -> int:
        return self._failures

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if self._clock() - self._opened_at >= self._reset_after:
            # Cool-down elapsed: half-open. One probe is allowed through, and the
            # stale open timestamp is dropped here so a *successful* probe leaves
            # the breaker fully closed rather than one failure away from
            # re-opening on the same trip.
            self._opened_at = None
            return False
        return True

    def record_success(self) -> None:
        """Close the breaker on any usable provider response.

        A 200 with valid audio bytes is proof ElevenLabs is healthy again, so
        both the consecutive-failure count and the open timestamp are cleared.
        Without the count reset a single transient hiccup would keep the breaker
        re-arming on the next blip, and the walk would stay pinned to native TTS
        instead of returning to the real voice.
        """
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = self._clock()
            logger.warning("TTS circuit opened after %d failures", self._failures)


@dataclass(frozen=True)
class SynthesisOutcome:
    """Result of synthesizing one sentence."""

    text: str
    audio: bytes | None = None
    content_type: str | None = None
    reason: str | None = None  # closed enum; None means success
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.audio is not None


@dataclass(frozen=True)
class StreamAttempt:
    """A started stream, or the reason it could not start.

    Exactly one of ``chunks`` / ``reason`` is set. The split exists so the HTTP
    layer can decide the ``use_native_tts`` flag *before* it commits to a
    status and headers -- once a streaming response has begun, the flag can no
    longer be sent.
    """

    chunks: AsyncIterator[bytes] | None = None
    content_type: str | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.chunks is not None


class TTSQueue:
    """Dispatches sentences immediately, yields results in order.

    Each sentence's audio is buffered in full before the semaphore is released.
    That is deliberate: a sentence is a couple of dozen kilobytes, and holding
    the slot until the body is read is what makes the concurrency accounting
    honest. It also makes the audio available for the token store.
    """

    def __init__(
        self,
        tts: ElevenLabsTTS | None,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._tts = tts
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        self._breaker = breaker or CircuitBreaker()

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def configured(self) -> bool:
        """True when a provider is wired up at all."""
        return self._tts is not None

    def _unavailable_reason(self) -> str | None:
        """Why synthesis is impossible right now, or None if it may proceed."""
        if self._tts is None:
            return "tts_not_configured"
        if self._breaker.is_open():
            return "circuit_open"
        return None

    async def synthesize(self, text: str) -> SynthesisOutcome:
        """Synthesize one sentence, never raising.

        Every failure mode becomes ``reason`` on the outcome, because the only
        correct response to a TTS problem is a flag in the response telling the
        client to use ``window.speechSynthesis``.
        """
        if (reason := self._unavailable_reason()) is not None:
            return SynthesisOutcome(text=text, reason=reason)

        async with self._sem:
            # Re-check inside the slot: the breaker may have opened while queued.
            if self._breaker.is_open():
                return SynthesisOutcome(text=text, reason="circuit_open")
            try:
                chunks = [
                    chunk
                    async for chunk in self._tts.stream_audio(text)  # type: ignore[union-attr]
                ]
            except TTSRateLimited as exc:
                self._breaker.record_failure()
                return SynthesisOutcome(
                    text=text, reason="elevenlabs_rate_limited", detail=str(exc)
                )
            except TTSError as exc:
                self._breaker.record_failure()
                return SynthesisOutcome(
                    text=text, reason="elevenlabs_error", detail=str(exc)
                )
            except Exception as exc:  # noqa: BLE001 - must never break a walk
                self._breaker.record_failure()
                logger.warning("unexpected TTS failure: %s", exc)
                return SynthesisOutcome(
                    text=text,
                    reason="elevenlabs_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )

            audio = b"".join(chunks)
            if not audio:
                self._breaker.record_failure()
                return SynthesisOutcome(
                    text=text, reason="elevenlabs_error", detail="empty audio body"
                )

            self._breaker.record_success()
            return SynthesisOutcome(
                text=text, audio=audio, content_type=self._tts.content_type
            )

    async def stream_sentence(self, text: str) -> StreamAttempt:
        """Begin streaming one sentence, deciding all failures before any bytes.

        The first chunk is awaited here so that a rate limit, a 5xx or a dead
        provider is discovered while the response can still be a JSON fallback
        carrying ``use_native_tts: true``. Once headers are flushed that is no
        longer possible, which is why this pre-flight step exists rather than
        simply handing the generator to the response.

        A failure *after* the first chunk cannot change the status code; the
        stream ends early instead, and the client falls back using the answer
        text it already received in the headers (docs/audio-streaming.md 9).
        """
        if (reason := self._unavailable_reason()) is not None:
            return StreamAttempt(reason=reason)

        await self._sem.acquire()
        # Re-check inside the slot: the breaker may have opened while queued.
        if self._breaker.is_open():
            self._sem.release()
            return StreamAttempt(reason="circuit_open")

        try:
            iterator = self._tts.stream_audio(text).__aiter__()  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - must never break a walk
            self._sem.release()
            self._breaker.record_failure()
            logger.warning("could not start TTS stream: %s", exc)
            return StreamAttempt(reason="elevenlabs_error")

        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            self._sem.release()
            self._breaker.record_failure()
            return StreamAttempt(reason="elevenlabs_error")
        except TTSRateLimited as exc:
            self._sem.release()
            self._breaker.record_failure()
            logger.info("TTS rate limited before first chunk: %s", exc)
            return StreamAttempt(reason="elevenlabs_rate_limited")
        except TTSError as exc:
            self._sem.release()
            self._breaker.record_failure()
            logger.warning("TTS failed before first chunk: %s", exc)
            return StreamAttempt(reason="elevenlabs_error")
        except Exception as exc:  # noqa: BLE001 - must never break a walk
            self._sem.release()
            self._breaker.record_failure()
            logger.warning("unexpected TTS failure: %s", exc)
            return StreamAttempt(reason="elevenlabs_error")

        self._breaker.record_success()
        return StreamAttempt(
            chunks=self._continue_stream(iterator, first),
            content_type=self._tts.content_type,  # type: ignore[union-attr]
        )

    async def _continue_stream(
        self, iterator: AsyncIterator[bytes], first: bytes
    ) -> AsyncIterator[bytes]:
        """Yield the buffered first chunk, then the rest, holding the slot."""
        try:
            yield first
            async for chunk in iterator:
                yield chunk
        except (TTSRateLimited, TTSError) as exc:
            self._breaker.record_failure()
            logger.warning("TTS stream failed mid-sentence: %s", exc)
            raise
        finally:
            self._sem.release()

    async def synthesize_all(self, texts: Sequence[str]) -> list[SynthesisOutcome]:
        """Dispatch every text, return outcomes in input order."""
        if not texts:
            return []
        tasks = [asyncio.create_task(self.synthesize(text)) for text in texts]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            raise

    async def synthesize_incremental(
        self, texts: AsyncIterator[str]
    ) -> AsyncIterator[SynthesisOutcome]:
        """Dispatch as each text arrives; yield outcomes in arrival order.

        Sentence 1 is synthesised while the model is still writing sentence 2.
        A feeder task pulls from ``texts`` so the consumer is never blocked
        waiting for the source, and results are emitted strictly in order.
        """
        queue: asyncio.Queue[asyncio.Task | None] = asyncio.Queue()
        done = object()

        async def feed() -> None:
            try:
                async for text in texts:
                    await queue.put(asyncio.create_task(self.synthesize(text)))
            finally:
                await queue.put(done)  # type: ignore[arg-type]

        feeder = asyncio.create_task(feed())
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                yield await item  # type: ignore[misc]
        except BaseException:
            # Consumer closed early (barge-in). Stop pulling from the source.
            feeder.cancel()
            raise

        # The source is exhausted. Awaiting the feeder propagates a source
        # failure instead of truncating the stream silently: because ``feed``
        # publishes its sentinel from a ``finally``, an exception would
        # otherwise be stored on a task nobody ever reads.
        await feeder


__all__ = [
    "CircuitBreaker",
    "StreamAttempt",
    "SynthesisOutcome",
    "TTSQueue",
]
