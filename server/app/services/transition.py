"""The bridge line spoken between an answer and the walk it interrupted.

One fixed phrase, synthesised once and cached for the process. It exists because
the client must not be able to ask the TTS provider for arbitrary text — that is
an open text-to-speech proxy, and ``audio.py`` refuses to be one — yet a resumed
walk still wants a spoken bridge in the *same* voice rather than the browser's.

Nothing here is user-controlled: the text is a module constant, so the endpoint
that exposes it cannot be turned into a synthesis service.
"""

from __future__ import annotations

import logging

from app.services.tts_queue import SynthesisOutcome, TTSQueue

logger = logging.getLogger("echoes.speech.transition")

#: The line. Fixed on the server, so the client can only ever ask for *this*.
TRANSITION_PHRASE = "Now, continuing our walk."

#: Keyed by the phrase. The audio is byte-identical every time, so it is
#: synthesised once per process and replayed for every resume after that.
_CACHE: dict[str, SynthesisOutcome] = {}


async def transition_audio(tts_queue: TTSQueue) -> SynthesisOutcome:
    """The bridge's audio, from cache when we have it.

    Only successes are cached. A failure is a fact about *now* — a rate limit, an
    open breaker — not about the phrase, and must be retried on the next resume.
    """
    cached = _CACHE.get(TRANSITION_PHRASE)
    if cached is not None:
        return cached

    outcome = await tts_queue.synthesize(TRANSITION_PHRASE)
    if outcome.ok:
        _CACHE[TRANSITION_PHRASE] = outcome
    return outcome


def clear_cache() -> None:
    """Drop the cached bridge. For tests, and for a provider change."""
    _CACHE.clear()


__all__ = ["TRANSITION_PHRASE", "clear_cache", "transition_audio"]
