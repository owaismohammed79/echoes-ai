"""The LLM seam, and the null implementation of it.

Narration and Q&A depend on a one-method protocol, not on Gemma: neither module
needs to know which model is behind it, and both stay testable with a scripted
stub and no network.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol


class LLMProvider(Protocol):
    """Anything that can stream narration text."""

    def stream(self, *, system: str, user: str) -> AsyncIterator[str]: ...


class LLMError(RuntimeError):
    """The model could not be reached, or replied with an error.

    Defined at the seam rather than inside one provider so the narration
    orchestrator can reason about failures without importing a concrete SDK
    wrapper. ``app.services.gemma`` re-exports it, so existing importers of
    ``gemma.LLMError`` are unaffected.
    """


class LLMOutageError(LLMError):
    """A *transient* provider failure: HTTP 5xx, or a timed-out request.

    Distinct from a plain :class:`LLMError` because it is worth retrying -- the
    same provider may recover, or the configured fallback may still answer --
    while a 4xx or a malformed stream will fail the same way on every attempt.
    """


class LLMTemporaryOutage(LLMOutageError):
    """Every provider in the cascade failed inside the retry budget.

    Raised only by :class:`~app.services.gemma.FailoverLLMProvider`, after the
    cumulative budget is spent. The API layer turns it into the structured
    ``LLM_TEMPORARY_OUTAGE`` response the client shows a retry modal for.
    """

    code = "LLM_TEMPORARY_OUTAGE"
    message = "Gemma is experiencing high global traffic. Please try again shortly."

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(self.message)
        self.detail = detail


class NullLLMProvider:
    """Yields nothing, sending the orchestrator down its static-copy path.

    Used when no LLM key is configured: the walk stays available instead of
    returning 500s, and the deterministic NOTICE path does the honest thing
    rather than pretending a model is there.
    """

    async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
        return
        yield ""  # pragma: no cover - makes this an async generator


__all__ = [
    "LLMError",
    "LLMOutageError",
    "LLMProvider",
    "LLMTemporaryOutage",
    "NullLLMProvider",
]
