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


class NullLLMProvider:
    """Yields nothing, sending the orchestrator down its static-copy path.

    Used when no LLM key is configured: the walk stays available instead of
    returning 500s, and the deterministic NOTICE path does the honest thing
    rather than pretending a model is there.
    """

    async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
        return
        yield ""  # pragma: no cover - makes this an async generator


__all__ = ["LLMProvider", "NullLLMProvider"]
