"""Opaque, single-use handles to synthesised audio.

Implements section 6 of docs/audio-streaming.md.

``GET /api/audio/{token}`` spends real money per character, so the relay must
not be usable as an open text-to-speech proxy. Three properties keep that true:

* the token is an **opaque server-side key**, never the text;
* keys are **single-use and short-TTL**, so a leaked URL is worthless shortly
  after it was minted and worthless a second time immediately;
* keys are **bound to the session**, so one walker cannot consume another's
  synthesised sentence.

An unknown, expired or foreign token all fail the same way: ``None``, and the
route returns 404 **without** synthesising anything on demand.
"""

from __future__ import annotations

import secrets
import time
from collections import deque
from dataclasses import dataclass

DEFAULT_TTL_SECONDS = 120.0


@dataclass(frozen=True)
class AudioBlob:
    """One sentence's audio, plus the text it was synthesised from."""

    data: bytes
    content_type: str
    text: str


class AudioTokenStore:
    """In-memory token registry with TTL and session binding."""

    def __init__(self, *, ttl: float = DEFAULT_TTL_SECONDS, clock=time.monotonic) -> None:
        self._ttl = ttl
        self._clock = clock
        self._entries: dict[str, tuple[float, str, AudioBlob]] = {}
        self._order: deque[str] = deque()

    def issue(self, *, session_id: str, blob: AudioBlob) -> str:
        """Mint a single-use token for ``blob`` and return it."""
        self._purge()
        token = secrets.token_urlsafe(24)
        self._entries[token] = (self._clock() + self._ttl, session_id, blob)
        self._order.append(token)
        return token

    def redeem(self, token: str, *, session_id: str | None = None) -> AudioBlob | None:
        """Consume a token. Wrong session or expiry yields None; never re-synth."""
        self._purge()
        entry = self._entries.pop(token, None)
        if entry is None:
            return None
        expires_at, owner, blob = entry
        if self._clock() >= expires_at:
            return None
        if session_id is not None and owner != session_id:
            return None
        return blob

    def __len__(self) -> int:
        self._purge()
        return len(self._entries)

    def _purge(self) -> None:
        now = self._clock()
        while self._order:
            token = self._order[0]
            entry = self._entries.get(token)
            if entry is not None and entry[0] > now:
                break
            self._order.popleft()
            self._entries.pop(token, None)


__all__ = ["AudioBlob", "AudioTokenStore"]
