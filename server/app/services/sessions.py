"""Per-walk session state: cached facts, conversation history, staleness.

Implements section 11 of docs/audio-streaming.md.

The store is deliberately process-local. That is correct for one instance and
silently wrong for two -- a second instance means a Push-To-Talk request can
land where the session is unknown and the walker hears a refusal for no reason
they can see. ``/api/health`` advertises the limitation rather than letting it
be discovered in production, and section 11 says to fail loudly rather than
quietly degrade.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.services.geo import haversine_m

logger = logging.getLogger("echoes.sessions")

#: Longer than the longest walk (45 minutes) plus a question after it.
SESSION_TTL_SECONDS = 3600.0

#: Beyond this distance from where the context was built, the facts no longer
#: describe where the walker is. Answering confidently about a landmark behind
#: them is a grounding failure even though every citation is valid.
STALE_CONTEXT_METRES = 1500.0

MAX_HISTORY_TURNS = 4


@dataclass
class WalkSession:
    """One walk: the facts it was built on, and what has been said since."""

    session_id: str
    facts: list[Any] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    anchor: tuple[float, float] | None = None
    created_at: float = 0.0
    #: Refreshed on every continuation so a 30-minute walk is not purged at 15.
    last_seen: float = 0.0
    #: The validated narration the walk spoke, so a follow-up question can
    #: resolve "what was that about the temple?" against the landmarks the model
    #: actually just talked about (docs/audio-streaming.md 10.2).
    narration_text: str = ""
    #: Walker's chosen length, in minutes. ``None`` is the one-shot path.
    duration_min: int | None = None
    mood: str | None = None
    #: How many cycles have already been generated (the next one to serve).
    cycle_index: int = 0
    #: Running sentence index so Channel B tokens never collide across cycles.
    sentence_index: int = 0
    used_fact_ids: list[str] = field(default_factory=list)
    news: list[Any] = field(default_factory=list)
    weather: Any = None
    last_serp_at: float = 0.0
    #: The generic fallback copy is spoken at most once per walk: a model outage
    #: must not make every cycle repeat the same lines.
    static_served: bool = False

    def remember(self, role: str, text: str) -> None:
        """Record one turn, keeping only the most recent few.

        Always stores the *validated* text -- if the validator dropped the
        model's sentence and substituted a refusal, the history must show the
        refusal, or the model is primed by its own hallucination next turn
        (docs/audio-streaming.md 10.2).
        """
        if not text:
            return
        self.history.append((role, text))
        self.history = self.history[-(MAX_HISTORY_TURNS * 2) :]


class SessionStore:
    """Process-local session cache with TTL."""

    def __init__(self, *, ttl: float = SESSION_TTL_SECONDS, clock=time.monotonic):
        self._ttl = ttl
        self._clock = clock
        self._sessions: dict[str, WalkSession] = {}

    def now(self) -> float:
        """The store's clock. Walk expiry and TTL both read this."""
        return self._clock()

    def create(
        self,
        *,
        facts: Sequence[Any],
        anchor: tuple[float, float] | None = None,
        session_id: str | None = None,
        duration_min: int | None = None,
        mood: str | None = None,
        news: Sequence[Any] = (),
        weather: Any = None,
    ) -> WalkSession:
        self._purge()
        stamp = self._clock()
        session = WalkSession(
            session_id=session_id or secrets.token_urlsafe(16),
            facts=list(facts),
            anchor=anchor,
            created_at=stamp,
            last_seen=stamp,
            duration_min=duration_min,
            mood=mood,
            news=list(news),
            weather=weather,
            last_serp_at=stamp,
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str | None) -> WalkSession | None:
        self._purge()
        if not session_id:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self._clock() - session.last_seen >= self._ttl:
            self._sessions.pop(session_id, None)
            return None
        return session

    def touch(self, session: WalkSession) -> None:
        """Keep a live walk from expiring mid-route."""
        session.last_seen = self._clock()

    def __len__(self) -> int:
        self._purge()
        return len(self._sessions)

    def _purge(self) -> None:
        now = self._clock()
        for key, session in list(self._sessions.items()):
            stamp = session.last_seen or session.created_at
            if now - stamp >= self._ttl:
                self._sessions.pop(key, None)


def context_is_stale(
    session: WalkSession, lat: float | None, lon: float | None
) -> bool:
    """True when the walker has moved beyond the context's useful radius.

    Answering confidently about a landmark behind them is a grounding failure
    even when every citation is technically valid
    (docs/audio-streaming.md 10.3).
    """
    if lat is None or lon is None or session.anchor is None:
        return False
    anchor_lat, anchor_lon = session.anchor
    moved = haversine_m(anchor_lat, anchor_lon, lat, lon)
    if moved > STALE_CONTEXT_METRES:
        logger.info("session context stale: walker moved %.0fm", moved)
        return True
    return False


__all__ = [
    "MAX_HISTORY_TURNS",
    "SESSION_TTL_SECONDS",
    "STALE_CONTEXT_METRES",
    "SessionStore",
    "WalkSession",
    "context_is_stale",
]
