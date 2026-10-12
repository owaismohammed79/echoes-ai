"""Cost guardrails: per-IP walk limits and a global daily spend cap.

Implements AGENTS.md 4 ("Rate limits ... and a global daily spend cap for
ElevenLabs and the LLM API" / "If budgets are exceeded, seamlessly fall back to
a clearly labeled recorded demo mode").

Three deliberate properties:

* **Nothing here raises.** A guard returns a decision; the HTTP layer turns that
  decision into a status code or a demo response. Keeping the decision separate
  is what lets the spend cap *degrade* a walk instead of failing it, which is the
  whole point of a kill-switch.
* **Process-local, like the session store.** Two instances mean two limiters and
  two ledgers, so the real ceiling is twice the configured cap. `/api/health`
  advertises this rather than letting it be discovered on a credit card. The
  stores are injected, so moving to Redis is a new class, not a rewrite.
* **Prices are inputs, never claims.** The defaults below are placeholders whose
  only job is to make the cap proportional to real usage. Set them from the
  vendors' current price lists; `/api/health` reports whether you have.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Deque

logger = logging.getLogger("echoes.budgets")

__all__ = [
    "Admission",
    "Guardrails",
    "PerIPWalkLimiter",
    "Prices",
    "RateDecision",
    "SpendLedger",
    "utc_day",
]


def utc_day() -> date:
    """The current UTC date. The spend window is a calendar day, not a rolling
    one, because that is how vendor dashboards and budgets are read."""
    return datetime.now(timezone.utc).date()


# ---------------------------------------------------------------------------
# Per-IP walk limit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateDecision:
    """The outcome of one rate-limit check."""

    allowed: bool
    #: Seconds until the oldest hit ages out. Zero when allowed.
    retry_after_s: int = 0
    #: Requests still available in the window after this one.
    remaining: int = 0


class PerIPWalkLimiter:
    """A rolling window of walk generations, keyed by client identity.

    A sliding window rather than a fixed bucket: with a fixed hourly bucket a
    client can spend the whole allowance at 12:59 and the whole next allowance
    at 13:00, which is exactly the burst the limit exists to stop.
    """

    def __init__(
        self,
        *,
        limit: int,
        window_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = max(0, int(limit))
        self.window_s = float(window_s)
        self._clock = clock
        self._hits: dict[str, Deque[float]] = {}

    def check(self, identity: str) -> RateDecision:
        """Ask without spending."""
        self._trim(identity)
        hits = self._hits.get(identity)
        used = len(hits) if hits else 0
        if used >= self.limit:
            oldest = hits[0]  # type: ignore[index]
            return RateDecision(
                allowed=False,
                retry_after_s=max(1, int(self.window_s - (self._clock() - oldest)) + 1),
                remaining=0,
            )
        return RateDecision(allowed=True, remaining=self.limit - used)

    def consume(self, identity: str) -> RateDecision:
        """Spend one walk against the window."""
        self._trim(identity)
        hits = self._hits.setdefault(identity, deque())
        if len(hits) >= self.limit:
            oldest = hits[0]
            return RateDecision(
                allowed=False,
                retry_after_s=max(1, int(self.window_s - (self._clock() - oldest)) + 1),
                remaining=0,
            )
        hits.append(self._clock())
        return RateDecision(allowed=True, remaining=self.limit - len(hits))

    def peek(self, identity: str) -> int:
        """Walks used in the window, for diagnostics."""
        self._trim(identity)
        return len(self._hits.get(identity, ()))

    def reset(self) -> None:
        self._hits.clear()

    def _trim(self, identity: str) -> None:
        """Drop hits that have aged out, oldest first."""
        hits = self._hits.get(identity)
        if not hits:
            return
        cutoff = self._clock() - self.window_s
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if not hits:
            self._hits.pop(identity, None)


# ---------------------------------------------------------------------------
# Daily spend cap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prices:
    """Unit prices used to estimate spend.

    Defaults are placeholders, not quotations. ``configured`` records whether an
    operator supplied real numbers, and ``/api/health`` reports it, so the cap
    never silently runs on invented arithmetic.
    """

    tts_usd_per_1k_chars: float = 0.30
    llm_usd_per_1k_chars: float = 0.02
    configured: bool = False


@dataclass
class SpendLedger:
    """Estimated spend for the current UTC day, with a hard ceiling.

    The ceiling is checked *before* a walk, so the walk that crosses the cap
    still completes and the next one is served from the demo. A guard, not a
    billing boundary: it can overshoot by at most one walk's worth of spend.
    """

    cap_usd: float
    prices: Prices = field(default_factory=Prices)
    clock: Callable[[], date] = utc_day

    def __post_init__(self) -> None:
        self._day: date = self.clock()
        self._tts_chars = 0
        self._llm_chars = 0
        self._walks = 0
        self._demo_served = 0

    # -- accounting -------------------------------------------------------

    def record_tts(self, chars: int) -> None:
        """Characters actually sent to ElevenLabs. Exact, not estimated."""
        self._roll()
        if chars > 0:
            self._tts_chars += int(chars)

    def record_llm(self, chars: int) -> None:
        """Characters sent to and received from the model.

        Characters, not tokens: the provider seam does not surface usage, and
        inventing a token count would be worse than an approximation that is
        labelled as one.
        """
        self._roll()
        if chars > 0:
            self._llm_chars += int(chars)

    def record_walk(self) -> None:
        self._roll()
        self._walks += 1

    def record_demo(self) -> None:
        """A walk served from the demo. Counted, because it is the signal that
        the kill-switch fired and someone should raise the cap."""
        self._roll()
        self._demo_served += 1

    # -- state ------------------------------------------------------------

    @property
    def tts_usd(self) -> float:
        return self._tts_chars / 1000.0 * self.prices.tts_usd_per_1k_chars

    @property
    def llm_usd(self) -> float:
        return self._llm_chars / 1000.0 * self.prices.llm_usd_per_1k_chars

    @property
    def spent_usd(self) -> float:
        return self.tts_usd + self.llm_usd

    @property
    def exhausted(self) -> bool:
        """True once the day's budget is spent."""
        self._roll()
        return self.cap_usd > 0 and self.spent_usd >= self.cap_usd

    def snapshot(self) -> dict[str, object]:
        """Everything an operator needs, for ``/api/health``."""
        self._roll()
        return {
            "cap_usd": round(self.cap_usd, 4),
            "spent_usd": round(self.spent_usd, 4),
            "exhausted": self.exhausted,
            "prices_configured": self.prices.configured,
            "tts_chars": self._tts_chars,
            "llm_chars": self._llm_chars,
            "walks": self._walks,
            "demo_walks": self._demo_served,
            "day": self._day.isoformat(),
        }

    def _roll(self) -> None:
        """Reset at the UTC day boundary."""
        today = self.clock()
        if today != self._day:
            logger.info(
                "spend ledger rolling over %s -> %s (spent %.4f USD)",
                self._day,
                today,
                self.spent_usd,
            )
            self._day = today
            self._tts_chars = 0
            self._llm_chars = 0
            self._walks = 0
            self._demo_served = 0


# ---------------------------------------------------------------------------
# The decision the HTTP layer acts on
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Admission:
    """Why a request is being served, in one object."""

    #: False means refuse with 429; nothing was generated or spent.
    allowed: bool
    #: True means serve the recorded demo, whatever the spend state.
    demo: bool
    #: Seconds until the limit frees up. Only meaningful when not allowed.
    retry_after_s: int = 0
    #: Walks left in the window, for the X-RateLimit-* headers.
    walks_remaining: int = 0
    #: Short machine-readable why, for logs and the X-Demo-Mode reason.
    reason: str | None = None


class Guardrails:
    """The rate limiter and the ledger, behind one decision."""

    def __init__(
        self,
        *,
        limiter: PerIPWalkLimiter,
        ledger: SpendLedger,
        force_demo: bool = False,
    ) -> None:
        self.limiter = limiter
        self.ledger = ledger
        self.force_demo = force_demo

    @property
    def demo_mode(self) -> bool:
        """True when *every* request must be served from the demo."""
        return self.force_demo or self.ledger.exhausted

    def demo_reason(self) -> str | None:
        """Why the deployment is in demo mode, or None when it is not.

        A pure question, so ``/api/health`` can ask it without moving a counter.
        """
        if self.force_demo:
            return "demo_forced"
        if self.ledger.exhausted:
            return "spend_cap"
        return None

    def admit(self, identity: str, *, enforce_walk_limit: bool = True) -> Admission:
        """Decide how to serve one request, without spending anything.

        The spend cap is global and applies to every paid path; the per-IP
        allowance is specifically about *generating walks*, so push-to-talk opts
        out of it with ``enforce_walk_limit=False`` and still respects the cap.
        """
        reason = self.demo_reason()
        if reason is not None:
            if reason == "spend_cap":
                logger.warning(
                    "daily spend cap reached (%.4f USD of %.4f); serving demo mode",
                    self.ledger.spent_usd,
                    self.ledger.cap_usd,
                )
            return Admission(
                allowed=True,
                demo=True,
                walks_remaining=max(
                    0, self.limiter.limit - self.limiter.peek(identity)
                ),
                reason=reason,
            )

        if not enforce_walk_limit:
            return Admission(
                allowed=True, demo=False, walks_remaining=self.limiter.limit
            )

        decision = self.limiter.check(identity)
        if not decision.allowed:
            return Admission(
                allowed=False,
                demo=False,
                retry_after_s=decision.retry_after_s,
                walks_remaining=0,
                reason="rate_limited",
            )
        return Admission(
            allowed=True, demo=False, walks_remaining=decision.remaining
        )

    def begin_walk(self, identity: str) -> RateDecision:
        """Spend one walk against the window. Call only when generating."""
        decision = self.limiter.consume(identity)
        if decision.allowed:
            self.ledger.record_walk()
        return decision
