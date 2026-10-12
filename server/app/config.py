"""Environment-driven configuration for the Echoes backend.

Everything that varies between deployments lives here, so no other module
reaches for ``os.getenv`` on its own. Values are read once at import time: a
value change means a restart, not a hot reload.

Endpoints and search bounds are declared here but bound into the module that
uses them (``services.wikipedia``, ``services.parks``), which keeps them
patchable in tests.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

logger = logging.getLogger("echoes.config")

#: Load ``server/.env`` for local runs, resolved from this file rather than the
#: working directory so ``python main.py`` behaves the same from anywhere.
#: ``override=False`` (the default) means a real environment variable always
#: wins, so a deployed instance is unaffected and a stray local file can never
#: shadow a dashboard secret.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# Network budget
# ---------------------------------------------------------------------------

#: Hard ceiling for any single external API call. PRD section 5 requires
#: time-to-first-audio-byte < 2.5s, so a 3.0s network bound keeps the pipeline
#: inside budget even when an upstream stalls.
EXTERNAL_TIMEOUT_SECONDS: Final[float] = 3.0

#: Connection-pool sizing for the shared client. The pipeline fans out to
#: Wikipedia, SerpApi, Gemma and ElevenLabs, so the pool must outlive a single
#: request without being unbounded.
MAX_CONNECTIONS: Final[int] = 20
MAX_KEEPALIVE_CONNECTIONS: Final[int] = 10

# ---------------------------------------------------------------------------
# Context aggregation
# ---------------------------------------------------------------------------

WIKIPEDIA_ENDPOINT: Final[str] = "https://en.wikipedia.org/w/api.php"
SERPAPI_ENDPOINT: Final[str] = "https://serpapi.com/search"

#: Open-Meteo: keyless, so weather adds no secret and no quota to the
#: kill-switch's accounting. It is the named provider docs/llm-orchestration.md
#: section 6 requires before weather may be supplied to a walk at all.
WEATHER_ENDPOINT: Final[str] = "https://api.open-meteo.com/v1/forecast"

#: The radius a walk searches first. Tight on purpose: it is what makes the
#: narration hyper-local, and it is cheap to widen afterwards.
SEARCH_RADIUS_M: Final[int] = 1500

#: Second attempt when the first radius returns too little. A residential
#: street can be empty inside 1.5km while the same neighbourhood is dense with
#: history at 5km (measured in Bengaluru: 2 surviving facts at 1.5km, 8 at 5km).
SEARCH_RADIUS_EXPANDED_M: Final[int] = 5_000

#: Hard ceiling for the widening ladder, and also the ceiling MediaWiki's
#: coordinate-scoped search accepts, so the city-history fallback cannot reach
#: further than this either.
SEARCH_RADIUS_MAX_M: Final[int] = 10_000

MAX_RESULTS_PER_SOURCE: Final[int] = 20

#: Facts a walk needs before it is worth narrating a story. Fewer than this and
#: the aggregator widens the search (or falls back to city history) rather than
#: hand Gemma two thin facts to frame a walk around. Below this after every
#: fallback, the walk still narrates -- it is just a quieter one -- so this is a
#: retrieval target, not a hard requirement (AGENTS.md 1 keeps that to NOTICE).
MIN_CONTEXT_FACTS: Final[int] = 5

#: The term for the city-history fallback (PRD 3): a coordinate-scoped search
#: for "history" returns the city's own articles when the street has none.
WIKIPEDIA_HISTORY_QUERY: Final[str] = "history"

#: Wikimedia's API rejects requests without a descriptive User-Agent with
#: HTTP 403. This is a hard requirement, not a nicety.
USER_AGENT: Final[str] = os.getenv(
    "ECHOES_USER_AGENT",
    "Echoes/0.1 (voice-first walking companion; contact@example.org)",
)

#: SerpApi key (Render secrets / local .env). An unset key is not fatal: the
#: park source simply contributes nothing and the walk still runs.
SERPAPI_KEY: Final[str] = os.getenv("SERPAPI_KEY", "")

# ---------------------------------------------------------------------------
# Speech services
# ---------------------------------------------------------------------------

#: Bounded synthesis fan-out per walk (docs/audio-streaming.md 5). Over HTTP
#: every in-flight request occupies a provider concurrency slot for its whole
#: duration, so this is a cost control, not just a politeness setting.
TTS_MAX_CONCURRENCY: Final[int] = int(os.getenv("TTS_MAX_CONCURRENCY", "2"))

# ---------------------------------------------------------------------------
# Local vs deployed
# ---------------------------------------------------------------------------

#: Render sets ``RENDER=true`` on every service it runs; anything else is treated
#: as a local run. Used only to pick a safe default for the voice below.
IS_DEPLOYED: Final[bool] = os.getenv("RENDER", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

#: Force the browser voice even when ElevenLabs *is* configured. Defaults on for
#: a local run and off in production: a laptop must not spend the free-tier
#: credits on every walk it narrates, while a deployed instance exists to use the
#: real voice. Either side can be overridden with the env var.
FORCE_NATIVE_TTS: Final[bool] = os.getenv(
    "ECHOES_FORCE_NATIVE_TTS", "0" if IS_DEPLOYED else "1"
).strip().lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Cost guardrails (AGENTS.md 4)
# ---------------------------------------------------------------------------

#: Walk generations allowed per client in the window below. The client is an IP,
#: resolved through the trusted proxy chain rather than taken from the socket.
WALK_LIMIT_PER_HOUR: Final[int] = int(
    os.getenv("ECHOES_WALK_LIMIT_PER_HOUR", "5")
)

#: The rolling window the walk limit is measured over.
WALK_LIMIT_WINDOW_SECONDS: Final[float] = float(
    os.getenv("ECHOES_WALK_LIMIT_WINDOW_SECONDS", "3600")
)

#: How many proxies append to ``X-Forwarded-For`` in front of this app.
#: Render's edge is exactly one, so the client is the last entry in the chain
#: and a forged prefix before it is ignored. Zero trusts the socket alone, which
#: is correct only when nothing sits in front of the app.
TRUSTED_PROXY_HOPS: Final[int] = int(
    os.getenv("ECHOES_TRUSTED_PROXY_HOPS", "1")
)

#: Estimated spend ceiling per UTC day, across ElevenLabs and the LLM. Exceeding
#: it does not fail a walk: it switches the deployment to the recorded demo.
#: Zero disables the cap, which is only sensible in development.
DAILY_SPEND_CAP_USD: Final[float] = float(
    os.getenv("ECHOES_DAILY_SPEND_CAP_USD", "5.00")
)

#: Manual kill-switch. Set to 1 to serve the demo regardless of spend -- for a
#: demo day, a venue with no wifi, or an incident.
DEMO_MODE_FORCED: Final[bool] = os.getenv(
    "ECHOES_FORCE_DEMO_MODE", ""
).strip().lower() in {"1", "true", "yes", "on"}


def _price(name: str, default: float) -> tuple[float, bool]:
    """Read a unit price, and record whether it was actually configured.

    The defaults are placeholders, not quotations: they exist so the cap scales
    with real usage rather than sitting inert. ``/api/health`` reports whether
    real numbers were supplied, so a cap never silently runs on invented
    arithmetic. A malformed value falls back to the default and is logged.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default, False
    try:
        return float(raw), True
    except ValueError:
        logger.warning(
            "%s=%r is not a number; falling back to the placeholder %s",
            name,
            raw,
            default,
        )
        return default, False


#: ElevenLabs is billed per character, so this is the meaningful unit.
TTS_PRICE_USD_PER_1K_CHARS, TTS_PRICE_CONFIGURED = _price(
    "ECHOES_TTS_PRICE_USD_PER_1K_CHARS", 0.30
)

#: The LLM is billed per token. The provider seam does not surface usage, so the
#: ledger counts characters and this price is applied to them; label the
#: resulting figure an estimate, never a bill.
LLM_PRICE_USD_PER_1K_CHARS, LLM_PRICE_CONFIGURED = _price(
    "ECHOES_LLM_PRICE_USD_PER_1K_CHARS", 0.02
)

PRICES_CONFIGURED: Final[bool] = TTS_PRICE_CONFIGURED and LLM_PRICE_CONFIGURED

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS: Final[tuple[str, ...]] = tuple(
    origin.strip()
    for origin in os.getenv("ECHOES_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
)

#: The audio endpoints carry transcript, answer and citation metadata in
#: headers, so a cross-origin frontend cannot read a single one of them unless
#: the names are exposed explicitly.
EXPOSED_HEADERS: Final[tuple[str, ...]] = (
    "X-Use-Native-Tts",
    "use_native_tts",
    "X-Fallback-Reason",
    "X-Transcript",
    "X-Answer-Text",
    "X-Answer-Citations",
    "X-Answer-Sources",
    "X-Answer-Refused",
    "X-Refusal-Reason",
    "X-Timings-Ms",
    # The bridge line, so a client whose remote voice is down can still speak the
    # same words it would have heard.
    "X-Transition-Text",
    # The kill-switch and its quota accounting have to be readable by a browser,
    # or a demo walk is indistinguishable from a live one on the client.
    "X-Demo-Mode",
    "X-Demo-Reason",
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "Retry-After",
)

__all__ = [
    "ALLOWED_ORIGINS",
    "DAILY_SPEND_CAP_USD",
    "DEMO_MODE_FORCED",
    "EXPOSED_HEADERS",
    "EXTERNAL_TIMEOUT_SECONDS",
    "FORCE_NATIVE_TTS",
    "IS_DEPLOYED",
    "LLM_PRICE_USD_PER_1K_CHARS",
    "MAX_CONNECTIONS",
    "MAX_KEEPALIVE_CONNECTIONS",
    "MAX_RESULTS_PER_SOURCE",
    "MIN_CONTEXT_FACTS",
    "PRICES_CONFIGURED",
    "SEARCH_RADIUS_EXPANDED_M",
    "SEARCH_RADIUS_M",
    "SEARCH_RADIUS_MAX_M",
    "SERPAPI_ENDPOINT",
    "SERPAPI_KEY",
    "TRUSTED_PROXY_HOPS",
    "TTS_MAX_CONCURRENCY",
    "TTS_PRICE_USD_PER_1K_CHARS",
    "USER_AGENT",
    "WALK_LIMIT_PER_HOUR",
    "WALK_LIMIT_WINDOW_SECONDS",
    "WEATHER_ENDPOINT",
    "WIKIPEDIA_ENDPOINT",
    "WIKIPEDIA_HISTORY_QUERY",
]
