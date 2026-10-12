"""The application's service container and its FastAPI dependency.

Everything the request handlers need -- the shared HTTP client, the LLM, the
transcriber, the synthesis queue and the two process-local stores -- is built
once during startup and handed over as a single typed object.

This replaces scattering ``getattr(request.app.state, "llm", None)`` through the
routes: a missing dependency is now a construction error at startup, not a
``None`` that silently disables a feature mid-walk.
"""

from __future__ import annotations

from typing import Annotated, NamedTuple

import httpx
from fastapi import Depends, Request

from app.config import (
    DAILY_SPEND_CAP_USD,
    DEMO_MODE_FORCED,
    EXTERNAL_TIMEOUT_SECONDS,
    LLM_PRICE_USD_PER_1K_CHARS,
    MAX_CONNECTIONS,
    MAX_KEEPALIVE_CONNECTIONS,
    PRICES_CONFIGURED,
    TTS_PRICE_USD_PER_1K_CHARS,
    USER_AGENT,
    WALK_LIMIT_PER_HOUR,
    WALK_LIMIT_WINDOW_SECONDS,
)
from app.services.audio_tokens import AudioTokenStore
from app.services.budgets import Guardrails, PerIPWalkLimiter, Prices, SpendLedger
from app.services.factories import (
    build_llm_provider,
    build_stt_provider,
    build_tts_queue,
)
from app.services.providers import LLMProvider
from app.services.sessions import SessionStore
from app.services.stt import GroqWhisper
from app.services.tts_queue import TTSQueue


class Container(NamedTuple):
    """Long-lived collaborators, one set per process."""

    http: httpx.AsyncClient
    llm: LLMProvider | None
    stt: GroqWhisper | None
    tts_queue: TTSQueue
    sessions: SessionStore
    audio_tokens: AudioTokenStore
    guardrails: Guardrails


def create_http_client() -> httpx.AsyncClient:
    """One connection pool for the process lifetime.

    Wikipedia 403s without a descriptive User-Agent, so it is set here rather
    than per-call.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(EXTERNAL_TIMEOUT_SECONDS),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
        ),
    )


def build_guardrails() -> Guardrails:
    """The per-IP limiter and the day's spend ledger, from configuration.

    Process-local, exactly like the session store above: two instances mean two
    allow counters and two ceilings, so the effective caps are the configured
    ones times the instance count. Injecting the stores means a Redis-backed
    limiter is a new class rather than a rewrite.
    """
    return Guardrails(
        limiter=PerIPWalkLimiter(
            limit=WALK_LIMIT_PER_HOUR, window_s=WALK_LIMIT_WINDOW_SECONDS
        ),
        ledger=SpendLedger(
            cap_usd=DAILY_SPEND_CAP_USD,
            prices=Prices(
                tts_usd_per_1k_chars=TTS_PRICE_USD_PER_1K_CHARS,
                llm_usd_per_1k_chars=LLM_PRICE_USD_PER_1K_CHARS,
                configured=PRICES_CONFIGURED,
            ),
        ),
        force_demo=DEMO_MODE_FORCED,
    )


def build_container(client: httpx.AsyncClient) -> Container:
    """Wire every provider, degrading to unconfigured rather than failing.

    Session state is process-local (docs/audio-streaming.md 11): correct for one
    instance and silently wrong for two, which ``/api/health`` advertises.
    """
    return Container(
        http=client,
        llm=build_llm_provider(),
        stt=build_stt_provider(client),
        tts_queue=build_tts_queue(client),
        sessions=SessionStore(),
        audio_tokens=AudioTokenStore(),
        guardrails=build_guardrails(),
    )


def get_container(request: Request) -> Container:
    """FastAPI dependency: the process-wide container."""
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - only reachable without lifespan
        raise RuntimeError(
            "application container is not initialised; is the lifespan running?"
        )
    return container


#: Handler signature shorthand: ``container: ContainerDep``.
ContainerDep = Annotated[Container, Depends(get_container)]


__all__ = [
    "Container",
    "ContainerDep",
    "build_container",
    "build_guardrails",
    "create_http_client",
    "get_container",
]
