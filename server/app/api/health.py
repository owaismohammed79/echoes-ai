"""``GET /api/health`` -- liveness and configuration.

Reports which integrations are configured and, deliberately, that the two state
stores are process-local. A single instance makes that correct; two make it
silently wrong, and section 11 of docs/audio-streaming.md asks for that to fail
loudly rather than surface as an unexplained refusal mid-walk.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.config import EXTERNAL_TIMEOUT_SECONDS, SERPAPI_KEY, TRUSTED_PROXY_HOPS
from app.dependencies import ContainerDep
from app.services.demo import load_manifest, missing_clips

router = APIRouter()


@router.get("/api/health", summary="Service health")
async def api_health(container: ContainerDep) -> dict[str, Any]:
    """Liveness, configured integrations, the state-store caveat, and the budget."""
    guardrails = container.guardrails
    manifest = load_manifest()
    demo_reason = guardrails.demo_reason()
    return {
        "status": "ok",
        "serpapi_key_configured": bool(SERPAPI_KEY),
        "timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "llm_configured": container.llm is not None,
        "tts_configured": container.tts_queue.configured,
        "stt_configured": container.stt is not None,
        # Process-local: correct for a single instance, wrong for two.
        "session_store": "process_local",
        "audio_token_store": "process_local",
        # The guardrails are process-local too, and in the same direction: two
        # instances allow two allowances and spend two budgets against one cap.
        "guardrails": "process_local",
        # Read off the live objects, never off the module constants: what this
        # process is actually enforcing is the only figure an operator can act
        # on, and a test that configures a tighter limiter should see it here.
        "walk_limit_per_hour": guardrails.limiter.limit,
        "walk_limit_window_seconds": guardrails.limiter.window_s,
        "trusted_proxy_hops": TRUSTED_PROXY_HOPS,
        "daily_spend_cap_usd": guardrails.ledger.cap_usd,
        "spend": guardrails.ledger.snapshot(),
        "demo": {
            "active": demo_reason is not None,
            "reason": demo_reason,
            "forced": guardrails.force_demo,
            # A kill-switch that fires into 404s is worse than none: the walker
            # hears silence and cannot tell why.
            "clips_recorded": manifest.recorded,
            "clips_missing": list(missing_clips()),
            "note": manifest.note,
        },
    }


__all__ = ["router"]
