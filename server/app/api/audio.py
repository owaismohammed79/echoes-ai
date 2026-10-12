"""``GET /api/audio/{token}`` -- relay synthesised audio for one sentence.

Implements section 6 of docs/audio-streaming.md (Channel B).

An endpoint that will synthesize whatever text it is handed is a free
text-to-speech service for anyone who finds it. So this route never synthesises:
it only redeems a token that the walk flow already paid for. Unknown, expired
and foreign-session tokens all fail the same way -- 404, with no upstream call.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.dependencies import ContainerDep

logger = logging.getLogger("echoes.api.audio")

router = APIRouter()


@router.get("/api/audio/{token}", summary="Relay synthesised audio for one sentence")
async def api_audio(
    token: str, request: Request, container: ContainerDep
) -> Response:
    """Serve a synthesised sentence. Never synthesises on demand."""
    session_id = request.headers.get("x-session-id")
    blob = container.audio_tokens.redeem(token, session_id=session_id)
    if blob is None:
        raise HTTPException(status_code=404, detail="unknown or expired audio token")
    return Response(
        content=blob.data,
        media_type=blob.content_type,
        headers={"Cache-Control": "private, no-store"},
    )


__all__ = ["router"]
