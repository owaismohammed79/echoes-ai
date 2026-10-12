"""``POST /api/transition`` -- the bridge back into a resumed walk.

The resume half of the hands-free loop. When a push-to-talk answer finishes, the
client asks for one fixed line to speak before the narration picks up again, so
the walk does not restart mid-air.

A POST with no body on purpose: it changes nothing and takes nothing. The phrase
is the server's, which is what keeps this from being a synthesis endpoint — the
same reason ``audio.py`` only redeems tokens the walk already paid for.
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from app.api.headers import fallback_headers
from app.dependencies import ContainerDep
from app.services.transition import TRANSITION_PHRASE, transition_audio

router = APIRouter()

#: The line also travels in a header, so the client's browser fallback speaks the
#: same words the remote voice would have. Exposed via ``config.EXPOSED_HEADERS``.
TRANSITION_HEADER = "X-Transition-Text"


@router.post("/api/transition", summary="Speak the bridge into a resumed walk")
async def api_transition(container: ContainerDep) -> Response:
    """Return the bridge's audio, or the text for the client to speak itself."""
    outcome = await transition_audio(container.tts_queue)
    # Percent-encoded once, like every other text header (headers.py).
    text_header = {TRANSITION_HEADER: quote(TRANSITION_PHRASE, safe="")}

    if not outcome.ok:
        # No remote voice. The same flag the rest of the API uses, and the text
        # in the body so the client can speak it with speechSynthesis.
        logger_reason = outcome.reason or "tts_unavailable"
        return JSONResponse(
            status_code=200,
            content={"text": TRANSITION_PHRASE, "use_native_tts": True},
            headers={
                **fallback_headers(use_native_tts=True, reason=logger_reason),
                **text_header,
            },
        )

    return Response(
        content=outcome.audio,
        media_type=outcome.content_type or "audio/mpeg",
        headers={
            **fallback_headers(use_native_tts=False),
            **text_header,
            "Cache-Control": "private, no-store",
        },
    )


__all__ = ["router"]
