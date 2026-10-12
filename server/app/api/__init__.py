"""HTTP layer.

One router per concern, all aggregated into :data:`api_router` so the app
factory has a single thing to mount. Handlers here parse requests and shape
responses; every decision about *what* to do lives in ``app.services``.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.audio import router as audio_router
from app.api.context import router as context_router
from app.api.health import router as health_router
from app.api.qa import router as qa_router
from app.api.transition import router as transition_router
from app.api.walk import router as walk_router

api_router = APIRouter()
api_router.include_router(context_router)
api_router.include_router(health_router)
api_router.include_router(walk_router)
api_router.include_router(audio_router)
api_router.include_router(qa_router)
api_router.include_router(transition_router)

__all__ = ["api_router"]
