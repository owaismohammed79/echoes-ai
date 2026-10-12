"""Application factory.

Deliberately small: it builds the lifespan (which owns the shared HTTP client
and the service container), applies CORS, mounts the routers, and stops. All
behaviour lives in ``app/services`` and ``app/api``.

``server/main.py`` remains the ASGI entrypoint (``uvicorn main:app``); it simply
calls :func:`create_app`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.config import ALLOWED_ORIGINS, EXPOSED_HEADERS
from app.dependencies import build_container, create_http_client
from app.services.demo import static_root

logger = logging.getLogger("echoes.app")

DESCRIPTION = (
    "Hyper-local context aggregation and grounded narration for the Echoes "
    "walking companion. Aggregates Wikipedia GeoSearch and SerpApi park data "
    "into FactObject instances with a deterministic STORY/NOTICE routing flag, "
    "then narrates them through a citation validator, a streaming TTS queue, "
    "and a Push-To-Talk question path."
)

VERSION = "0.4.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the process-wide resources for the life of the app.

    One connection pool is shared by every outbound call (Wikipedia, SerpApi,
    Gemma, ElevenLabs, Groq) so a walk never pays TCP setup for a second
    dependency.
    """
    async with create_http_client() as client:
        app.state.container = build_container(client)
        logger.info("echoes ready")
        yield
        logger.info("echoes shutting down")


def create_app() -> FastAPI:
    """Build the ASGI application."""
    app = FastAPI(
        title="Echoes API",
        description=DESCRIPTION,
        version=VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_ORIGINS),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        # The audio and PTT endpoints return their metadata in headers, so a
        # cross-origin frontend cannot read any of it unless it is exposed.
        expose_headers=list(EXPOSED_HEADERS),
    )

    app.include_router(api_router)

    # Pre-recorded assets: the pedestrian reminder and the demo clips. Plain files
    # on disk, so serving one costs a stat and a read -- no handler, no provider,
    # and no way for a request to turn into a bill. This also makes walk.py's
    # `/static/reminder.wav` reference resolvable, which it was not before: the
    # directory did not exist and nothing mounted it, so that URL 404'd.
    app.mount("/static", StaticFiles(directory=static_root()), name="static")
    return app


__all__ = ["create_app", "lifespan"]
