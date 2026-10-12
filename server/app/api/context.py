"""``POST /api/context`` -- aggregate hyper-local context.

A thin handler: parse, delegate to the service, return. All policy (concurrency,
failure degradation, ranking) lives in :mod:`app.services.context`.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import ContainerDep
from app.models import ContextRequest, ContextResponse
from app.services.context import aggregate_context

router = APIRouter()


@router.post(
    "/api/context",
    response_model=ContextResponse,
    summary="Aggregate hyper-local context",
)
async def api_context(
    payload: ContextRequest, container: ContainerDep
) -> ContextResponse:
    """Aggregate Wikipedia + park data into grounded facts and a routing flag."""
    return await aggregate_context(
        container.http, lat=payload.lat, lon=payload.lon
    )


__all__ = ["router"]
