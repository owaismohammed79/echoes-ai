"""Geospatial helpers.

Small and pure, so both the context aggregator (distance ranking) and the walk
session store (staleness check) can share them without either depending on the
other.
"""

from __future__ import annotations

import hashlib
import math
from typing import Final

_EARTH_RADIUS_M: Final[float] = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return _EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def make_fact_id(title: str, lat: float, lon: float) -> str:
    """Deterministic, collision-resistant id: stable across restarts.

    Stability matters because the prompt tells the model to copy the id
    character for character; an id that changed between the prompt and the
    validation call would reject every sentence.
    """
    raw = f"{title}\x1f{lat:.3f}\x1f{lon:.3f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


__all__ = ["haversine_m", "make_fact_id"]
