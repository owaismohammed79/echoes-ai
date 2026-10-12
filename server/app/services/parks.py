"""SerpApi park provider.

AGENTS.md section 4 permits exactly one query: ``"Parks nearby"``. Notably it
forbids ``"Trees nearby"``, which drags in commercial plant nurseries and is
the documented reason the constraint exists.

The engine is ``google_maps``, which is the one SerpApi offers that scopes a
local search by *coordinates* -- all this backend knows about the walker. The
engine this module used to name, ``google_maps_local``, is not a SerpApi engine
at all: it answers HTTP 400 ``Unsupported `google_maps_local` search engine``.
Because only the status line was logged, the park source was silently empty on
every walk -- a failure that looked exactly like "there are no parks here".
"""

from __future__ import annotations

import logging
from typing import Any, Final, Mapping
from urllib.parse import quote

import httpx

from app.config import SERPAPI_ENDPOINT, SERPAPI_KEY
from app.models import FactObject
from app.services.denylist import is_allowed_title
from app.services.geo import haversine_m, make_fact_id
from app.services.redact import redact

logger = logging.getLogger("echoes.context.parks")

#: The one permitted query (AGENTS.md section 4). Named as a constant so it is
#: obvious that no other value may be substituted.
PARKS_QUERY: Final[str] = "Parks nearby"

#: The coordinate-scoped Maps engine. See the module docstring for the engine
#: that does not exist and the silence it caused.
PARKS_ENGINE: Final[str] = "google_maps"

#: Maps zoom for a walk: 14z spans a few kilometres, which is the ground the
#: walker is actually crossing.
PARKS_ZOOM: Final[int] = 14

#: How many parks this source may contribute. Google returns twenty, which is a
#: park directory rather than a walk: the prompt offers the narrator eight facts,
#: and one source must not be able to fill all eight with the same kind of place.
PARKS_LIMIT: Final[int] = 5

#: Where a walker's source panel should point. ``place_id_search`` in the
#: response is a SerpApi endpoint that needs our key, so it is useless in a
#: client; a Maps link by ``place_id`` opens the place itself.
MAPS_PLACE_URL: Final[str] = "https://www.google.com/maps/place/?q=place_id:{}"
MAPS_SEARCH_URL: Final[str] = "https://www.google.com/maps/search/?api=1&query={}"

__all__ = ["PARKS_ENGINE", "PARKS_LIMIT", "PARKS_QUERY", "fetch_park_facts"]


def _source_url(item: Mapping[str, Any], title: str) -> str:
    """A link the walker can open for this place."""
    place_id = str(item.get("place_id") or "").strip()
    if place_id:
        return MAPS_PLACE_URL.format(quote(place_id))

    query = " ".join(
        part
        for part in (title, str(item.get("address") or "").strip())
        if part
    )
    return MAPS_SEARCH_URL.format(quote(query))


async def fetch_park_facts(
    client: httpx.AsyncClient, lat: float, lon: float
) -> list[FactObject]:
    """Fetch nearby parks from SerpApi.

    Degrades to an empty list on every failure -- no API key, a 4xx, a timeout,
    an unparseable body -- so the caller still receives a well-formed response
    and routes to NOTICE mode. A park lookup that fails is a quieter walk, never
    a crashed one (AGENTS.md section 5).

    What it is *not* allowed to be is a silent failure. SerpApi explains a
    rejection in the response body ("Invalid API key", "Unsupported ... search
    engine", "Your account has run out of searches"), and that sentence is
    logged -- with the key it travelled with redacted out of it.
    """
    if not SERPAPI_KEY:
        logger.warning("SERPAPI_KEY unset; skipping park lookup")
        return []

    params: dict[str, Any] = {
        "engine": PARKS_ENGINE,
        "type": "search",
        "q": PARKS_QUERY,
        "ll": f"@{lat},{lon},{PARKS_ZOOM}z",
        "api_key": SERPAPI_KEY,
    }

    try:
        response = await client.get(SERPAPI_ENDPOINT, params=params)
        payload = response.json()
    except httpx.HTTPError as exc:
        # Covers a connection failure and a timeout. A *rejection* is not an
        # exception here: the status is read below, so its body can be logged.
        logger.warning(
            "SerpApi park lookup failed (%s); no parks this walk", redact(exc)
        )
        return []
    except ValueError as exc:
        logger.warning("SerpApi returned an unusable body: %s", exc)
        return []

    if response.status_code != 200 or payload.get("error"):
        logger.warning(
            "SerpApi park lookup rejected it (HTTP %s): %s",
            response.status_code,
            redact(payload.get("error") or "(no error message in the body)"),
        )
        return []

    facts: list[FactObject] = []
    for item in payload.get("local_results", []) or []:
        title = str(item.get("title", "")).strip()
        if not title or not is_allowed_title(title):
            continue

        # ``gps_coordinates`` is the documented shape. Without it there is no
        # distance, and a park with no distance used to be stored as 0.0 metres
        # -- which ranked a park across the city as the thing beside the walker.
        # So a park that cannot be placed is dropped, and logged.
        gps = item.get("gps_coordinates") or {}
        try:
            park_lat = float(gps["latitude"])
            park_lon = float(gps["longitude"])
        except (KeyError, TypeError, ValueError):
            logger.info("no location for park %r; dropping", title)
            continue

        thumbnail = item.get("thumbnail")

        # The narrator reads ``detail`` as the substance of the fact, and the
        # Maps shape has no ``description``: "Park" alone tells it nothing, so
        # the type is joined with the address Google returned.
        detail = str(item.get("description") or "").strip()
        if not detail:
            detail = ", ".join(
                str(part).strip()
                for part in (item.get("type"), item.get("address"))
                if part
            )

        facts.append(
            FactObject(
                fact_id=make_fact_id(title, lat, lon),
                title=title,
                snippet=detail,
                source_url=_source_url(item, title),
                distance_m=round(haversine_m(lat, lon, park_lat, park_lon), 1),
                image_url=str(thumbnail) if thumbnail else None,
                kind="park",
            )
        )

    # Nearest first, then capped. SerpApi's order is its own relevance, not
    # distance (a live response put a park at 9km above one at 1.4km), so the
    # nearest are chosen deliberately rather than taken as they arrive.
    facts.sort(key=lambda fact: fact.distance_m)
    return facts[:PARKS_LIMIT]
