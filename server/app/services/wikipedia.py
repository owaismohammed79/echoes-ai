"""Wikipedia fact providers.

AGENTS.md section 1 restricts the backend to Wikipedia GeoSearch and SerpApi
(named parks only); OpenStreetMap is explicitly out of scope.

Two ways in, both keyed on the same coordinate:

* :func:`fetch_wikipedia_facts` -- GeoSearch around a point at a chosen radius.
* :func:`fetch_wikipedia_history_facts` -- coordinate-scoped *text* search, for
  the residential case where nothing historic sits within walking distance but
  the surrounding city is full of it.

Neither widens its own search: choosing a radius is a policy decision and lives
in :mod:`app.services.context`, which can also see what the other sources found.
Every fact is typed as it is built (:mod:`app.services.fact_kind`), because the
same radius returns temples and revenue divisions, and a walk has to know which
is which to narrate the right one.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable
from urllib.parse import quote

import httpx

from app.config import (
    MAX_RESULTS_PER_SOURCE,
    SEARCH_RADIUS_MAX_M,
    SEARCH_RADIUS_M,
    WIKIPEDIA_ENDPOINT,
    WIKIPEDIA_HISTORY_QUERY,
)
from app.models import FactObject
from app.services.denylist import is_allowed_title
from app.services.fact_kind import classify
from app.services.geo import haversine_m, make_fact_id

logger = logging.getLogger("echoes.context.wikipedia")

__all__ = ["fetch_wikipedia_facts", "fetch_wikipedia_history_facts"]

#: Shared by both queries: article intro text, a thumbnail for the source
#: panel, the categories the kind is read from, and the coordinates that stand
#: in for a distance when the API does not hand one over.
_EXTRACT_PARAMS: dict[str, Any] = {
    "prop": "extracts|coordinates|pageimages|categories",
    # A thumbnail for the source panel: 300px is the smallest size that
    # still reads on a phone, and pageimages returns nothing for pages that
    # have none, so the field is simply absent rather than guessed at.
    "piprop": "thumbnail",
    "pithumbsize": 300,
    # Categories are how a fact learns what it is (``fact_kind``), and they ride
    # along in the same round trip, so typing a fact costs no extra request.
    #
    # The limit is the trap: ``cllimit`` is a ceiling on the categories in the
    # whole response, not per page. At 30, a twenty-page GeoSearch returned
    # categories for three pages and a continuation token, so most facts fell
    # back to their titles -- and a concert called "The Stones in the Park"
    # stayed a park. 500 is the ceiling the API allows a client; the extra few
    # kilobytes buy typing for every page.
    "cllimit": 500,
    # Hidden categories are the wiki's housekeeping (stub notices, date formats,
    # coordinate markers): a dozen per page of noise in a title list. The API
    # cannot be asked to omit the rest, so ``_topical_categories`` filters what
    # the taxonomy would otherwise read as evidence.
    "clshow": "!hidden",
    "exintro": 1,
    "explaintext": 1,
    "exsentences": 2,
    "redirects": 1,
    "format": "json",
    "formatversion": 2,
}

#: Categories that only say something about the wiki, not about the place.
_MAINTENANCE_CATEGORY = re.compile(
    r"^(?:All |Articles |Use |Pages |CS1 |Webarchive|Wikipedia |Commons |"
    r"Short description|Coordinates |Infobox|Good articles|Featured |"
    r"Hatnote|EngvarB|Dynamic lists)",
)


def _topical_categories(page: dict[str, Any]) -> tuple[str, ...]:
    """The page's categories, minus the ones about the wiki itself."""
    names = (str(entry.get("title", "")) for entry in (page.get("categories") or []))
    return tuple(
        name.removeprefix("Category:")
        for name in names
        if name and not _MAINTENANCE_CATEGORY.match(name.removeprefix("Category:"))
    )


async def fetch_wikipedia_facts(
    client: httpx.AsyncClient, lat: float, lon: float, *, radius_m: int = SEARCH_RADIUS_M
) -> list[FactObject]:
    """Fetch nearby Wikipedia articles from GeoSearch, filtered by the deny-list.

    A single round-trip combines two module invocations:

    * ``list=geosearch`` -- returns a reliable ``dist`` (metres) for every hit.
    * ``generator=geosearch`` + ``prop=extracts`` -- returns the intro text.

    Distance deliberately comes from ``list=geosearch`` rather than from
    ``prop=coordinates``: geosearch matches articles whose coordinates are
    stored on Wikidata, while ``prop=coordinates`` only returns locally-stored
    ones. Live measurement at Trafalgar Square showed **10 of 20** pages
    returning no coordinates at all, so a coordinates-based distance silently
    yielded 0.0 and sorted those articles as if they were the closest.

    ``radius_m`` is a parameter so the caller can widen the search (see
    :mod:`app.services.context`); the default is the hyper-local first attempt.
    """
    coord = f"{lat}|{lon}"
    params: dict[str, Any] = {
        "action": "query",
        # Distance + identity for every hit (always populated).
        "list": "geosearch",
        # MediaWiki >= 1.35 takes a single "lat|lon" tuple; the older
        # gsrlat/gsrlon pair now fails with `missingparam`.
        "gscoord": coord,
        "gsradius": radius_m,
        "gslimit": MAX_RESULTS_PER_SOURCE,
        # Same search again as a generator, so we get article text in the
        # same response. The ggs-prefixed params drive the generator.
        "generator": "geosearch",
        "ggscoord": coord,
        "ggsradius": radius_m,
        "ggslimit": MAX_RESULTS_PER_SOURCE,
        **_EXTRACT_PARAMS,
    }

    payload = await _query(client, params)
    query = payload.get("query") or {}
    hits = query.get("geosearch") or []
    distance_by_pageid = {
        hit.get("pageid"): float(hit.get("dist", 0.0)) for hit in hits
    }

    return _facts_from_pages(query.get("pages") or [], distance_by_pageid, lat, lon)


async def fetch_wikipedia_history_facts(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    *,
    radius_m: int = SEARCH_RADIUS_MAX_M,
) -> list[FactObject]:
    """Fetch the surrounding *city's* history, scoped to a coordinate.

    The fallback for a residential location, where GeoSearch returns a metro
    station and little else. It is a text search biased to the local area by
    MediaWiki's ``nearcoord:`` keyword, so no geocoder and no city name are
    needed -- which matters, because the backend has neither (AGENTS.md 1).

    Two things about that keyword are load-bearing:

    * The radius **needs a unit** (``nearcoord:10km,12.9|77.5``). Passing bare
      metres does not fail loudly: MediaWiki drops the clause, returns HTTP
      200, and answers with the whole encyclopedia (a live probe for
      "history" came back with *History of Turkey*). The warning it emits
      alongside those results is treated as a hard failure below rather than
      as a usable page set.
    * Search results carry no ``dist``, so every distance is computed from the
      page's own coordinates. A page without them is dropped, exactly as in
      GeoSearch: an invented distance would rank a citywide article beside the
      walker's feet.
    """
    params: dict[str, Any] = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{WIKIPEDIA_HISTORY_QUERY} {_nearcoord(lat, lon, radius_m)}",
        "gsrlimit": MAX_RESULTS_PER_SOURCE,
        **_EXTRACT_PARAMS,
    }

    payload = await _query(client, params)
    warning = str(
        ((payload.get("warnings") or {}).get("search") or {}).get("warnings") or ""
    )
    if "nearcoord" in warning.lower():
        raise RuntimeError(f"Wikipedia ignored the coordinate scope: {warning}")

    pages = (payload.get("query") or {}).get("pages") or []
    return _facts_from_pages(pages, {}, lat, lon)


def _nearcoord(lat: float, lon: float, radius_m: int) -> str:
    """The ``nearcoord`` clause for a coordinate-scoped search (unit required)."""
    return f"nearcoord:{max(1, round(radius_m / 1000))}km,{lat},{lon}"


async def _query(client: httpx.AsyncClient, params: dict[str, Any]) -> dict[str, Any]:
    """Run one MediaWiki query, turning an in-body error into a failure.

    A 200 response carrying an ``error`` object is not "no facts here": it is a
    malformed request, and reading it as an empty result would silently send
    the walk to Notice Mode (or, with ``nearcoord`` above, to the wrong city).
    """
    response = await client.get(WIKIPEDIA_ENDPOINT, params=params)
    response.raise_for_status()
    payload = response.json()

    if "error" in payload:
        raise RuntimeError(f"Wikipedia API error: {payload['error']}")
    return payload


def _facts_from_pages(
    pages: Iterable[dict[str, Any]],
    distance_by_pageid: dict[Any, float],
    lat: float,
    lon: float,
) -> list[FactObject]:
    """Turn MediaWiki pages into graded facts, dropping what cannot be graded.

    Each fact is typed from its title and categories as it is built (see
    ``services.fact_kind``), which is what lets the aggregator offer a monument
    before a school without either being dropped.
    """
    facts: list[FactObject] = []

    for page in pages:
        title = str(page.get("title", "")).strip()
        # A railway station is the one piece of infrastructure a walk may
        # narrate: AGENTS.md asks for a ranking signal rather than a filter, and
        # a station's own article carries the date and the architecture that
        # make it a story. Metro, bus and tram stops stay filtered out above.
        if not title or not is_allowed_title(title, allow_railway_stations=True):
            continue

        # Prefer geosearch's own distance; fall back to the returned
        # coordinates only if this page was somehow absent from the list.
        distance_m = distance_by_pageid.get(page.get("pageid"))
        if distance_m is None:
            coords = page.get("coordinates") or []
            if not coords:
                # Never invent a distance -- a fake 0.0 would rank a distant
                # article as the nearest one.
                logger.warning("no distance available for %r; dropping", title)
                continue
            try:
                distance_m = haversine_m(
                    lat, lon, float(coords[0]["lat"]), float(coords[0]["lon"])
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("unusable coordinates for %r; dropping", title)
                continue

        thumbnail = page.get("thumbnail") or {}
        image_url = thumbnail.get("source") if isinstance(thumbnail, dict) else None

        facts.append(
            FactObject(
                fact_id=make_fact_id(title, lat, lon),
                title=title,
                snippet=str(page.get("extract", "")).strip(),
                source_url=(
                    "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
                ),
                distance_m=round(distance_m, 1),
                image_url=str(image_url) if image_url else None,
                kind=classify(title, _topical_categories(page)),
            )
        )

    return facts
