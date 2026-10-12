"""SerpApi local-news provider.

AGENTS.md section 4 permits exactly two SerpApi queries: the park lookup in
``app/services/parks.py`` and the one this module makes. News is the second
because it is the one thing the other sources cannot supply: Wikipedia knows
what a place *was*, and nothing in the pipeline knows what is happening in it
this week.

**Which SerpApi route, and why.** Measured live at 12.9638, 77.5480 and
51.5074, -0.1278 (both engines, one search each):

* ``engine=google`` with ``q=local news`` and the ``ll`` coordinate parameter --
  the obvious reading of "scope the search to the walker" -- returns **no**
  ``local_news`` block at all, because ``ll`` is a Maps/Local parameter and the
  Google engine ignores it. The block only appears once the query is ``news``
  *and* the location is sent as a ``uule``, and then it is unreliable: one of
  two coordinates came back with a Kansas City feed.
* ``engine=google_news`` with a **place-named** query ("Vijayanagar Bengaluru
  news") returns hyper-local, current headlines every time -- a compound wall
  collapse on Vijayanagar Food Street, encroachments cleared on the same street
  -- with no ``gl`` guess needed.

So the route is ``google_news`` scoped by the place name the walk already knows
(see :func:`app.services.context._place_hint`), not by the coordinate. That is a
real constraint on the caller and is stated here rather than hidden: with no
place name this provider makes no request at all, because the coordinate-only
route is the one that lies about where the walker is.

Degradation follows AGENTS.md section 5. A news lookup that fails, times out or
returns nothing is an empty list: news is enrichment, and a walk narrated
without it is a quieter walk, never a crashed one. The failure is logged -- with
the key redacted (``app/services/redact.py``), since this request carries it in
the query string.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Final, Mapping

import httpx

from app.config import SERPAPI_ENDPOINT, SERPAPI_KEY
from app.models import NewsItem
from app.services.redact import redact

logger = logging.getLogger("echoes.context.news")

#: The news engine. See the module docstring for the Google engine and the
#: coordinate-scoped route that were measured and rejected.
NEWS_ENGINE: Final[str] = "google_news"

#: How many headlines the walk may be told about. Two or three is a passing
#: thought; a list of ten is a bulletin, and the narrator may allude to exactly
#: one of them anyway (``prompts.MAX_NEWS_REFERENCES``).
NEWS_LIMIT: Final[int] = 3

#: The news lookup's own ceiling. Stricter than the shared 3.0s network budget
#: because news is not on the critical path: it is read once the facts are in,
#: so it gets what is left of the budget, and never more than this.
NEWS_TIMEOUT_SECONDS: Final[float] = 1.5

#: The language the walk is spoken in. The place name already localises the
#: search, so this only keeps the feed in English.
NEWS_LANGUAGE: Final[str] = "en"

#: What turns a place name into a news query ("Vijayanagar, Bengaluru news").
NEWS_QUERY_SUFFIX: Final[str] = "news"

__all__ = [
    "NEWS_ENGINE",
    "NEWS_LIMIT",
    "NEWS_TIMEOUT_SECONDS",
    "fetch_local_news",
    "news_query",
]


def news_query(place_hint: str) -> str:
    """The one news query: the walk's own place name, scoped to news.

    A single constant-in-query rather than a template with options, so it is
    obvious that this module asks exactly one question.
    """
    return f"{place_hint.strip()} {NEWS_QUERY_SUFFIX}"


def _source_name(source: Any) -> str | None:
    """The publisher's name.

    ``news_results`` items carry ``source.name`` in the shape SerpApi documents,
    but a plain string turns up in real responses, so both are read.
    """
    if isinstance(source, Mapping):
        name = str(source.get("name") or "").strip()
        return name or None
    text = str(source or "").strip()
    return text or None


def _news_items(payload: Mapping[str, Any]) -> list[NewsItem]:
    """Read the documented response shape into headlines, newest first.

    ``news_results`` is ordered by Google's own relevance, which at a place
    query means the most-covered story rather than the most recent one (a
    response for this walk led with a story from May while one from last week
    sat below it). The walk's own claim is that the city feels *alive*, so the
    newest are what go in the prompt; items with no usable date keep the order
    the provider gave them rather than being sorted to either end.
    """
    items: list[NewsItem] = []
    for entry in payload.get("news_results", []) or []:
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        items.append(
            NewsItem(
                title=title,
                snippet=str(entry.get("snippet") or "").strip(),
                source=_source_name(entry.get("source")),
                date=str(entry.get("iso_date") or entry.get("date") or "").strip()
                or None,
                url=str(entry.get("link") or "").strip() or None,
            )
        )

    # ``iso_date`` is RFC-3339 and sorts lexicographically; the provider's own
    # ``date`` string ("19 hours ago") does not, so an item whose date cannot be
    # compared keeps the position it was given instead of being guessed at.
    def dated(item: NewsItem) -> bool:
        return bool(item.date) and item.date[:4].isdigit()

    newest = sorted(
        (item for item in items if dated(item)),
        key=lambda item: item.date or "",
        reverse=True,
    )
    return (newest + [item for item in items if not dated(item)])[:NEWS_LIMIT]


async def fetch_local_news(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    *,
    place_hint: str | None = None,
) -> list[NewsItem]:
    """Fetch 2-3 recent local headlines, or an empty list.

    ``place_hint`` is the walk's own name for where it is -- a locality the
    Wikipedia ladder found (``_place_hint`` in :mod:`app.services.context`).
    Without one this returns immediately and makes no request: the coordinate
    alone cannot scope a *reliable* news query (see the module docstring), and
    a query that silently answers for another city is worse than no news.

    Every failure -- no key, no hint, a timeout, a 4xx, an unparseable body --
    is an empty list, and says so in the log. SerpApi explains a rejection in
    the response body, and that sentence is logged with the key redacted out.
    """
    if not SERPAPI_KEY:
        logger.warning("SERPAPI_KEY unset; skipping local news lookup")
        return []

    hint = (place_hint or "").strip()
    if not hint:
        logger.info(
            "no place name for lat=%.3f lon=%.3f; no local news this walk", lat, lon
        )
        return []

    params: dict[str, Any] = {
        "engine": NEWS_ENGINE,
        "q": news_query(hint),
        "hl": NEWS_LANGUAGE,
        "api_key": SERPAPI_KEY,
    }

    try:
        response = await asyncio.wait_for(
            client.get(SERPAPI_ENDPOINT, params=params), NEWS_TIMEOUT_SECONDS
        )
        payload = response.json()
    except (TimeoutError, httpx.HTTPError) as exc:
        # Covers a connection failure and the timeout above. A *rejection* is
        # not an exception here: its status is read below so its body can be
        # logged, which is how the unsupported-engine bug in ``parks`` was found.
        logger.warning(
            "SerpApi news lookup failed (%s); no news this walk", redact(exc)
        )
        return []
    except ValueError as exc:
        logger.warning("SerpApi returned an unusable news body: %s", exc)
        return []

    if response.status_code != 200 or payload.get("error"):
        logger.warning(
            "SerpApi news lookup rejected it (HTTP %s): %s",
            response.status_code,
            redact(payload.get("error") or "(no error message in the body)"),
        )
        return []

    items = _news_items(payload)
    if not items:
        # Not a failure: plenty of coordinates have nothing local in the feed.
        logger.info("no local news reported for %r", hint)
    return items
