"""Context aggregation: sources -> grounded facts -> routing decision.

Implements PRD section 3 steps 2-3.

The aggregator is the only place that decides how a partial or total upstream
failure is presented, and its rule is fixed by AGENTS.md section 5: no facts
means NOTICE, never an HTTP 500. It is also the only place that decides *how
wide* to search -- a fact provider answers for one radius, and widening is a
policy that needs the other sources' results to decide.

Three sources are asked for material: Wikipedia (a widening ladder of
GeoSearch radii), SerpApi's park lookup, and SerpApi's news lookup. The first two
fan out together; news is read immediately after, because its provider is scoped
by a place name and the name comes from the facts (see :func:`_place_hint`). All
three share one network budget, so the extra stage cannot stretch the walk.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Iterable

import httpx

from app.config import (
    MIN_CONTEXT_FACTS,
    SEARCH_RADIUS_EXPANDED_M,
    SEARCH_RADIUS_M,
    SEARCH_RADIUS_MAX_M,
)
from app.models import ContextResponse, FactObject, NewsItem, WeatherInfo
from app.services import news as news_service
from app.services import parks, weather, wikipedia
from app.services.fact_kind import priority

logger = logging.getLogger("echoes.context.aggregator")

__all__ = ["aggregate_context", "decide_mode"]


def decide_mode(facts: list[FactObject]) -> str:
    """Deterministic router (PRD section 3).

    Zero qualifying facts must always fall back to NOTICE so the client
    never attempts to narrate an empty context.
    """
    return "STORY" if facts else "NOTICE"


async def _safe_fetch(
    coro, source: str, *, timeout: float | None = None
) -> list[FactObject]:
    """Await a fetch, degrading to an empty list instead of failing the request.

    AGENTS.md section 5 makes NOTICE mode the deterministic consequence of
    having no facts, so an upstream timeout must not surface as a 500. The
    failure is logged with enough detail to alert on.

    ``timeout`` is the *remaining* share of the network budget when several
    fetches share one (see :func:`_gather_wikipedia_facts`); ``None`` means the
    client's own timeout is the only ceiling.
    """
    try:
        return await asyncio.wait_for(coro, timeout)
    except TimeoutError:
        logger.warning("%s lookup exceeded its remaining budget", source)
    except httpx.TimeoutException:
        logger.warning(
            "%s lookup exceeded the %.1fs budget", source, _timeout_budget()
        )
    except httpx.HTTPError as exc:
        logger.warning("%s lookup failed: %s", source, exc)
    except (RuntimeError, ValueError) as exc:
        logger.warning("%s returned an unusable payload: %s", source, exc)
    return []


async def _safe_weather(coro) -> WeatherInfo | None:
    """Await the weather lookup, degrading to ``None`` instead of failing.

    Conditions are a cue, not a fact: a walk without them is a quieter walk,
    and a failed lookup must never take the request down (AGENTS.md section 5).
    """
    try:
        return await coro
    except httpx.HTTPError as exc:
        logger.warning("weather lookup failed: %s", exc)
    except (RuntimeError, ValueError) as exc:
        logger.warning("weather returned an unusable payload: %s", exc)
    return None


def _timeout_budget() -> float:
    """Read the budget lazily so tests can patch it on the config module."""
    from app import config

    return config.EXTERNAL_TIMEOUT_SECONDS


def _merge_facts(*groups: Iterable[FactObject]) -> list[FactObject]:
    """Deduplicate facts by id, first occurrence winning.

    The same page arrives from several places -- Wikipedia and SerpApi for a
    park, the narrow search and a widened one for a monument -- and the first
    occurrence is the one graded at the shortest distance, which is the one the
    ranking should keep.
    """
    merged: dict[str, FactObject] = {}
    for group in groups:
        for fact in group:
            merged.setdefault(fact.fact_id, fact)
    return list(merged.values())


#: One entry per widening step: what to call it in the logs, and how to fetch
#: it. A factory rather than a coroutine so a step that is skipped (because the
#: walk already has enough facts, or the budget is gone) never creates a
#: coroutine nobody awaits.
_WikipediaStep = tuple[str, Callable[[], Awaitable[list[FactObject]]]]

#: The two radii asked *together*. The narrow one is what keeps the narration
#: hyper-local; the wide one is a superset of it -- GeoSearch returns the
#: nearest matches first, so a wider radius adds facts past the first one's
#: edge and removes none -- and it is what a residential street needs.
#: Measured at 12.9638, 77.5480: the narrow radius leaves two facts, a locality
#: and the city's geography article, because everything else within 1.5km is a
#: metro station or a constituency; the wide radius finds temples, gardens, a
#: basilica and a memorial hall.
_HEDGED_RADII: tuple[int, ...] = (SEARCH_RADIUS_M, SEARCH_RADIUS_EXPANDED_M)


def _wikipedia_label(radius_m: int) -> str:
    """How one rung of the ladder is named in the logs."""
    return "Wikipedia" if radius_m == SEARCH_RADIUS_M else f"Wikipedia ({radius_m}m)"


def _further_ladder(
    client: httpx.AsyncClient, lat: float, lon: float
) -> tuple[_WikipediaStep, ...]:
    """The rungs after the hedged pair, for a walk that is still thin.

    Widest GeoSearch first, then the surrounding city's history: how far out a
    text search has to reach before it stops being about the walker's location.
    """
    return (
        (
            _wikipedia_label(SEARCH_RADIUS_MAX_M),
            lambda: wikipedia.fetch_wikipedia_facts(
                client, lat, lon, radius_m=SEARCH_RADIUS_MAX_M
            ),
        ),
        (
            "Wikipedia (city history)",
            lambda: wikipedia.fetch_wikipedia_history_facts(
                client, lat, lon, radius_m=SEARCH_RADIUS_MAX_M
            ),
        ),
    )


async def _gather_wikipedia_facts(
    client: httpx.AsyncClient, lat: float, lon: float
) -> list[FactObject]:
    """Fetch Wikipedia facts, widening the search until there is enough material.

    The narrow radius and the wide one are asked **together**, not one after the
    other: asking for the wide radius only once the narrow one has answered
    makes the walk's material a function of the narrow query's latency, and
    measured here at 0.6-1.3s per call against a 3.0s budget, the second request
    was sometimes cut off behind the first -- the walk then narrated two facts
    while the material for eight was one in-flight request away.

    The ladder shares one network budget. Without that, a stalling Wikipedia
    would multiply the 3.0s ceiling by the number of steps, and the walker's
    first audio byte would be late by the same factor (PRD section 5).
    """
    budget = _timeout_budget()
    deadline = time.monotonic() + budget
    facts: list[FactObject] = []

    hedged = [
        asyncio.ensure_future(
            _safe_fetch(
                wikipedia.fetch_wikipedia_facts(client, lat, lon, radius_m=radius),
                _wikipedia_label(radius),
                timeout=budget,
            )
        )
        for radius in _HEDGED_RADII
    ]
    try:
        # The first answer with enough material decides the walk; the other
        # request is abandoned (see :data:`_HEDGED_RADII` for why that cannot
        # lose a fact this walk wanted).
        for finished in asyncio.as_completed(hedged):
            facts = _merge_facts(facts, await finished)
            if len(facts) >= MIN_CONTEXT_FACTS:
                break
    finally:
        for task in hedged:
            task.cancel()
        # Let the cancellations land here rather than after the response, so no
        # request outlives the walk that ordered it.
        await asyncio.gather(*hedged, return_exceptions=True)

    # Still thin. Sequential from here: both hedged results are in hand, so the
    # only question left is whether one more request is worth its latency.
    for source, fetch in _further_ladder(client, lat, lon):
        if len(facts) >= MIN_CONTEXT_FACTS:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "Wikipedia ladder out of budget at %d facts; %s not attempted",
                len(facts),
                source,
            )
            break
        facts = _merge_facts(
            facts, await _safe_fetch(fetch(), source, timeout=remaining)
        )

    return facts


#: Kinds whose title names where the walk *is* rather than what it passes. The
#: news lookup is scoped by a place name (``services.news``) and this is the one
#: the pipeline already knows: a locality article from the GeoSearch around the
#: walker's own coordinate.
_PLACE_KINDS: tuple[str, ...] = ("locality",)


def _place_hint(facts: Iterable[FactObject]) -> str | None:
    """The walk's own name for where it is, or ``None``.

    A Wikipedia title of the form "Vijayanagar, Bengaluru" is preferred: that
    comma is the encyclopedia's own disambiguation between settlements that
    share a name, and it is what makes the news query answer for *this*
    Vijayanagar (a bare "Westminster news" answers for Westminster, Colorado).
    A single-word locality is still better than no news at all, so it is the
    fallback rather than a rejection.
    """
    localities = [fact for fact in facts if fact.kind in _PLACE_KINDS]
    for fact in localities:
        if "," in fact.title:
            return fact.title
    return localities[0].title if localities else None


def _rank(fact: FactObject) -> tuple[int, float]:
    """How a fact is offered to the narrator: kind first, then distance.

    Distance alone put a school and a revenue division ahead of the temples and
    gardens a few hundred metres further out, and with only eight slots in the
    prompt the landmarks lost. Within a kind the nearest still leads, so this is
    an ordering, not a filter: the school is narrated when the school is the
    neighbourhood (AGENTS.md section 1).
    """
    return (priority(fact.kind), fact.distance_m)


async def aggregate_context(
    client: httpx.AsyncClient, *, lat: float, lon: float
) -> ContextResponse:
    """Aggregate Wikipedia + park + news data into facts and a routing flag."""
    # Concurrent, not sequential: total latency stays near the slowest single
    # call (~3.0s ceiling) rather than their sum. The Wikipedia branch is itself
    # a ladder, but it carries its own single budget, so widening cannot stretch
    # the ceiling either.
    deadline = time.monotonic() + _timeout_budget()
    wiki_facts, park_facts, conditions = await asyncio.gather(
        # Wrapped even though the ladder already degrades per step: the
        # aggregator's contract is that *no* source failure reaches the walk as
        # an HTTP error, and that is enforced at the boundary, not assumed.
        _safe_fetch(_gather_wikipedia_facts(client, lat, lon), "Wikipedia"),
        _safe_fetch(parks.fetch_park_facts(client, lat, lon), "SerpApi"),
        _safe_weather(weather.fetch_weather(client, lat, lon)),
    )

    # Deduplicate on fact_id (the same park can arrive from both sources), then
    # offer the walk's most narratable material first: kind, then distance.
    facts = sorted(_merge_facts(wiki_facts, park_facts), key=_rank)

    # News comes second rather than in the gather above, and the reason is the
    # news provider's contract rather than a preference: SerpApi's news engine
    # can only be scoped by a place name (see ``services.news``), and the name is
    # what the facts just produced. It is still bounded by the same request
    # budget -- ``min`` of its own 1.5s ceiling and whatever is left of the 3.0s
    # -- so it can never extend the walk's time-to-first-audio.
    news: list[NewsItem] = []
    hint = _place_hint(facts)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        logger.warning("context budget exhausted; news not attempted")
    else:
        news = await _safe_fetch(
            news_service.fetch_local_news(client, lat, lon, place_hint=hint),
            "SerpApi news",
            timeout=min(news_service.NEWS_TIMEOUT_SECONDS, remaining),
        )

    if len(facts) < MIN_CONTEXT_FACTS:
        # Every source answered as well as it could and the walk is still thin:
        # there is nothing left to widen, so narrate from what exists. Logged
        # because it is the signal that retrieval, not the model, is the limit.
        logger.warning(
            "only %d facts at lat=%.3f lon=%.3f after every fallback; "
            "narrating from sparse material",
            len(facts),
            lat,
            lon,
        )

    logger.info(
        "context lat=%.3f lon=%.3f wiki=%d parks=%d news=%d mode=%s",
        lat,
        lon,
        len(wiki_facts),
        len(park_facts),
        len(news),
        decide_mode(facts),
    )

    return ContextResponse(
        mode=decide_mode(facts), facts=facts, weather=conditions, news=news
    )
