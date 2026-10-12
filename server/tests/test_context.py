"""Tests for the Echoes context aggregator.

Run from ``server/``:  python -m pytest -q

Async functions are driven with ``asyncio.run`` inside sync tests so pytest
needs no asyncio plugin.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import (
    EXTERNAL_TIMEOUT_SECONDS,
    MIN_CONTEXT_FACTS,
    SEARCH_RADIUS_EXPANDED_M,
    SEARCH_RADIUS_MAX_M,
    SEARCH_RADIUS_M,
)
from app.main import create_app
from app.models import ContextRequest, FactObject, NewsItem
from app.services import news as news_service
from app.services import parks, weather, wikipedia
from app.services.context import decide_mode
from app.services.denylist import is_allowed_title
from app.services.geo import haversine_m, make_fact_id
from app.services.wikipedia import (
    fetch_wikipedia_facts,
    fetch_wikipedia_history_facts,
)

#: One app for the whole module, exactly as the previous ``main.app`` was.
app = create_app()

# A real (trimmed) MediaWiki response, captured live with
# list=geosearch + generator=geosearch&prop=extracts&formatversion=2.
#
# Note the deliberate absence of "coordinates" on pageid 4: that is the real
# shape returned for Wikidata-only coordinates, and it must not degrade into a
# fabricated 0.0m distance now that dist comes from list=geosearch.
LIVE_PAYLOAD: dict[str, Any] = {
    "batchcomplete": True,
    "query": {
        "geosearch": [
            {"pageid": 1, "title": "Nelson's Column", "dist": 21.6},
            {"pageid": 2, "title": "Charing Cross tube station", "dist": 24.0},
            {"pageid": 3, "title": "City of Westminster wholesale district", "dist": 30.5},
            {"pageid": 4, "title": "Dump Trump (statue)", "dist": 45.2},
        ],
        "pages": [
            {
                "pageid": 1,
                "title": "Nelson's Column",
                "extract": "Nelson's Column is a monument in Trafalgar Square.",
                "coordinates": [{"lat": 51.50772222, "lon": -0.12797222}],
            },
            {
                "pageid": 2,
                "title": "Charing Cross tube station",
                "extract": "Charing Cross is a London Underground station.",
            },
            {
                "pageid": 3,
                "title": "City of Westminster wholesale district",
                "extract": "A commercial district.",
            },
            {
                "pageid": 4,
                "title": "Dump Trump (statue)",
                "extract": "A temporary sculpture in Trafalgar Square.",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


def test_request_quantises_coordinates_to_3dp() -> None:
    req = ContextRequest(lat=51.5074123, lon=-0.1278987)
    assert req.lat == 51.507
    assert req.lon == -0.128


def test_request_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(Exception):
        ContextRequest(lat=91.0, lon=0.0)


# ---------------------------------------------------------------------------
# Deny-list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "London Underground Metro Station",
        "Wandsworth Constituency",
        "Wholesale Market",
        "Cutting Edge Workshop",
        "Greater London Assembly",
        "City of Westminster District",
    ],
)
def test_denylist_rejects_infrastructure_and_commercial(title: str) -> None:
    assert is_allowed_title(title) is False


@pytest.mark.parametrize(
    "title",
    ["Hyde Park", "Bridgetown", "Nelson's Column", "Trafalgar Square", "Parliament Hill"],
)
def test_denylist_allows_places(title: str) -> None:
    assert is_allowed_title(title) is True


def test_denylist_is_case_insensitive_and_word_anchored() -> None:
    assert is_allowed_title("METRO") is False
    # interior substrings must not trigger a match
    assert is_allowed_title("Metronome Park") is True
    assert is_allowed_title("Recutting Garden") is True


def test_a_named_railway_station_is_the_one_allowed_exception() -> None:
    """Asked for by name: a station's article carries its own history.

    AGENTS.md asks for a ranking signal rather than a filter that drops valid
    adjacent history, so the geosearch path may ask for stations to be kept --
    but only the railway kind. A metro stop or a bus stand is where the walker
    is standing, not somewhere to send them.
    """
    assert is_allowed_title("KSR Bengaluru City railway station") is False
    assert is_allowed_title("KSR Bengaluru City railway station", allow_railway_stations=True) is True
    assert is_allowed_title("Nelson's Column") is True

    for title in (
        "Hosahalli metro station",
        "Charing Cross tube station",
        "Mysuru Road Bus Station",
        "Bangalore City railway station metro station",
        "Wandsworth Constituency",
    ):
        assert is_allowed_title(title, allow_railway_stations=True) is False, title


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_decide_mode_story_when_facts_exist() -> None:
    fact = FactObject(
        fact_id="abc", title="Hyde Park", snippet="", source_url="u", distance_m=1.0
    )
    assert decide_mode([fact]) == "STORY"


def test_decide_mode_notice_when_empty() -> None:
    assert decide_mode([]) == "NOTICE"


def test_fact_id_is_deterministic_and_location_sensitive() -> None:
    assert make_fact_id("Hyde Park", 51.507, -0.127) == make_fact_id(
        "Hyde Park", 51.507, -0.127
    )
    assert make_fact_id("Hyde Park", 51.507, -0.127) != make_fact_id(
        "Hyde Park", 51.508, -0.127
    )


# ---------------------------------------------------------------------------
# Wikipedia client contract
# ---------------------------------------------------------------------------


def test_wikipedia_sends_modern_geosearch_params() -> None:
    """Guards the gsrlat -> gscoord rename that broke the Part 1 design."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=LIVE_PAYLOAD)

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_wikipedia_facts(c, 51.507, -0.127)

    facts = asyncio.run(run())

    assert captured["generator"] == "geosearch"
    assert captured["list"] == "geosearch"
    assert captured["ggscoord"] == "51.507|-0.127"
    assert captured["gscoord"] == "51.507|-0.127"
    assert captured["gsradius"] == "1500"
    assert captured["formatversion"] == "2"
    # the deprecated spellings must be gone
    assert "gsrlat" not in captured and "gsrlon" not in captured
    # image enrichment rides along in the same round trip
    assert captured["piprop"] == "thumbnail"
    assert captured["pithumbsize"] == "300"
    assert "pageimages" in captured["prop"]

    # station + wholesale/district pages are filtered; the monument and the
    # coordinate-less statue survive.
    assert [f.title for f in facts] == ["Nelson's Column", "Dump Trump (statue)"]
    assert facts[0].snippet.startswith("Nelson's Column is a monument")
    assert facts[0].source_url == "https://en.wikipedia.org/wiki/Nelson%27s_Column"
    # distance is taken straight from geosearch's own dist field
    assert facts[0].distance_m == pytest.approx(21.6, abs=0.01)


def test_distance_survives_missing_coordinates() -> None:
    """Regression: pages with no local coordinates must keep a real distance.

    Previously these were assigned 0.0m, which sorted distant articles to the
    top of the list as if they were on the pavement beside the walker.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=LIVE_PAYLOAD)

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_wikipedia_facts(c, 51.507, -0.127)

    facts = asyncio.run(run())
    statue = next(f for f in facts if f.title == "Dump Trump (statue)")

    assert statue.distance_m == pytest.approx(45.2, abs=0.01)
    assert statue.distance_m != 0.0
    assert all(f.distance_m > 1.0 for f in facts)


def test_client_enforces_3s_timeout_against_slow_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 3.0s ceiling must be enforced against a real stalled socket."""
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Stalling(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            time.sleep(5.0)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args: Any) -> None:  # silence test noise
            return

    server = HTTPServer(("127.0.0.1", 0), Stalling)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        wikipedia,
        "WIKIPEDIA_ENDPOINT",
        f"http://127.0.0.1:{server.server_address[1]}/",
    )

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(EXTERNAL_TIMEOUT_SECONDS)
        ) as c:
            return await fetch_wikipedia_facts(c, 51.507, -0.127)

    started = time.monotonic()
    try:
        with pytest.raises(httpx.TimeoutException):
            asyncio.run(run())
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()

    assert 2.0 <= elapsed <= 4.5, f"timeout fired after {elapsed:.2f}s, expected ~3.0s"


def test_wikipedia_raises_on_error_body() -> None:
    """A 200 response carrying a MediaWiki error must not read as 'no facts'."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"error": {"code": "missingparam", "info": "gscoord required"}}
        )

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_wikipedia_facts(c, 51.507, -0.127)

    with pytest.raises(RuntimeError, match="Wikipedia API error"):
        asyncio.run(run())


def test_wikipedia_attaches_a_thumbnail_when_present() -> None:
    payload = {
        "query": {
            "geosearch": [{"pageid": 1, "title": "Nelson's Column", "dist": 21.6}],
            "pages": [
                {
                    "pageid": 1,
                    "title": "Nelson's Column",
                    "extract": "A monument.",
                    "thumbnail": {"source": "https://upload.example/thumb.jpg"},
                }
            ],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_wikipedia_facts(c, 51.507, -0.127)

    facts = asyncio.run(run())
    assert facts[0].image_url == "https://upload.example/thumb.jpg"


# ---------------------------------------------------------------------------
# Fact retrieval: the widening ladder (PRD 3)
# ---------------------------------------------------------------------------


async def _no_facts(
    client: Any, lat: float, lon: float, *, radius_m: int | None = None
) -> list[FactObject]:
    """A Wikipedia provider that answers with nothing, at any radius."""
    return []


def _stub_wikipedia(monkeypatch: pytest.MonkeyPatch, fetch, *, history=None) -> None:
    """Patch *both* Wikipedia providers, so a test never reaches the network.

    Both, not one: a thin primary search is widened, and the last rung of that
    ladder is a different function. A test that patched only the GeoSearch
    provider would quietly issue a live request the moment its stub returned
    fewer than ``MIN_CONTEXT_FACTS``.
    """
    monkeypatch.setattr(wikipedia, "fetch_wikipedia_facts", fetch)
    monkeypatch.setattr(
        wikipedia, "fetch_wikipedia_history_facts", history or _no_facts
    )


def _no_parks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the park source off: the ladder tests are about Wikipedia alone."""

    async def none(client: Any, lat: float, lon: float) -> list[FactObject]:
        return []

    monkeypatch.setattr(parks, "fetch_park_facts", none)


#: A real (trimmed) coordinate-scoped history search: one page with no
#: coordinates, one infrastructure page, and two that can be graded.
HISTORY_PAYLOAD: dict[str, Any] = {
    "query": {
        "pages": [
            {
                "pageid": 11,
                "title": "Kote Venkataramana Temple, Bengaluru",
                "extract": "A temple in Bengaluru.",
                "coordinates": [{"lat": 12.9592, "lon": 77.5717}],
                "categories": [
                    {"ns": 14, "title": "Category:Hindu temples in Bengaluru"},
                    {"ns": 14, "title": "Category:Use dmy dates from June 2018"},
                ],
            },
            {
                "pageid": 12,
                "title": "Lal Bagh",
                "extract": "A botanical garden.",
                "coordinates": [{"lat": 12.9507, "lon": 77.5848}],
                # The title says nothing about what Lal Bagh is; the category does.
                "categories": [{"ns": 14, "title": "Category:Botanical gardens in India"}],
            },
            {
                "pageid": 13,
                "title": "Bengaluru",
                "extract": "The capital of Karnataka.",
            },
            {
                "pageid": 14,
                "title": "Bengaluru City railway station",
                "extract": "Opened in 1961, the station is the city's oldest.",
                "coordinates": [{"lat": 12.9779, "lon": 77.5662}],
                "categories": [{"ns": 14, "title": "Category:Railway stations in Bengaluru"}],
            },
            {
                "pageid": 15,
                "title": "Magadi Road metro station",
                "extract": "A metro station.",
                "coordinates": [{"lat": 12.9766, "lon": 77.5525}],
            },
        ]
    }
}


def test_history_search_is_coordinate_scoped_and_its_radius_carries_a_unit() -> None:
    """The ``km`` is load-bearing.

    Without it MediaWiki drops the clause, answers 200, and returns the whole
    encyclopedia: a live probe for "history" with a bare-metre radius came back
    with *History of Turkey*. So the request is pinned here, and distances are
    pinned to the pages' own coordinates -- search results carry no ``dist``.
    """
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=HISTORY_PAYLOAD)

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_wikipedia_history_facts(c, 12.9638, 77.548)

    facts = asyncio.run(run())

    assert captured["generator"] == "search"
    assert captured["gsrsearch"] == "history nearcoord:10km,12.9638,77.548"
    assert captured["formatversion"] == "2"

    # The coordinate-less citywide article drops out; what remains is graded by
    # real haversine distance and typed from its title and categories. The
    # railway station survives the deny-list (it is history the walk may use);
    # the metro stop does not.
    assert [f.title for f in facts] == [
        "Kote Venkataramana Temple, Bengaluru",
        "Lal Bagh",
        "Bengaluru City railway station",
    ]
    assert [f.kind for f in facts] == ["religious", "park", "station"]
    assert facts[0].distance_m == pytest.approx(
        haversine_m(12.9638, 77.548, 12.9592, 77.5717), abs=0.1
    )


def test_geosearch_keeps_a_named_railway_station_but_not_the_metro() -> None:
    """The station allowance is narrow on purpose.

    A named railway station's article carries its own date and architecture, so
    it is material a walk can use; a metro stop or a bus stand is where the
    walker already is.
    """
    payload = {
        "query": {
            "geosearch": [
                {"pageid": 1, "title": "KSR Bengaluru City railway station", "dist": 90.0},
                {"pageid": 2, "title": "Hosahalli metro station", "dist": 40.0},
                {"pageid": 3, "title": "Mysuru Road Bus Station", "dist": 60.0},
                {"pageid": 4, "title": "Sankey Tank", "dist": 120.0},
            ],
            "pages": [
                {
                    "pageid": 1,
                    "title": "KSR Bengaluru City railway station",
                    "extract": "Opened in 1961.",
                },
                {"pageid": 2, "title": "Hosahalli metro station", "extract": "A metro station."},
                {"pageid": 3, "title": "Mysuru Road Bus Station", "extract": "A bus station."},
                {
                    "pageid": 4,
                    "title": "Sankey Tank",
                    "extract": "A lake in Bengaluru.",
                    "categories": [{"ns": 14, "title": "Category:Lakes of Bengaluru"}],
                },
            ],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_wikipedia_facts(c, 12.9638, 77.548)

    facts = asyncio.run(run())
    assert [f.title for f in facts] == ["KSR Bengaluru City railway station", "Sankey Tank"]
    assert [f.kind for f in facts] == ["station", "water"]


def test_history_search_refuses_a_result_set_it_cannot_prove_is_local() -> None:
    """A parse warning means the scope was ignored, not that anywhere will do."""
    payload = dict(HISTORY_PAYLOAD)
    payload["warnings"] = {
        "search": {
            "warnings": (
                "The geo distance '10000' provided to 'nearcoord' could not be "
                "parsed."
            )
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_wikipedia_history_facts(c, 12.9638, 77.548)

    with pytest.raises(RuntimeError, match="ignored the coordinate scope"):
        asyncio.run(run())


def test_a_thin_primary_search_is_widened(monkeypatch: pytest.MonkeyPatch) -> None:
    """A residential street gets a richer walk, not a two-fact one."""
    _no_weather(monkeypatch)
    _no_parks(monkeypatch)
    seen: list[int] = []

    async def fetch(
        client: Any, lat: float, lon: float, *, radius_m: int = SEARCH_RADIUS_M
    ) -> list[FactObject]:
        seen.append(radius_m)
        if radius_m == SEARCH_RADIUS_M:
            # Measured live at 12.9638, 77.5480: everything else inside 1.5km is
            # a metro station or a constituency, so two facts survive.
            return [
                _fact("Vijayanagar, Bengaluru", 964.4),
                _fact("Bengaluru geography and environment", 1471.8),
            ]
        return [_fact(f"Landmark {n}", 2000.0 + n) for n in range(6)]

    _stub_wikipedia(monkeypatch, fetch)

    with TestClient(app) as client:
        body = client.post("/api/context", json={"lat": 12.9638, "lon": 77.548}).json()

    # Both radii were asked -- the wide one is what a residential walk needs --
    # and the walk ended with material for a story rather than two facts.
    assert seen == [SEARCH_RADIUS_M, SEARCH_RADIUS_EXPANDED_M]
    assert body["mode"] == "STORY"
    assert len(body["facts"]) >= MIN_CONTEXT_FACTS
    assert [f["distance_m"] for f in body["facts"]] == sorted(
        f["distance_m"] for f in body["facts"]
    )


def test_an_ample_local_result_ends_the_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A neighbourhood with enough facts asks for nothing wider or further."""
    _no_weather(monkeypatch)
    _no_parks(monkeypatch)
    seen: list[int] = []

    async def fetch(
        client: Any, lat: float, lon: float, *, radius_m: int = SEARCH_RADIUS_M
    ) -> list[FactObject]:
        seen.append(radius_m)
        return [_fact(f"Place {n}", 100.0 * n) for n in range(MIN_CONTEXT_FACTS)]

    _stub_wikipedia(monkeypatch, fetch)

    with TestClient(app) as client:
        body = client.post("/api/context", json={"lat": 51.507, "lon": -0.127}).json()

    # The widest GeoSearch and the city history are never asked for.
    assert set(seen) == {SEARCH_RADIUS_M, SEARCH_RADIUS_EXPANDED_M}
    assert len(body["facts"]) == MIN_CONTEXT_FACTS


def test_the_ladder_falls_back_to_the_surrounding_city_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the whole radius ladder is thin, the city answers instead."""
    _no_weather(monkeypatch)
    _no_parks(monkeypatch)
    seen: list[int] = []
    history_radii: list[int] = []

    async def fetch(
        client: Any, lat: float, lon: float, *, radius_m: int = SEARCH_RADIUS_M
    ) -> list[FactObject]:
        seen.append(radius_m)
        return [_fact("Vijayanagar, Bengaluru", 964.4)]

    async def history(
        client: Any, lat: float, lon: float, *, radius_m: int = SEARCH_RADIUS_MAX_M
    ) -> list[FactObject]:
        history_radii.append(radius_m)
        return [_fact(f"History {n}", 3000.0 + n) for n in range(4)]

    _stub_wikipedia(monkeypatch, fetch, history=history)

    with TestClient(app) as client:
        body = client.post("/api/context", json={"lat": 12.9638, "lon": 77.548}).json()

    # The two hedged radii, then the widest GeoSearch once both were thin.
    assert set(seen[:2]) == {SEARCH_RADIUS_M, SEARCH_RADIUS_EXPANDED_M}
    assert seen[2:] == [SEARCH_RADIUS_MAX_M]
    assert history_radii == [SEARCH_RADIUS_MAX_M]
    assert len(body["facts"]) == MIN_CONTEXT_FACTS


def test_a_slow_local_search_does_not_hold_up_the_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wide radius is already in flight, so a slow local one cannot gate it.

    Asking for the wide radius only once the narrow one has answered is what
    makes the material a function of the narrow query's latency: on a slow
    response the wider request was cut off behind it, and the walk narrated two
    facts while the material for eight was one in-flight request away.
    """
    import time

    from app import config

    monkeypatch.setattr(config, "EXTERNAL_TIMEOUT_SECONDS", 1.0)
    _no_weather(monkeypatch)
    _no_parks(monkeypatch)

    async def fetch(
        client: Any, lat: float, lon: float, *, radius_m: int = SEARCH_RADIUS_M
    ) -> list[FactObject]:
        if radius_m == SEARCH_RADIUS_M:
            await asyncio.sleep(0.9)
            return [_fact("Vijayanagar, Bengaluru", 964.4)]
        await asyncio.sleep(0.1)
        return [_fact(f"Landmark {n}", 2000.0 + n) for n in range(6)]

    _stub_wikipedia(monkeypatch, fetch)

    with TestClient(app) as client:
        started = time.monotonic()
        body = client.post(
            "/api/context", json={"lat": 12.9638, "lon": 77.548}
        ).json()
        elapsed = time.monotonic() - started

    assert len(body["facts"]) == 6
    assert elapsed < 0.6, f"waited {elapsed:.2f}s on the slow local search"


def test_the_widening_ladder_shares_one_network_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled Wikipedia must not multiply the 3.0s ceiling by the rungs."""
    import time

    from app import config

    monkeypatch.setattr(config, "EXTERNAL_TIMEOUT_SECONDS", 1.0)
    _no_weather(monkeypatch)
    _no_parks(monkeypatch)
    seen: list[int] = []

    async def fetch(
        client: Any, lat: float, lon: float, *, radius_m: int = SEARCH_RADIUS_M
    ) -> list[FactObject]:
        seen.append(radius_m)
        # The local rung answers inside its share of the budget; every rung after
        # it is slower than the whole budget put together.
        await asyncio.sleep(0.2 if radius_m == SEARCH_RADIUS_M else 1.5)
        return [_fact("Sparse", 964.4)]

    _stub_wikipedia(monkeypatch, fetch)

    with TestClient(app) as client:
        started = time.monotonic()
        body = client.post(
            "/api/context", json={"lat": 12.9638, "lon": 77.548}
        ).json()
        elapsed = time.monotonic() - started

    # What the first rung found survives; the second is cut off by what is left
    # of the budget, and the rungs after it are never attempted. Timed around
    # the request itself, not the client's startup.
    assert body["mode"] == "STORY"
    assert seen == [SEARCH_RADIUS_M, SEARCH_RADIUS_EXPANDED_M]
    assert elapsed < 1.4, f"ladder spent {elapsed:.2f}s of a 1.0s budget"


# ---------------------------------------------------------------------------
# SerpApi graceful degradation (AGENTS.md 5)
# ---------------------------------------------------------------------------


def test_park_lookup_degrades_to_empty_on_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A revoked SerpApi key is a quieter walk, not a 500."""
    monkeypatch.setattr(parks, "SERPAPI_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Invalid API key"})

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await parks.fetch_park_facts(c, 51.507, -0.127)

    assert asyncio.run(run()) == []


def test_park_lookup_degrades_on_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parks, "SERPAPI_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await parks.fetch_park_facts(c, 51.507, -0.127)

    assert asyncio.run(run()) == []


def test_park_lookup_degrades_on_a_serpapi_error_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parks, "SERPAPI_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "quota exhausted"})

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await parks.fetch_park_facts(c, 51.507, -0.127)

    assert asyncio.run(run()) == []


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------


def _fact(title: str, distance: float, *, kind: str = "place") -> FactObject:
    return FactObject(
        fact_id=make_fact_id(title, 51.507, -0.127),
        title=title,
        snippet=f"{title} excerpt",
        source_url="https://example.org/x",
        distance_m=distance,
        kind=kind,
    )


def _no_weather(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin conditions off, so an endpoint test never depends on the network."""

    async def none(client: Any, lat: float, lon: float):
        return None

    monkeypatch.setattr(weather, "fetch_weather", none)


def _no_news(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the news source off, for tests that are about something else.

    The news lookup is scoped by a locality the facts produced, so any test
    whose stub includes one would otherwise issue a live SerpApi request.
    """

    async def none(
        client: Any, lat: float, lon: float, *, place_hint: str | None = None
    ) -> list[NewsItem]:
        return []

    monkeypatch.setattr(news_service, "fetch_local_news", none)


def test_endpoint_returns_story_with_sorted_deduplicated_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_weather(monkeypatch)

    async def fake_wiki(
        client: Any, lat: float, lon: float, *, radius_m: int | None = None
    ) -> list[FactObject]:
        return [_fact("Far Park", 900.0), _fact("Shared Green", 500.0)]

    async def fake_parks(client: Any, lat: float, lon: float) -> list[FactObject]:
        # duplicates the Wikipedia hit; must collapse to a single entry
        return [_fact("Shared Green", 500.0), _fact("Near Gardens", 120.0)]

    _stub_wikipedia(monkeypatch, fake_wiki)
    monkeypatch.setattr(parks, "fetch_park_facts", fake_parks)

    with TestClient(app) as client:
        resp = client.post("/api/context", json={"lat": 51.5074, "lon": -0.1278})

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "STORY"
    assert [f["title"] for f in body["facts"]] == [
        "Near Gardens",
        "Shared Green",
        "Far Park",
    ]
    assert set(body["facts"][0]) == {
        "fact_id",
        "title",
        "snippet",
        "source_url",
        "distance_m",
        "image_url",
        "kind",
    }
    assert body["weather"] is None


def test_landmarks_are_offered_before_a_school_or_a_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kind decides first, distance decides inside a kind.

    Before this, ordering by distance alone handed the narrator the revenue
    division and the school, while the temple and the lake -- both under a
    kilometre away -- were the material that should have survived.
    """
    _no_weather(monkeypatch)
    _no_parks(monkeypatch)

    async def fetch(
        client: Any, lat: float, lon: float, *, radius_m: int | None = None
    ) -> list[FactObject]:
        return [
            _fact("Bengaluru division", 100.0, kind="administrative"),
            _fact("Venkat International Public School", 150.0, kind="institution"),
            _fact("Gavi Gangadhareshwara Temple", 800.0, kind="religious"),
            _fact("Sankey Tank", 900.0, kind="water"),
            _fact("Vijayanagar, Bengaluru", 950.0, kind="locality"),
        ]

    _stub_wikipedia(monkeypatch, fetch)
    # The stub carries a locality, which is exactly what the news query is
    # scoped by; pinned off here because this test is about ranking.
    _no_news(monkeypatch)

    with TestClient(app) as client:
        body = client.post("/api/context", json={"lat": 12.9638, "lon": 77.548}).json()

    assert [f["title"] for f in body["facts"]] == [
        "Gavi Gangadhareshwara Temple",
        "Sankey Tank",
        "Vijayanagar, Bengaluru",
        "Bengaluru division",
        "Venkat International Public School",
    ]
    # Nothing was dropped to achieve it: a thin neighbourhood still narrates
    # whatever it has.
    assert len(body["facts"]) == 5


def test_endpoint_returns_notice_when_no_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_weather(monkeypatch)

    async def empty(
        client: Any, lat: float, lon: float, *, radius_m: int | None = None
    ) -> list[FactObject]:
        return []

    _stub_wikipedia(monkeypatch, empty)
    monkeypatch.setattr(parks, "fetch_park_facts", empty)

    with TestClient(app) as client:
        resp = client.post("/api/context", json={"lat": 0.0, "lon": 0.0})

    assert resp.status_code == 200
    assert resp.json() == {
        "mode": "NOTICE",
        "facts": [],
        "weather": None,
        # No facts, therefore no place name for the news query: empty, and no
        # request was made (``services.news``).
        "news": [],
    }


def test_endpoint_reports_conditions_from_the_weather_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weather is a typed field from a named provider, never a guess."""
    from app.models import WeatherInfo

    async def conditions(client: Any, lat: float, lon: float) -> WeatherInfo:
        return WeatherInfo(
            temp=12.4,
            condition="light rain",
            sunset="6:42 PM",
            local_time="3:15 PM",
        )

    async def empty(
        client: Any, lat: float, lon: float, *, radius_m: int | None = None
    ) -> list[FactObject]:
        return []

    monkeypatch.setattr(weather, "fetch_weather", conditions)
    _stub_wikipedia(monkeypatch, empty)
    monkeypatch.setattr(parks, "fetch_park_facts", empty)

    with TestClient(app) as client:
        resp = client.post("/api/context", json={"lat": 51.507, "lon": -0.127})

    assert resp.json()["weather"] == {
        "temp": 12.4,
        "condition": "light rain",
        "sunset": "6:42 PM",
        "local_time": "3:15 PM",
    }


def test_endpoint_degrades_to_notice_on_upstream_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGENTS.md 5: a dead upstream yields NOTICE, never a 500."""
    _no_weather(monkeypatch)

    async def slow_wiki(
        client: Any, lat: float, lon: float, *, radius_m: int | None = None
    ) -> list[FactObject]:
        raise httpx.TimeoutException("exceeded 3.0s")

    async def failing_parks(client: Any, lat: float, lon: float) -> list[FactObject]:
        raise httpx.ConnectError("connection refused")

    _stub_wikipedia(monkeypatch, slow_wiki)
    monkeypatch.setattr(parks, "fetch_park_facts", failing_parks)

    with TestClient(app) as client:
        resp = client.post("/api/context", json={"lat": 51.507, "lon": -0.127})

    assert resp.status_code == 200
    assert resp.json() == {
        "mode": "NOTICE",
        "facts": [],
        "weather": None,
        "news": [],
    }


def test_endpoint_survives_partial_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One healthy source is enough to keep the walk in Story mode."""
    _no_weather(monkeypatch)

    async def slow_wiki(
        client: Any, lat: float, lon: float, *, radius_m: int | None = None
    ) -> list[FactObject]:
        raise httpx.TimeoutException("exceeded 3.0s")

    async def good_parks(client: Any, lat: float, lon: float) -> list[FactObject]:
        return [_fact("Near Gardens", 120.0)]

    _stub_wikipedia(monkeypatch, slow_wiki)
    monkeypatch.setattr(parks, "fetch_park_facts", good_parks)

    with TestClient(app) as client:
        resp = client.post("/api/context", json={"lat": 51.507, "lon": -0.127})

    assert resp.status_code == 200
    assert resp.json()["mode"] == "STORY"


# ---------------------------------------------------------------------------
# Local news
# ---------------------------------------------------------------------------


def test_endpoint_returns_the_headlines_the_news_provider_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """News rides to the client beside the facts it is *not*."""
    _no_weather(monkeypatch)
    _no_parks(monkeypatch)

    async def fetch(
        client: Any, lat: float, lon: float, *, radius_m: int | None = None
    ) -> list[FactObject]:
        return [_fact("Vijayanagar, Bengaluru", 950.0, kind="locality")]

    async def headlines(
        client: Any, lat: float, lon: float, *, place_hint: str | None = None
    ) -> list[NewsItem]:
        return [
            NewsItem(
                title="Residents protest against tree poisoning",
                source="The New Indian Express",
                date="2026-09-28T07:00:00Z",
                url="https://example.org/story",
            )
        ]

    _stub_wikipedia(monkeypatch, fetch)
    monkeypatch.setattr(news_service, "fetch_local_news", headlines)

    with TestClient(app) as client:
        body = client.post("/api/context", json={"lat": 12.9638, "lon": 77.548}).json()

    assert body["news"] == [
        {
            "title": "Residents protest against tree poisoning",
            "snippet": "",
            "source": "The New Indian Express",
            "date": "2026-09-28T07:00:00Z",
            "url": "https://example.org/story",
        }
    ]
    # A headline is not a fact and can never be cited as one.
    assert "fact_id" not in body["news"][0]


def test_news_is_scoped_by_the_locality_the_facts_produced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The query needs a place name, so the pipeline hands it the one it has.

    Two localities are offered on purpose: the comma form is Wikipedia's own
    disambiguation between same-named settlements, so "Vijayanagar, Bengaluru"
    is preferred over a bare "Westminster" that answers for Colorado.
    """
    _no_weather(monkeypatch)
    _no_parks(monkeypatch)
    seen: list[str | None] = []

    async def fetch(
        client: Any, lat: float, lon: float, *, radius_m: int | None = None
    ) -> list[FactObject]:
        return [
            _fact("Kempegowda Nagar", 100.0, kind="locality"),
            _fact("Vijayanagar, Bengaluru", 950.0, kind="locality"),
        ]

    async def headlines(
        client: Any, lat: float, lon: float, *, place_hint: str | None = None
    ) -> list[NewsItem]:
        seen.append(place_hint)
        return []

    _stub_wikipedia(monkeypatch, fetch)
    monkeypatch.setattr(news_service, "fetch_local_news", headlines)

    with TestClient(app) as client:
        client.post("/api/context", json={"lat": 12.9638, "lon": 77.548})

    assert seen == ["Vijayanagar, Bengaluru"]


def test_no_locality_in_the_facts_means_no_news_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_weather(monkeypatch)
    _no_parks(monkeypatch)
    seen: list[str | None] = []

    async def fetch(
        client: Any, lat: float, lon: float, *, radius_m: int | None = None
    ) -> list[FactObject]:
        return [_fact("Sankey Tank", 900.0, kind="water")]

    async def headlines(
        client: Any, lat: float, lon: float, *, place_hint: str | None = None
    ) -> list[NewsItem]:
        seen.append(place_hint)
        return []

    _stub_wikipedia(monkeypatch, fetch)
    monkeypatch.setattr(news_service, "fetch_local_news", headlines)

    with TestClient(app) as client:
        body = client.post("/api/context", json={"lat": 12.9638, "lon": 77.548}).json()

    # Called with nothing to name -- and answering with nothing, without ever
    # asking SerpApi for another city's news.
    assert seen == [None]
    assert body["news"] == []


def test_news_lookup_degrades_without_failing_the_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGENTS.md 5: a dead news provider is a quieter walk, not a 500."""
    _no_weather(monkeypatch)
    _no_parks(monkeypatch)

    async def fetch(
        client: Any, lat: float, lon: float, *, radius_m: int | None = None
    ) -> list[FactObject]:
        return [_fact("Vijayanagar, Bengaluru", 950.0, kind="locality")]

    async def broken(
        client: Any, lat: float, lon: float, *, place_hint: str | None = None
    ) -> list[NewsItem]:
        raise httpx.ConnectError("connection refused")

    _stub_wikipedia(monkeypatch, fetch)
    monkeypatch.setattr(news_service, "fetch_local_news", broken)

    with TestClient(app) as client:
        resp = client.post("/api/context", json={"lat": 12.9638, "lon": 77.548})

    assert resp.status_code == 200
    assert resp.json()["news"] == []
    assert resp.json()["mode"] == "STORY"


def test_health_reports_timeout_budget() -> None:
    with TestClient(app) as client:
        body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["timeout_seconds"] == 3.0


# ---------------------------------------------------------------------------
# Live integration (network required)
# ---------------------------------------------------------------------------


def test_live_wikipedia_roundtrip() -> None:
    """End-to-end against the real Wikipedia API at Trafalgar Square."""
    try:
        with TestClient(app) as client:
            resp = client.post("/api/context", json={"lat": 51.5074, "lon": -0.1278})
    except httpx.HTTPError as exc:  # pragma: no cover - offline environment
        pytest.skip(f"network unavailable: {exc}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] in {"STORY", "NOTICE"}

    if body["mode"] == "NOTICE":  # pragma: no cover - would signal a regression
        pytest.fail("expected facts near Trafalgar Square")

    assert all(f["snippet"] for f in body["facts"])
    # Sources are per provider now that the park lookup works again: Wikipedia
    # articles and Google Maps places. Before the engine name was fixed this
    # list could only ever contain the former.
    assert all(
        f["source_url"].startswith(
            ("https://en.wikipedia.org/wiki/", "https://www.google.com/maps/")
        )
        for f in body["facts"]
    )
    assert all(f["kind"] for f in body["facts"])
    # The same policy the pipeline asks for, railway stations included: a live
    # walk keeps historical stations and still refuses metro stops and
    # constituencies.
    assert all(
        is_allowed_title(f["title"], allow_railway_stations=True)
        for f in body["facts"]
    )
    # Ordered by kind, then by distance: the nearest of each kind leads, so
    # distances are non-decreasing within a kind rather than across the list.
    for kind in {f["kind"] for f in body["facts"]}:
        within = [f["distance_m"] for f in body["facts"] if f["kind"] == kind]
        assert within == sorted(within), kind
    distances = [f["distance_m"] for f in body["facts"]]
    assert min(distances) <= SEARCH_RADIUS_M
    assert all(0 <= d <= SEARCH_RADIUS_MAX_M for d in distances)
