"""Tests for the SerpApi park provider.

Run from ``server/``:  python -m pytest -q

Two things are pinned here, and both of them failed silently in production
before: the *engine name* (SerpApi answers an engine it does not have with HTTP
400 and an explanatory body, which read as "no parks nearby"), and the response
shape (``gps_coordinates``, not the ``gps`` string this provider used to look
for, which left every park without a distance).
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from app.models import FactObject
from app.services import parks
from app.services.geo import haversine_m
from app.services.parks import PARKS_ENGINE, PARKS_QUERY, fetch_park_facts

LAT, LON = 12.9638, 77.5480

#: A real (trimmed) response from the Google Maps engine, at these coordinates.
LIVE_PAYLOAD: dict = {
    "search_metadata": {"status": "Success"},
    "local_results": [
        {
            "position": 1,
            "title": "Subhash Chandra Bose Park",
            "type": "Park",
            "address": "1st Main Rd, Hampi Nagar, Vijayanagar",
            "gps_coordinates": {"latitude": 12.9724, "longitude": 77.5458},
            "place_id": "ChIJVVVVVVVVV",
            "thumbnail": "https://serpapi.com/searches/example/thumb.jpeg",
        },
        {
            "position": 2,
            "title": "Basaveshwara Park (Hampinagara)",
            "type": "Park",
            "address": "Hampinagara",
            "gps_coordinates": {"latitude": 12.9611, "longitude": 77.5501},
            "place_id": "ChIJWWWWWWWWW",
        },
        {
            "position": 3,
            "title": "Park with no coordinates",
            "type": "Park",
            "address": "Somewhere else",
        },
    ],
}


def _respond(payload: dict, *, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def _run(handler, monkeypatch: pytest.MonkeyPatch) -> list[FactObject]:
    monkeypatch.setattr(parks, "SERPAPI_KEY", "test-key")

    async def run() -> list[FactObject]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_park_facts(c, LAT, LON)

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def test_park_lookup_asks_a_supported_engine_scoped_by_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``google_maps_local`` is not a SerpApi engine, and this is the test for it.

    It answered ``Unsupported `google_maps_local` search engine`` on every walk,
    with only the status line logged -- so the park source never contributed a
    fact and never said why.
    """
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=LIVE_PAYLOAD)

    _run(handler, monkeypatch)

    assert captured["engine"] == PARKS_ENGINE == "google_maps"
    assert captured["q"] == PARKS_QUERY == "Parks nearby"
    # Coordinates, not a place name: the backend has no geocoder, and
    # ``google_local`` refuses a latitude and longitude outright.
    assert captured["ll"] == f"@{LAT},{LON},14z"
    assert captured["api_key"] == "test-key"


# ---------------------------------------------------------------------------
# The response
# ---------------------------------------------------------------------------


def test_park_lookup_reads_the_documented_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = _run(_respond(LIVE_PAYLOAD), monkeypatch)

    # Nearest first: SerpApi's own order is relevance, not distance.
    assert [f.title for f in facts] == [
        "Basaveshwara Park (Hampinagara)",
        "Subhash Chandra Bose Park",
    ]

    bose = next(f for f in facts if f.title == "Subhash Chandra Bose Park")
    assert bose.kind == "park"
    # The type and the address Google returned: "Park" alone is no substance.
    assert bose.snippet == "Park, 1st Main Rd, Hampi Nagar, Vijayanagar"
    assert bose.source_url == (
        "https://www.google.com/maps/place/?q=place_id:ChIJVVVVVVVVV"
    )
    assert bose.image_url == "https://serpapi.com/searches/example/thumb.jpeg"
    assert bose.distance_m == pytest.approx(
        haversine_m(LAT, LON, 12.9724, 77.5458), abs=0.1
    )


def test_a_park_without_coordinates_is_dropped_not_placed_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0.0 distance ranks a park across the city beside the walker."""
    facts = _run(_respond(LIVE_PAYLOAD), monkeypatch)
    assert "Park with no coordinates" not in [f.title for f in facts]
    assert all(f.distance_m > 1.0 for f in facts)


def test_a_park_without_a_place_id_falls_back_to_a_maps_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "local_results": [
            {
                "title": "Cubbon Park",
                "type": "Park",
                "address": "Kasturba Road",
                "gps_coordinates": {"latitude": 12.9763, "longitude": 77.5929},
            }
        ]
    }

    facts = _run(_respond(payload), monkeypatch)
    assert facts[0].source_url == (
        "https://www.google.com/maps/search/?api=1&query=Cubbon%20Park%20Kasturba%20Road"
    )


def test_parks_are_ordered_by_distance_and_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Twenty parks is a directory, not a walk, and SerpApi's order is not distance."""
    payload = {
        "local_results": [
            {
                "title": f"Park {index}",
                "type": "Park",
                # Deliberately unordered, and further out the higher the index.
                "gps_coordinates": {"latitude": LAT + 0.02 * index, "longitude": LON},
            }
            for index in reversed(range(1, 21))
        ]
    }

    facts = _run(_respond(payload), monkeypatch)

    assert len(facts) == parks.PARKS_LIMIT
    assert [f.distance_m for f in facts] == sorted(f.distance_m for f in facts)
    assert facts[0].title == "Park 1"


# ---------------------------------------------------------------------------
# Failures: quiet, but never silent
# ---------------------------------------------------------------------------


def test_a_rejected_lookup_logs_serpapi_s_own_explanation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The message is the fix: the status line alone hid this for a whole session."""
    payload = {"error": "Unsupported `google_maps_local` search engine."}

    with caplog.at_level(logging.WARNING, logger="echoes.context.parks"):
        facts = _run(_respond(payload, status=400), monkeypatch)

    assert facts == []
    assert "Unsupported `google_maps_local` search engine." in caplog.text
    assert "400" in caplog.text


def test_the_api_key_never_reaches_the_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """httpx puts the request URL -- key and all -- into its exception text."""
    monkeypatch.setattr(parks, "SERPAPI_KEY", "super-secret-key")

    def handler(request: httpx.Request) -> httpx.Response:
        # The shape real httpx errors take: the transport failure carries the URL.
        raise httpx.ConnectError(f"failed to connect to {request.url}", request=request)

    with caplog.at_level(logging.WARNING, logger="echoes.context.parks"):
        facts = _run(handler, monkeypatch)

    assert facts == []
    assert "super-secret-key" not in caplog.text
    assert "api_key=<redacted>" in caplog.text


def test_park_lookup_degrades_on_a_401(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A revoked key is a quieter walk, not a 500 -- and says so in the log."""
    with caplog.at_level(logging.WARNING, logger="echoes.context.parks"):
        facts = _run(
            _respond({"error": "Invalid API key"}, status=401), monkeypatch
        )

    assert facts == []
    assert "Invalid API key" in caplog.text
