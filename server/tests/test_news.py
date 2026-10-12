"""Tests for the SerpApi local-news provider.

Run from ``server/``:  python -m pytest -q

Three things are pinned here, and each of them was decided by a live probe
against the real API rather than by reading the docs:

* the *engine*. ``engine=google`` with the ``ll`` coordinate parameter returns
  no news block at all (``ll`` is a Maps parameter and the Google engine ignores
  it), and the coordinate-``uule`` variant of the same engine answered for
  Kansas City at one of two test coordinates. ``google_news`` with the walk's own
  place name answered hyper-locally every time.
* the *query*: one place name plus one constant. With no place name this module
  makes no request at all -- a query that silently answers for another city is
  worse than no news.
* the *degradation*: a timeout, a rejection or an empty feed is an empty list,
  and the key never reaches the log.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from app.models import NewsItem
from app.services import news
from app.services.news import (
    NEWS_ENGINE,
    NEWS_LIMIT,
    NEWS_TIMEOUT_SECONDS,
    fetch_local_news,
    news_query,
)

LAT, LON = 12.9638, 77.5480
PLACE = "Vijayanagar, Bengaluru"

#: A real (trimmed) ``google_news`` response for "Vijayanagar, Bengaluru news",
#: including the two source shapes the engine returns: an object with a ``name``
#: and a bare string.
LIVE_PAYLOAD: dict = {
    "search_metadata": {"status": "Success"},
    "news_results": [
        {
            "position": 1,
            "title": "Vijayanagar Club compound wall collapses in Bengaluru",
            "source": {"name": "The Hindu", "icon": "https://example.org/icon.png"},
            "link": "https://www.thehindu.com/news/cities/bangalore/example.ece",
            "date": "05/30/2026, 07:00 AM, +0000 UTC",
            "iso_date": "2026-05-30T07:00:00Z",
        },
        {
            "position": 2,
            "title": "Residents protest against Vijayanagar tree poisoning",
            "source": "The New Indian Express",
            "link": "https://www.newindianexpress.com/example",
            "date": "09/28/2026, 07:00 AM, +0000 UTC",
            "iso_date": "2026-09-28T07:00:00Z",
        },
        {
            "position": 3,
            "title": "Birthday banners back on Bengaluru's Vijayanagar streets",
            "snippet": "Footpath access has narrowed on the neighbourhood's main roads.",
            "source": {"name": "The Times of India"},
            "date": "07/20/2026, 07:00 AM, +0000 UTC",
            "iso_date": "2026-07-20T07:00:00Z",
        },
        {
            "position": 4,
            "title": "A headline the provider gave no date for",
            "source": {"name": "City Today"},
        },
        # Dropped: a headline is the one field this provider cannot invent.
        {"position": 5, "title": "  ", "source": {"name": "The Hindu"}},
    ],
}


class SlowTransport(httpx.AsyncBaseTransport):
    """A transport that outlives the provider's timeout, asynchronously.

    The mock transport's sync handler blocks the event loop, which would make
    ``asyncio.wait_for`` unable to cancel in time -- the timeout would then be
    honoured or not depending on how the race landed. This one awaits, so the
    deadline is genuinely exercised.
    """

    def __init__(self, delay: float = 0.5) -> None:
        self._delay = delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self._delay)
        return httpx.Response(200, json={"news_results": []})


def _respond(payload: dict, *, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def _run(
    handler=None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    *,
    place_hint: str | None = PLACE,
    transport: httpx.AsyncBaseTransport | None = None,
    key: str = "test-key",
) -> list[NewsItem]:
    monkeypatch.setattr(news, "SERPAPI_KEY", key)

    async def run() -> list[NewsItem]:
        client = httpx.AsyncClient(
            transport=transport or httpx.MockTransport(handler)
        )
        async with client:
            return await fetch_local_news(
                client, LAT, LON, place_hint=place_hint
            )

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def test_news_lookup_asks_the_news_engine_for_the_walks_own_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=LIVE_PAYLOAD)

    _run(handler, monkeypatch)

    assert captured["engine"] == NEWS_ENGINE == "google_news"
    assert captured["q"] == news_query(PLACE) == "Vijayanagar, Bengaluru news"
    assert captured["api_key"] == "test-key"


def test_no_place_name_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """The coordinate alone cannot scope this query, so it is not sent.

    Measured: the coordinate-scoped Google route answered for Kansas City at one
    of two probe coordinates. Answering for the wrong city is worse than not
    asking, because the narrator would then tell the walker about someone else's
    neighbourhood.
    """
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=LIVE_PAYLOAD)

    assert _run(handler, monkeypatch, place_hint=None) == []
    assert _run(handler, monkeypatch, place_hint="   ") == []
    assert calls == []


def test_no_api_key_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=LIVE_PAYLOAD)

    assert _run(handler, monkeypatch, key="") == []
    assert calls == []


# ---------------------------------------------------------------------------
# The response
# ---------------------------------------------------------------------------


def test_news_lookup_reads_the_documented_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = _run(_respond(LIVE_PAYLOAD), monkeypatch)

    # Newest first, capped at three: the provider's own order is how widely a
    # story was covered, not how recent it is.
    assert [item.title for item in items] == [
        "Residents protest against Vijayanagar tree poisoning",
        "Birthday banners back on Bengaluru's Vijayanagar streets",
        "Vijayanagar Club compound wall collapses in Bengaluru",
    ]
    assert len(items) == NEWS_LIMIT

    banners = items[1]
    assert banners.snippet == (
        "Footpath access has narrowed on the neighbourhood's main roads."
    )
    assert banners.source == "The Times of India"
    assert banners.date == "2026-07-20T07:00:00Z"
    # No link in the provider's response for this one: absent, not invented.
    assert banners.url is None

    # A bare string is a real source shape too, not only ``{name: ...}``.
    protest = items[0]
    assert protest.source == "The New Indian Express"
    assert protest.url is not None and protest.url.startswith(
        "https://www.newindianexpress.com/"
    )
    assert protest.snippet == ""


def test_a_headline_without_a_date_is_kept_and_not_misplaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No date is not the same as an old date: it keeps the provider's order."""
    payload = {
        "news_results": [
            {
                "title": "Undated but first",
                "source": {"name": "City Today"},
            },
            {
                "title": "Dated later",
                "iso_date": "2026-10-01T00:00:00Z",
                "source": {"name": "The Hindu"},
            },
        ]
    }

    items = _run(_respond(payload), monkeypatch)
    assert [item.title for item in items] == ["Dated later", "Undated but first"]
    assert items[0].date == "2026-10-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Failures: quiet, but never silent
# ---------------------------------------------------------------------------


def test_a_timeout_is_an_empty_list(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(news, "NEWS_TIMEOUT_SECONDS", 0.05)

    with caplog.at_level(logging.WARNING, logger="echoes.context.news"):
        items = _run(monkeypatch=monkeypatch, transport=SlowTransport(delay=0.5))

    assert items == []
    assert "failed" in caplog.text


def test_the_timeout_is_stricter_than_the_shared_network_budget() -> None:
    """News is enrichment: it may never extend the walk's time-to-first-audio."""
    from app.config import EXTERNAL_TIMEOUT_SECONDS

    assert NEWS_TIMEOUT_SECONDS < EXTERNAL_TIMEOUT_SECONDS


def test_a_rejected_lookup_logs_serpapi_s_own_explanation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="echoes.context.news"):
        items = _run(
            _respond({"error": "Invalid API key"}, status=401), monkeypatch
        )

    assert items == []
    assert "Invalid API key" in caplog.text
    assert "401" in caplog.text


def test_an_empty_feed_is_logged_but_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="echoes.context.news"):
        items = _run(_respond({"news_results": []}), monkeypatch)

    assert items == []
    assert "no local news reported" in caplog.text


def test_an_unusable_body_degrades(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    with caplog.at_level(logging.WARNING, logger="echoes.context.news"):
        assert _run(handler, monkeypatch) == []


def test_the_api_key_never_reaches_the_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """httpx puts the request URL -- key and all -- into its exception text."""
    monkeypatch.setattr(news, "SERPAPI_KEY", "super-secret-key")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed to connect to {request.url}", request=request)

    with caplog.at_level(logging.WARNING, logger="echoes.context.news"):
        items = _run(handler, monkeypatch)

    assert items == []
    assert "super-secret-key" not in caplog.text
    assert "api_key=<redacted>" in caplog.text
