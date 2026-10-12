"""Tests for the Open-Meteo weather provider.

Weather is only allowed into a walk as a typed field from a named provider
(docs/llm-orchestration.md section 6), so these tests pin the two things that
make it trustworthy: an unrecognised code is never dressed up as a real
condition, and a failed lookup is ``None`` rather than a guess.
"""

from __future__ import annotations

import asyncio

import httpx

from app.models import WeatherInfo
from app.services.weather import (
    describe_code,
    fetch_weather,
    format_clock,
    hour_from_clock,
)


def test_describe_code_maps_known_wmo_codes() -> None:
    assert describe_code(0) == "clear"
    assert describe_code(3) == "overcast"
    assert describe_code(61) == "light rain"
    assert describe_code(95) == "thunderstorms"


def test_describe_code_is_honest_about_the_unknown() -> None:
    assert describe_code(None) == "unknown"
    assert describe_code(1234) == "unknown"
    assert describe_code("rain") == "unknown"


def test_format_clock_renders_a_twelve_hour_time() -> None:
    assert format_clock("2026-10-11T18:42") == "6:42 PM"
    assert format_clock("2026-10-11T00:05") == "12:05 AM"
    assert format_clock("2026-10-11T12:00") == "12:00 PM"
    assert format_clock("nonsense") is None
    assert format_clock(None) is None


def test_hour_from_clock_inverts_format_clock() -> None:
    """Anything reasoning about the walker's hour reads the provider's clock."""
    for value in ("2026-10-11T18:42", "2026-10-11T00:05", "2026-10-11T12:00"):
        rendered = format_clock(value)
        assert rendered is not None
        assert hour_from_clock(rendered) == int(value[11:13])

    assert hour_from_clock("7:00 PM") == 19
    assert hour_from_clock("12:30 AM") == 0
    assert hour_from_clock("not a time") is None
    assert hour_from_clock(None) is None


def test_fetch_weather_parses_current_and_sunset() -> None:
    payload = {
        "current": {
            "time": "2026-10-11T15:15",
            "temperature_2m": 12.4,
            "weather_code": 3,
        },
        "daily": {
            "sunset": ["2026-10-11T18:42"],
            "sunrise": ["2026-10-11T07:12"],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.open-meteo.com" in str(request.url)
        # The provider is asked for local times and a current reading.
        assert dict(request.url.params)["timezone"] == "auto"
        return httpx.Response(200, json=payload)

    async def run() -> WeatherInfo | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_weather(c, 51.507, -0.127)

    assert asyncio.run(run()) == WeatherInfo(
        temp=12.4,
        condition="overcast",
        sunset="6:42 PM",
        local_time="3:15 PM",
    )


def test_fetch_weather_is_none_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    async def run() -> WeatherInfo | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_weather(c, 51.507, -0.127)

    assert asyncio.run(run()) is None


def test_fetch_weather_is_none_without_a_current_reading() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"daily": {"sunset": []}})

    async def run() -> WeatherInfo | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_weather(c, 51.507, -0.127)

    assert asyncio.run(run()) is None
