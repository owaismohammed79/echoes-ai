"""Open-Meteo weather and daylight provider.

AGENTS.md section 4 restricts the backend's *facts* to Wikipedia GeoSearch and
SerpApi. Weather is not a fact about a landmark -- it is a sensory cue the
walker is already experiencing -- and docs/llm-orchestration.md section 6 says
that if it is ever supplied to a walk it must arrive as a typed field from a
named, explicitly approved provider rather than as a model guess.

Open-Meteo is that provider: it needs no API key, so it adds no secret to the
deployment and no quota to the kill-switch's accounting. Its absence is a
``None``, never a default guess -- the walk simply carries no conditions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Final

import httpx

from app.config import WEATHER_ENDPOINT
from app.models import WeatherInfo

logger = logging.getLogger("echoes.context.weather")

__all__ = [
    "WMO_CONDITIONS",
    "describe_code",
    "fetch_weather",
    "format_clock",
    "hour_from_clock",
]

#: WMO 4677 weather codes, as Open-Meteo reports them, mapped to the plain
#: phrases a narrator would actually say. Anything unrecognised is "unknown"
#: rather than a plausible-sounding invention.
WMO_CONDITIONS: Final[dict[int, str]] = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


def describe_code(code: Any) -> str:
    """Map a WMO code to a readable condition, or ``"unknown"``."""
    if isinstance(code, bool) or not isinstance(code, (int, float)):
        return "unknown"
    return WMO_CONDITIONS.get(int(code), "unknown")


def format_clock(value: Any) -> str | None:
    """``"2026-10-11T18:42"`` -> ``"6:42 PM"``, or ``None`` when unparseable.

    Open-Meteo returns local wall-clock times once ``timezone=auto`` is set, so
    the value is read as-is rather than converted.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    hour = moment.hour % 12 or 12
    meridiem = "AM" if moment.hour < 12 else "PM"
    return f"{hour}:{moment.minute:02d} {meridiem}"


def hour_from_clock(clock: str | None) -> int | None:
    """``"7:00 PM"`` -> ``19``: the inverse of :func:`format_clock`.

    For anything that has to reason about the *walker's* hour rather than the
    server's -- which is every walker outside the deployment's timezone. The
    provider reports local wall-clock time, so the walk that is happening at
    7 PM in Anchorage must not be greeted as a morning one because the server
    sits on another meridian.
    """
    if not isinstance(clock, str) or not clock:
        return None
    try:
        return datetime.strptime(clock, "%I:%M %p").hour
    except ValueError:
        return None


async def fetch_weather(
    client: httpx.AsyncClient, lat: float, lon: float
) -> WeatherInfo | None:
    """Fetch current conditions and today's sunset.

    Never raises: a failed lookup returns ``None`` so the walk carries no
    conditions rather than no walk. The failure is logged with enough detail to
    alert on.
    """
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weather_code",
        "daily": "sunset,sunrise",
        "timezone": "auto",
    }

    try:
        response = await client.get(WEATHER_ENDPOINT, params=params)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning("Open-Meteo lookup failed (%s); no conditions this walk", exc)
        return None
    except ValueError as exc:
        logger.warning("Open-Meteo returned an unusable body: %s", exc)
        return None

    current = payload.get("current") or {}
    local_time = format_clock(current.get("time"))
    if not current or local_time is None:
        # A payload without a current reading is not conditions we can speak.
        logger.warning("Open-Meteo payload carried no current reading")
        return None

    daily = payload.get("daily") or {}
    sunsets = daily.get("sunset") or []
    temp = current.get("temperature_2m")

    return WeatherInfo(
        temp=float(temp) if isinstance(temp, (int, float)) and not isinstance(temp, bool) else None,
        condition=describe_code(current.get("weather_code")),
        sunset=format_clock(sunsets[0]) if sunsets else None,
        local_time=local_time,
    )
