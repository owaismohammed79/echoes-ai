"""Shared request and response models.

``FactObject`` is the project's central data contract: it is what the context
aggregator produces, what the prompts are given, and what the validator checks
citations against (AGENTS.md section 1). It lives in its own module so the
services layer can describe it without importing the HTTP layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ContextRequest(BaseModel):
    """Inbound location.

    Coordinates are quantised to 3 decimal places (~100m) as required by
    PRD section 5 (Privacy). The rounding happens in the schema so every
    downstream consumer sees already-quantised values.
    """

    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude")
    # Carried for payload parity with /api/walk so the client can post one shape.
    # The aggregator itself does not route on either value.
    duration: int | None = Field(
        default=None, ge=1, description="Requested walk length in minutes"
    )
    mood: str | None = Field(default=None, description="Selected mood key")

    @field_validator("lat", "lon")
    @classmethod
    def _quantise(cls, v: float) -> float:
        """Reduce precision to 3dp so callers cannot submit a precise fix."""
        return round(v, 3)

    model_config = {
        "json_schema_extra": {"example": {"lat": 51.507, "lon": -0.127}}
    }


class FactObject(BaseModel):
    """A single grounded fact. Story Mode may only cite these (AGENTS.md 1)."""

    fact_id: str = Field(..., description="Stable unique identifier for this fact")
    title: str = Field(..., description="Title/heading of the source")
    snippet: str = Field(..., description="Brief excerpt or summary")
    source_url: str = Field(..., description="URL to the original source")
    distance_m: float = Field(
        ..., ge=0.0, description="Distance from the request point, in metres"
    )
    image_url: str | None = Field(
        default=None, description="Source thumbnail URL, when the provider supplied one"
    )
    kind: str = Field(
        default="place",
        description=(
            "What sort of thing this is (temple, lake, monument, railway station, "
            "school...). Ranked on, so a monument is narrated before a school, "
            "and shown to the walker beside the source."
        ),
    )


class NewsItem(BaseModel):
    """A recent local headline, for atmosphere only.

    Deliberately *not* a ``FactObject``: news is not graded, ranked or cited.
    It has no ``fact_id``, so the narrator cannot cite it, and the only thing it
    may do with it is allude to one item conversationally (see
    ``prompts.format_news_block``). A headline is a snippet of a story this
    system has not read, which is exactly why it can never be stated as fact.
    """

    title: str = Field(..., description="The headline")
    snippet: str = Field(
        default="", description="Summary the provider gave, often empty"
    )
    source: str | None = Field(default=None, description="Publisher, e.g. 'The Hindu'")
    date: str | None = Field(default=None, description="Provider's date string")
    url: str | None = Field(default=None, description="Link to the article")


class WeatherInfo(BaseModel):
    """Conditions the walker is already experiencing, from a named provider.

    Weather is a sensory cue, never a claim the model may fabricate
    (docs/llm-orchestration.md section 6). It is supplied by Open-Meteo -- a
    keyless provider -- rather than inferred, and its absence is a `None`, not a
    default guess.
    """

    temp: float | None = Field(default=None, description="Air temperature in Celsius")
    condition: str = Field(..., description="Human-readable sky condition")
    sunset: str | None = Field(
        default=None, description="Local sunset time, e.g. '6:42 PM'"
    )
    local_time: str = Field(..., description="Local time, e.g. '3:15 PM'")


class ContextResponse(BaseModel):
    """Structured response: routing flag, grounded facts, and conditions."""

    mode: str = Field(..., description="Routing decision: 'STORY' or 'NOTICE'")
    facts: list[FactObject] = Field(
        default_factory=list, description="Grounded facts, nearest first"
    )
    weather: WeatherInfo | None = Field(
        default=None, description="Conditions at the request point, when available"
    )
    news: list[NewsItem] = Field(
        default_factory=list,
        description=(
            "Recent local headlines at the request point, newest first. Context "
            "for the narration's atmosphere, never citable: empty when no news "
            "provider answered, which is a quieter walk, not a failed one"
        ),
    )


class WalkRequest(BaseModel):
    """A walk step: where the walker is, and which walk they are continuing."""

    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    session_id: str | None = None
    duration: int | None = Field(
        default=None, ge=1, description="Requested walk length in minutes"
    )
    mood: str | None = Field(default=None, description="Selected mood key")


__all__ = [
    "ContextRequest",
    "ContextResponse",
    "FactObject",
    "NewsItem",
    "WalkRequest",
    "WeatherInfo",
]
