"""Walk pacing: cycle kinds, time budgets, and round-robin context.

The client is a dumb player. It asks ``POST /api/walk`` with a ``session_id``,
plays whatever frames come back, then asks again when the local ambient bed
ends. This module is the server's state machine for those requests:

* whether the requested duration has elapsed (``is_expired``);
* how many words the next cycle may speak, derived from the **time left**
  rather than a fixed sentence count (``plan_cycle``);
* how long the client should sit in ambient silence afterwards, sized from the
  words that were *actually* spoken (``plan_pause_ms``, ``plan_pad_ms``);
* which slice of context the next cycle may see (``cycle_kind``,
  ``select_cycle_context``) -- never a fact or headline already narrated.

AGENTS.md forbids pre-generating a 15-minute script, and Render will drop a
request that stays open that long. One cycle per HTTP request is the shape
that satisfies both.

The time model, in one paragraph: speech runs at ``WORDS_PER_SECOND``. A normal
cycle spends ``SPEECH_SHARE`` of the remaining time talking (clamped to a
sensible chunk) and the rest as a silence gap for looking around. Once the
remaining time fits in ``FINAL_WINDOW_S`` the cycle becomes the *final* one: it
is sized to fill exactly what is left, there is no trailing gap, and the walk
ends. Landing within a few seconds of the chosen duration is the target, not
the exact second.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence

from app.models import FactObject, NewsItem, WeatherInfo

CycleKind = Literal["atmosphere", "wiki", "local", "full"]

#: Round-robin for a duration walk: weather/mood, then one wiki fact, then
#: one park or news item. ``full`` is the one-shot legacy path (no duration).
CYCLE_ORDER: tuple[CycleKind, ...] = ("atmosphere", "wiki", "local")

#: Longest silence bed the client is ever asked to play, in milliseconds. The
#: server dictates the gap so the player does not invent its own.
AMBIENT_PAUSE_MS = 75_000

#: Shortest gap worth calling a pause: below this the walker has no time to look.
MIN_PAUSE_MS = 30_000

#: Post-walk microphone grace the client plays as a bed, not a JS timer.
QA_GRACE_MS = 45_000

#: SerpApi (parks + news) is one query per source per walk unless the walker
#: has moved and several minutes have passed (AGENTS.md 4 + 6).
SERP_REFRESH_SECONDS = 6 * 60

# --- the time model ---------------------------------------------------------

#: Spoken pace used to turn seconds into words. ElevenLabs and the browser voice
#: both land near 150 words a minute for this kind of prose.
WORDS_PER_SECOND = 2.5

#: Share of the remaining time a normal cycle spends speaking.
SPEECH_SHARE = 0.4

#: A normal cycle is one chunk, not a monologue.
MIN_CYCLE_WORDS = 40
MAX_CYCLE_WORDS = 110

#: Silence is this many times the speech it follows, inside [MIN, MAX]_PAUSE_MS.
PAUSE_PER_SPEECH = 1.5

#: Once this little time is left the cycle fills it and the walk ends.
FINAL_WINDOW_S = 90

#: Below this there is nothing worth generating: the walk simply ends.
MIN_FINAL_S = 15

#: Context + model + first synthesis, before the first word is heard.
OVERHEAD_S = 6

#: The pre-recorded "stay on marked paths" clip the server appends at the end.
REMINDER_S = 4

#: A cycle is not scheduled if the one after it would have less than this left
#: to work with; the gap absorbs the remainder instead.
MIN_NEXT_CYCLE_S = 35

#: The final cycle must still say *something* worth a speaker's time.
MIN_FINAL_WORDS = 20

#: Average spoken sentence, for translating a word budget into a sentence cap.
WORDS_PER_SENTENCE = 16

#: A final cycle that finishes this early pads the difference with ambient
#: silence rather than ending the walk visibly short.
MIN_PAD_S = 8

#: How far over budget a cycle may run before sentences are trimmed.
BUDGET_SLACK = 1.15

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\u2019:.\-]*")


def count_words(text: str) -> int:
    """Spoken words in ``text`` -- the unit the time model is written in."""
    return len(_WORD.findall(text))


@dataclass(frozen=True)
class CyclePlan:
    """How big the next cycle may be, derived from the time left."""

    words: int
    sentences: int
    final: bool
    remaining_s: float


@dataclass(frozen=True)
class CycleContext:
    """The only material Gemma may see for one cycle."""

    kind: CycleKind
    facts: tuple[FactObject, ...] = ()
    news: tuple[NewsItem, ...] = ()
    weather: WeatherInfo | None = None


def is_expired(
    *,
    started_at: float,
    duration_min: int | None,
    now: float,
) -> bool:
    """True when the walker's chosen duration has elapsed.

    ``duration_min is None`` is the one-shot path: expiry is not a clock, the
    handler completes in the same response as the only cycle.
    """
    if duration_min is None:
        return False
    return (now - started_at) >= duration_min * 60


def remaining_seconds(
    *, started_at: float, duration_min: int | None, now: float
) -> float | None:
    """Seconds left in the walk, or ``None`` for the one-shot path."""
    if duration_min is None:
        return None
    return duration_min * 60 - (now - started_at)


def sentences_for(words: int) -> int:
    """Sentence cap that lets ``words`` be spoken as short sentences."""
    return max(3, min(14, round(words / WORDS_PER_SENTENCE)))


def plan_cycle(remaining_s: float) -> CyclePlan | None:
    """Size the next cycle from the time left; ``None`` means the walk is over.

    Normal cycle: ``SPEECH_SHARE`` of what remains, clamped to one chunk.
    Final cycle (``remaining_s <= FINAL_WINDOW_S``): everything that is left,
    less the model's head start and the closing reminder, so the walk ends
    when the time does.
    """
    if remaining_s < MIN_FINAL_S:
        return None

    if remaining_s <= FINAL_WINDOW_S:
        budget_s = remaining_s - OVERHEAD_S - REMINDER_S
        words = max(MIN_FINAL_WORDS, round(budget_s * WORDS_PER_SECOND))
        return CyclePlan(
            words=words,
            sentences=sentences_for(words),
            final=True,
            remaining_s=remaining_s,
        )

    raw = round(remaining_s * SPEECH_SHARE * WORDS_PER_SECOND)
    words = max(MIN_CYCLE_WORDS, min(MAX_CYCLE_WORDS, raw))
    return CyclePlan(
        words=words,
        sentences=sentences_for(words),
        final=False,
        remaining_s=remaining_s,
    )


def speech_seconds(words: int) -> float:
    return words / WORDS_PER_SECOND


def plan_pause_ms(remaining_s: float, spoken_words: int) -> int:
    """Silence after a normal cycle, from the words actually spoken.

    Proportional to the speech it follows. When the cycle after this one would
    be too short to be worth running (or would not fit at all), the gap absorbs
    the difference so the walk lands on its duration instead of overshooting it
    or ending with an orphan fragment.
    """
    spoke_s = speech_seconds(spoken_words)
    pause_s = max(
        MIN_PAUSE_MS / 1000,
        min(AMBIENT_PAUSE_MS / 1000, spoke_s * PAUSE_PER_SPEECH),
    )
    after_s = remaining_s - OVERHEAD_S - spoke_s - pause_s
    if after_s < MIN_NEXT_CYCLE_S:
        pause_s = max(0.0, pause_s + after_s)
    return int(round(pause_s * 1000))


def plan_pad_ms(remaining_s: float, spoken_words: int) -> int:
    """Silence to play after a *final* cycle that finished early.

    A model that wrote fewer words than asked, or a validator that cut some,
    would otherwise end the walk well short of the chosen duration.
    """
    left_s = (
        remaining_s - OVERHEAD_S - speech_seconds(spoken_words) - REMINDER_S
    )
    if left_s < MIN_PAD_S:
        return 0
    return int(round(left_s * 1000))


def fit_to_budget(word_counts: Sequence[int], budget_words: int) -> list[int]:
    """Indices of the sentences to keep so the cycle fits its word budget.

    The first sentence (it sets the scene) and the last (the closing
    invitation to look around) are never dropped; the overshoot is taken out of
    the middle, from the back. A model that wrote a little over is left alone.
    """
    keep = list(range(len(word_counts)))
    limit = budget_words * BUDGET_SLACK
    total = sum(word_counts)
    while total > limit and len(keep) > 2:
        dropped = keep.pop(len(keep) - 2)
        total -= word_counts[dropped]
    return keep


def cycle_kind(cycle_index: int, *, duration_min: int | None) -> CycleKind:
    """Which context slice cycle ``cycle_index`` (0-based) may see."""
    if duration_min is None:
        return "full"
    return CYCLE_ORDER[cycle_index % len(CYCLE_ORDER)]


def is_park(fact: FactObject) -> bool:
    """Parks are the SerpApi-scoped slice of ``local`` cycles."""
    return fact.kind == "park"


def wiki_facts(facts: Sequence[FactObject]) -> list[FactObject]:
    """Historical / geographical facts, parks excluded."""
    return [fact for fact in facts if not is_park(fact)]


def park_facts(facts: Sequence[FactObject]) -> list[FactObject]:
    return [fact for fact in facts if is_park(fact)]


def news_key(item: NewsItem) -> str:
    """Stable identity for a headline, recorded alongside fact ids once used."""
    return "news:" + (item.url or item.title)


def _next_unused(
    candidates: Sequence[FactObject], used_ids: Sequence[str]
) -> FactObject | None:
    """The first fact this walk has not narrated yet, else ``None``.

    Deliberately does not wrap around: narrating the same fact twice in one walk
    is the failure this exists to prevent. The caller decides what to say
    instead (see :func:`select_cycle_context`).
    """
    used = set(used_ids)
    for fact in candidates:
        if fact.fact_id not in used:
            return fact
    return None


def _next_unused_news(
    news: Sequence[NewsItem], used_ids: Sequence[str]
) -> NewsItem | None:
    used = set(used_ids)
    for item in news:
        if news_key(item) not in used:
            return item
    return None


def _wiki_slice(
    facts: Sequence[FactObject],
    used_fact_ids: Sequence[str],
    weather: WeatherInfo | None,
) -> CycleContext | None:
    chosen = _next_unused(wiki_facts(facts), used_fact_ids)
    if chosen is None:
        return None
    return CycleContext(kind="wiki", facts=(chosen,), weather=weather)


def _local_slice(
    facts: Sequence[FactObject],
    news: Sequence[NewsItem],
    used_fact_ids: Sequence[str],
    weather: WeatherInfo | None,
) -> CycleContext | None:
    park = _next_unused(park_facts(facts), used_fact_ids)
    if park is not None:
        return CycleContext(kind="local", facts=(park,), weather=weather)
    headline = _next_unused_news(news, used_fact_ids)
    if headline is not None:
        return CycleContext(kind="local", news=(headline,), weather=weather)
    return None


def select_cycle_context(
    kind: CycleKind,
    *,
    facts: Sequence[FactObject],
    news: Sequence[NewsItem] = (),
    weather: WeatherInfo | None = None,
    used_fact_ids: Sequence[str] = (),
) -> CycleContext:
    """Pick the single slice of context this cycle is allowed to narrate.

    Atmosphere sees weather (and mood, which lives on the request, not here).
    Wiki sees one unused Wikipedia fact. Local sees one unused park, or one
    unused headline when no park remains. ``full`` passes everything through,
    which is the one-shot walk the existing tests and demo still expect.

    Nothing is ever narrated twice. When the slot's own material is used up, the
    sibling slot (wiki <-> local) is tried before falling back to an atmosphere
    cycle, so a walker in a sparse neighbourhood hears fresh facts for as long as
    any exist and weather/mood after that. Facts that arrive from a refreshed
    location are new ids, so they are picked up here automatically.
    """
    if kind == "atmosphere":
        return CycleContext(kind=kind, weather=weather)

    if kind == "wiki":
        chosen = _wiki_slice(facts, used_fact_ids, weather) or _local_slice(
            facts, news, used_fact_ids, weather
        )
    elif kind == "local":
        chosen = _local_slice(facts, news, used_fact_ids, weather) or _wiki_slice(
            facts, used_fact_ids, weather
        )
    else:
        return CycleContext(
            kind="full",
            facts=tuple(facts),
            news=tuple(news),
            weather=weather,
        )

    return chosen or CycleContext(kind="atmosphere", weather=weather)


__all__ = [
    "AMBIENT_PAUSE_MS",
    "BUDGET_SLACK",
    "CYCLE_ORDER",
    "CycleContext",
    "CycleKind",
    "CyclePlan",
    "FINAL_WINDOW_S",
    "MAX_CYCLE_WORDS",
    "MIN_CYCLE_WORDS",
    "MIN_PAUSE_MS",
    "QA_GRACE_MS",
    "SERP_REFRESH_SECONDS",
    "WORDS_PER_SECOND",
    "count_words",
    "cycle_kind",
    "fit_to_budget",
    "is_expired",
    "is_park",
    "news_key",
    "park_facts",
    "plan_cycle",
    "plan_pad_ms",
    "plan_pause_ms",
    "remaining_seconds",
    "select_cycle_context",
    "sentences_for",
    "speech_seconds",
    "wiki_facts",
]
