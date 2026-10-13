"""Walk pacing, round-robin context, and citation validation across cycles.

The client is a dumb player. These tests pin the server's state machine:
duration expiry, the weather → wiki → park/news rotation, and the validator
still dropping uncited historical claims.
"""

from __future__ import annotations

import pytest

from app.models import NewsItem, WeatherInfo
from app.services import parks, wikipedia
from app.services.pacing import (
    AMBIENT_PAUSE_MS,
    FINAL_WINDOW_S,
    MAX_CYCLE_WORDS,
    MIN_CYCLE_WORDS,
    MIN_NEXT_CYCLE_S,
    MIN_PAUSE_MS,
    OVERHEAD_S,
    REMINDER_S,
    WORDS_PER_SECOND,
    count_words,
    cycle_kind,
    fit_to_budget,
    is_expired,
    news_key,
    plan_cycle,
    plan_pad_ms,
    plan_pause_ms,
    select_cycle_context,
    speech_seconds,
)
from app.services.prompts import WALK_HANDOFF
from app.services.sessions import SessionStore
from app.validation import RejectionReason, SentenceRejected, validate_story_sentence
from tests.helpers import (
    FID,
    OTHER,
    FakeClock,
    ScriptedLLM,
    StubTTS,
    build_container,
    client_for,
    fact,
    frame_text,
    ndjson,
    spoken,
)

WEATHER = WeatherInfo(
    temp=9.0, condition="light rain", sunset="6:42 PM", local_time="3:15 PM"
)
ATMOSPHERE = "The evening light is soft here. Look around when you are ready."
WIKI_LINE = f"Nelson's Column rises over the square.[cite:{FID}]"
PARK_ID = "aaaaaaaaaaaaaaaa"
PARK_LINE = f"The park keeps a quiet lawn.[cite:{PARK_ID}]"


def _stub(monkeypatch: pytest.MonkeyPatch, wiki, park) -> None:
    async def wiki_facts(client, lat, lon, *, radius_m: int | None = None):
        return wiki

    async def park_facts(client, lat, lon):
        return park

    async def no_history(client, lat, lon, *, radius_m: int | None = None):
        return []

    monkeypatch.setattr(wikipedia, "fetch_wikipedia_facts", wiki_facts)
    monkeypatch.setattr(wikipedia, "fetch_wikipedia_history_facts", no_history)
    monkeypatch.setattr(parks, "fetch_park_facts", park_facts)


def test_cycle_kind_rotates_weather_wiki_local() -> None:
    assert cycle_kind(0, duration_min=15) == "atmosphere"
    assert cycle_kind(1, duration_min=15) == "wiki"
    assert cycle_kind(2, duration_min=15) == "local"
    assert cycle_kind(3, duration_min=15) == "atmosphere"
    assert cycle_kind(0, duration_min=None) == "full"


def test_expiry_uses_the_requested_duration() -> None:
    assert is_expired(started_at=0, duration_min=5, now=299) is False
    assert is_expired(started_at=0, duration_min=5, now=300) is True
    assert is_expired(started_at=0, duration_min=15, now=15 * 60) is True
    assert is_expired(started_at=0, duration_min=30, now=30 * 60 - 1) is False
    assert is_expired(started_at=0, duration_min=None, now=10_000) is False


def test_round_robin_feeds_one_slice_per_cycle() -> None:
    wiki = fact(FID, "Nelson's Column")
    park = fact(PARK_ID, "Hyde Park", kind="park")
    news = [NewsItem(title="Local fair opens", snippet="A weekend fair.")]

    atm = select_cycle_context("atmosphere", facts=[wiki, park], news=news, weather=WEATHER)
    assert atm.facts == ()
    assert atm.news == ()
    assert atm.weather is WEATHER

    one = select_cycle_context("wiki", facts=[wiki, park], news=news, weather=WEATHER)
    assert [f.fact_id for f in one.facts] == [FID]
    assert one.news == ()

    local = select_cycle_context("local", facts=[wiki, park], news=news, weather=WEATHER)
    assert [f.fact_id for f in local.facts] == [PARK_ID]


def test_uncited_wiki_claims_are_still_stripped() -> None:
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(
            "The fort was built here in 1843.",
            valid_fact_ids={FID: "https://en.wikipedia.org/wiki/x"},
            allow_uncited_framing=True,
            allow_atmosphere=True,
        )
    assert exc.value.reason is RejectionReason.NO_CITATION


def test_atmosphere_allows_clock_and_weather_without_a_citation() -> None:
    out = validate_story_sentence(
        "It is 6:42 PM and the air is cooling toward sunset.",
        valid_fact_ids={},
        allow_uncited_framing=True,
        allow_atmosphere=True,
    )
    assert out.citations == ()
    assert "6:42" in out.text


def _cycle_client(monkeypatch: pytest.MonkeyPatch, clock: FakeClock):
    _stub(
        monkeypatch,
        [fact(FID, "Nelson's Column"), fact(OTHER, "Charing Cross")],
        [fact(PARK_ID, "Hyde Park", kind="park")],
    )
    llm = ScriptedLLM([ATMOSPHERE, WIKI_LINE, PARK_LINE])
    sessions = SessionStore(clock=clock)
    container = build_container(llm=llm, tts=StubTTS(), sessions=sessions)
    return container, llm


@pytest.mark.parametrize("minutes", [5, 15, 30])
def test_walk_with_duration_returns_one_cycle_then_pauses(
    monkeypatch: pytest.MonkeyPatch, minutes: int
) -> None:
    """A duration walk is one HTTP cycle, not a 30-minute stream."""
    clock = FakeClock()
    container, llm = _cycle_client(monkeypatch, clock)

    with client_for(container) as client:
        frames = ndjson(
            client.post(
                "/api/walk",
                json={
                    "lat": 51.507,
                    "lon": -0.127,
                    "duration": minutes,
                    "session_id": "walk-a",
                    "mood": "calm",
                },
            )
        )

    assert frames[0]["type"] == "data"
    assert frames[0]["context"] == "atmosphere"
    assert frames[-1]["type"] == "indicator"
    assert frames[-1]["status"] == "cycle_complete"
    # Sized from the words actually spoken, inside the server's bounds.
    assert MIN_PAUSE_MS <= frames[-1]["pause_ms"] <= AMBIENT_PAUSE_MS
    assert frames[-1]["complete"] is False
    assert any(f["type"] == "text" for f in frames)
    assert any(f["type"] == "audio" for f in frames)
    assert llm.calls[0]["user"].startswith("WALKWAY CONTEXT")
    assert "tag:" not in llm.calls[0]["user"]
    assert "no facts block" in llm.calls[0]["user"]


def test_sequential_cycles_rotate_context_and_complete_when_time_is_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    container, llm = _cycle_client(monkeypatch, clock)
    body = {
        "lat": 51.507,
        "lon": -0.128,
        "duration": 5,
        "session_id": "walk-b",
        "mood": "calm",
    }

    with client_for(container) as client:
        first = ndjson(client.post("/api/walk", json=body))
        assert first[-1]["status"] == "cycle_complete"
        assert first[0]["context"] == "atmosphere"

        second = ndjson(client.post("/api/walk", json=body))
        assert second[-1]["status"] == "cycle_complete"
        assert second[0]["context"] == "wiki"
        assert any(f.get("citations") == [FID] for f in spoken(second))

        third = ndjson(client.post("/api/walk", json=body))
        assert third[0]["context"] == "local"

        session = container.sessions.get("walk-b")
        assert session is not None
        assert session.duration_min == 5
        session.created_at = container.sessions.now() - (5 * 60)
        done = ndjson(client.post("/api/walk", json=body))

    assert done[-1]["type"] == "indicator"
    assert done[-1]["status"] == "walk_complete"
    assert done[-1]["complete"] is True
    assert any(f.get("role") == "reminder" for f in done)
    assert len(llm.calls) == 3


def test_continuation_does_not_spend_another_walk_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    container, _llm = _cycle_client(monkeypatch, clock)
    body = {
        "lat": 51.507,
        "lon": -0.127,
        "duration": 15,
        "session_id": "walk-c",
    }

    with client_for(container) as client:
        first = client.post("/api/walk", json=body)
        remaining = first.headers["x-ratelimit-remaining"]
        second = client.post("/api/walk", json=body)

    assert second.headers["x-ratelimit-remaining"] == remaining
    assert ndjson(second)[-1]["status"] == "cycle_complete"


def test_oneshot_walk_without_duration_completes_in_the_same_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, [fact(FID)], [])
    container = build_container(
        llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), tts=StubTTS()
    )

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 51.507, "lon": -0.127}))

    assert frames[-1]["status"] == "walk_complete"
    assert frame_text(spoken(frames)[1]) == "A claim."


# ---------------------------------------------------------------------------
# Time-based word budgets
# ---------------------------------------------------------------------------


def test_count_words_counts_spoken_words() -> None:
    assert count_words("The evening light is soft here.") == 6
    assert count_words("It is 6:42 PM and 9 degrees.") == 7
    assert count_words("") == 0


def test_normal_cycle_is_one_clamped_chunk() -> None:
    long_walk = plan_cycle(45 * 60)
    assert long_walk is not None and not long_walk.final
    assert long_walk.words == MAX_CYCLE_WORDS

    # Just outside the final window the share of the remaining time wins.
    edge = plan_cycle(FINAL_WINDOW_S + 5)
    assert edge is not None and not edge.final
    assert MIN_CYCLE_WORDS <= edge.words <= MAX_CYCLE_WORDS


def test_final_cycle_fills_the_time_that_is_left() -> None:
    plan = plan_cycle(FINAL_WINDOW_S)
    assert plan is not None and plan.final
    expected = round((FINAL_WINDOW_S - OVERHEAD_S - REMINDER_S) * WORDS_PER_SECOND)
    assert plan.words == expected
    # Fewer seconds left means fewer words, never more.
    shorter = plan_cycle(50)
    assert shorter is not None and shorter.final
    assert shorter.words < plan.words


def test_nothing_is_generated_when_the_walk_has_run_out() -> None:
    assert plan_cycle(14) is None
    assert plan_cycle(0) is None
    assert plan_cycle(-30) is None


def test_pause_scales_with_speech_and_is_bounded() -> None:
    long_speech = plan_pause_ms(2700, MAX_CYCLE_WORDS)
    assert long_speech == round(speech_seconds(MAX_CYCLE_WORDS) * 1.5 * 1000)
    assert long_speech <= AMBIENT_PAUSE_MS
    assert plan_pause_ms(2700, 5) == MIN_PAUSE_MS


@pytest.mark.parametrize("remaining", range(91, 2701, 53))
@pytest.mark.parametrize("words", [10, 40, 75, 110])
def test_the_gap_never_overshoots_and_never_strands_a_fragment(
    remaining: int, words: int
) -> None:
    pause_s = plan_pause_ms(remaining, words) / 1000
    after = remaining - OVERHEAD_S - speech_seconds(words) - pause_s
    # Either a real next cycle fits, or the walk lands on its duration.
    assert after >= MIN_NEXT_CYCLE_S - 0.01 or abs(after) < 0.01


@pytest.mark.parametrize("minutes", [5, 15, 30, 45])
@pytest.mark.parametrize("overhead", [3, 6, 10])
@pytest.mark.parametrize("fill", [1.0, 0.8, 0.6])
def test_a_simulated_walk_ends_on_its_duration(
    minutes: int, overhead: int, fill: float
) -> None:
    """Drive the planner like the client does and check where the walk lands.

    Each cycle spends ``overhead`` seconds before the first word, then the
    planned words at the speaking pace; the gap is whatever the server asked
    for. Landing within a few seconds of the chosen length is the requirement.
    """
    total = minutes * 60
    t = 0.0
    cycles = 0
    while True:
        remaining = total - t
        plan = plan_cycle(remaining)
        if plan is None:
            break
        cycles += 1
        assert cycles < 200, "walk never ended"
        # ``fill`` models a model that writes less than it was asked to.
        words = round(plan.words * fill)
        t += overhead + speech_seconds(words)
        if plan.final:
            t += plan_pad_ms(remaining, words) / 1000 + REMINDER_S
            break
        t += plan_pause_ms(remaining, words) / 1000

    # The user's tolerance is a few seconds either way; the only error left is
    # the model's real head start differing from the planner's estimate.
    assert abs(t - total) <= 5, f"{minutes}min walk ended at {t:.0f}s of {total}s"
    assert cycles >= 2


def test_a_short_final_cycle_is_padded_not_cut_short() -> None:
    # Asked for ~175 words but only 100 survived: pad the difference.
    pad = plan_pad_ms(FINAL_WINDOW_S - 10, 100)
    assert pad > 8_000
    # A cycle that hit its budget needs no padding.
    plan = plan_cycle(FINAL_WINDOW_S - 10)
    assert plan is not None
    assert plan_pad_ms(plan.remaining_s, plan.words) < 8_000


def test_fit_to_budget_trims_the_middle_and_keeps_the_ends() -> None:
    assert fit_to_budget([10, 10, 10], 100) == [0, 1, 2]
    # Over budget: drop from the back of the middle, keep first and last.
    assert fit_to_budget([10, 10, 10, 10, 10, 10], 30) == [0, 1, 5]
    # Never below two sentences, however small the budget.
    assert fit_to_budget([30, 30, 30], 5) == [0, 2]


# ---------------------------------------------------------------------------
# Nothing is narrated twice
# ---------------------------------------------------------------------------


def test_exhausted_wiki_slot_falls_back_to_the_park_then_to_atmosphere() -> None:
    wiki = fact(FID, "Nelson's Column")
    park = fact(PARK_ID, "Hyde Park", kind="park")

    first = select_cycle_context("wiki", facts=[wiki, park], used_fact_ids=[])
    assert [f.fact_id for f in first.facts] == [FID]

    # Wiki used up: the sibling slot's park is offered instead of a repeat.
    second = select_cycle_context("wiki", facts=[wiki, park], used_fact_ids=[FID])
    assert second.kind == "local"
    assert [f.fact_id for f in second.facts] == [PARK_ID]

    # Everything used: weather and mood, never the same fact again.
    last = select_cycle_context(
        "wiki", facts=[wiki, park], used_fact_ids=[FID, PARK_ID], weather=WEATHER
    )
    assert last.kind == "atmosphere"
    assert last.facts == ()
    assert last.weather is WEATHER


def test_local_slot_uses_each_headline_once() -> None:
    a = NewsItem(title="Fair opens", url="https://n/a")
    b = NewsItem(title="Bridge reopens", url="https://n/b")

    first = select_cycle_context("local", facts=[], news=[a, b], used_fact_ids=[])
    assert first.kind == "local"
    assert first.news == (a,)

    second = select_cycle_context(
        "local", facts=[], news=[a, b], used_fact_ids=[news_key(a)]
    )
    assert second.news == (b,)

    spent = select_cycle_context(
        "local", facts=[], news=[a, b], used_fact_ids=[news_key(a), news_key(b)]
    )
    assert spent.kind == "atmosphere"


def _spoken_text(frames: list[dict]) -> list[str]:
    return [frame_text(f) for f in spoken(frames)]


def test_a_walk_never_repeats_a_fact_across_many_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    container, llm = _cycle_client(monkeypatch, clock)
    body = {"lat": 51.507, "lon": -0.127, "duration": 45, "session_id": "walk-n"}

    contexts: list[str] = []
    with client_for(container) as client:
        for _ in range(8):
            frames = ndjson(client.post("/api/walk", json=body))
            contexts.append(frames[0]["context"])
            clock.advance(60)

    # Three facts exist (two wiki, one park); the rotation uses each exactly
    # once and then settles into atmosphere rather than wrapping around.
    prompts = [call["user"] for call in llm.calls]
    for fid in (FID, OTHER, PARK_ID):
        assert sum(f"id: {fid}" in p for p in prompts) == 1, fid
    assert contexts[:3] == ["atmosphere", "wiki", "local"]
    assert contexts[3:5] == ["atmosphere", "wiki"]
    assert contexts[-1] == "atmosphere"


def test_the_handoff_is_spoken_once_after_the_first_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    container, _llm = _cycle_client(monkeypatch, clock)
    body = {"lat": 51.507, "lon": -0.127, "duration": 30, "session_id": "walk-h"}

    cycles: list[list[str]] = []
    with client_for(container) as client:
        for _ in range(4):
            cycles.append(_spoken_text(ndjson(client.post("/api/walk", json=body))))
            clock.advance(90)

    assert cycles[0][-1] == WALK_HANDOFF
    for later in cycles[1:]:
        assert WALK_HANDOFF not in later


def test_later_cycles_are_told_what_was_already_said_and_how_long_to_be(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    container, llm = _cycle_client(monkeypatch, clock)
    body = {"lat": 51.507, "lon": -0.127, "duration": 30, "session_id": "walk-s"}

    with client_for(container) as client:
        ndjson(client.post("/api/walk", json=body))
        clock.advance(90)
        ndjson(client.post("/api/walk", json=body))

    first, second = llm.calls[0]["user"], llm.calls[1]["user"]
    assert "ALREADY SAID" not in first
    assert "ALREADY SAID" in second
    assert "evening light is soft" in second  # the first cycle's own words
    for prompt in (first, second):
        assert f"LENGTH: about {MAX_CYCLE_WORDS} words" in prompt


def test_the_final_cycle_is_sized_to_the_time_left_and_ends_the_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    container, llm = _cycle_client(monkeypatch, clock)
    body = {"lat": 51.507, "lon": -0.127, "duration": 5, "session_id": "walk-f"}

    with client_for(container) as client:
        ndjson(client.post("/api/walk", json=body))
        session = container.sessions.get("walk-f")
        assert session is not None
        # 80 seconds of the walk remain.
        session.created_at = container.sessions.now() - (5 * 60 - 80)
        frames = ndjson(client.post("/api/walk", json=body))

    plan = frames[0]["plan"]
    assert plan["final"] is True
    assert plan["words"] == round((80 - OVERHEAD_S - REMINDER_S) * WORDS_PER_SECOND)
    assert f"LENGTH: about {plan['words']} words" in llm.calls[-1]["user"]
    assert frames[-1]["status"] == "walk_complete"
    assert frames[-1]["complete"] is True
    assert "pad_ms" in frames[-1]
    # No "I'll leave you to wander" on the way out.
    assert WALK_HANDOFF not in _spoken_text(frames)


def test_generic_fallback_copy_is_served_once_not_on_a_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead model must not make every cycle repeat the same canned lines."""
    clock = FakeClock()
    _stub(monkeypatch, [fact(FID)], [])
    container = build_container(
        llm=ScriptedLLM([]),  # every call yields nothing
        tts=StubTTS(),
        sessions=SessionStore(clock=clock),
    )
    body = {"lat": 51.507, "lon": -0.127, "duration": 30, "session_id": "walk-x"}

    with client_for(container) as client:
        first = ndjson(client.post("/api/walk", json=body))
        clock.advance(90)
        second = ndjson(client.post("/api/walk", json=body))
        clock.advance(90)
        third = ndjson(client.post("/api/walk", json=body))

    assert first[-1]["used_static_fallback"] is True
    assert _spoken_text(first), "the canned copy is spoken once"
    for later in (second, third):
        assert _spoken_text(later) == []
        assert later[-1]["status"] == "cycle_complete"
        assert later[-1]["pause_ms"] >= MIN_PAUSE_MS
