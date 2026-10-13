"""Tests for ``POST /api/walk`` (Channel A) and ``GET /api/audio`` (Channel B).

The two channels are deliberately separate, and these tests pin down the
contract between them: every sentence frame carries tag-free text plus a
single-use ``audio_url``, the reminder is a static asset that never touches the
TTS provider, and an unconfigured provider flips ``use_native_tts`` proactively
so the client never issues a doomed request.
"""

from __future__ import annotations

import pytest

from app.models import NewsItem
from app.services import news as news_service
from app.services import parks, weather, wikipedia
from app.services.audio_tokens import AudioBlob
from app.services.prompts import PEDESTRIAN_REMINDER
from tests.helpers import (
    FAKE_AUDIO,
    FID,
    OTHER,
    ScriptedLLM,
    StubTTS,
    build_container,
    client_for,
    fact,
    frame_text,
    ndjson,
    spoken,
)

STORY_RESPONSE = (
    f"Nelson's Column rises over the square.[cite:{FID}] "
    f"Charing Cross sits a short walk away.[cite:{OTHER}]"
)

NOTICE_RESPONSE = (
    "If you're somewhere safe to pause, notice what your feet are resting on. "
    "When you're ready, listen for the nearest sound."
)


def _stub_sources(monkeypatch: pytest.MonkeyPatch, wiki, park) -> list[int | None]:
    """Pin every fact source, so a walk test never touches the network.

    The Wikipedia double serves every rung of the widening ladder with the same
    facts and records the radius it was asked for, so a test can assert both
    what came back and how wide the search had to go to find it. The city
    history fallback is pinned too: it is the ladder's last rung, and a walk
    test that left it live would depend on the network the moment its stub
    returned fewer than five facts.
    """
    seen_radii: list[int | None] = []

    async def wiki_facts(client, lat, lon, *, radius_m: int | None = None):
        seen_radii.append(radius_m)
        return wiki

    async def park_facts(client, lat, lon):
        return park

    async def no_history(client, lat, lon, *, radius_m: int | None = None):
        return []

    monkeypatch.setattr(wikipedia, "fetch_wikipedia_facts", wiki_facts)
    monkeypatch.setattr(wikipedia, "fetch_wikipedia_history_facts", no_history)
    monkeypatch.setattr(parks, "fetch_park_facts", park_facts)

    return seen_radii


# ---------------------------------------------------------------------------
# Channel A -- narration metadata
# ---------------------------------------------------------------------------


def test_walk_streams_sentence_frames_with_audio_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_sources(
        monkeypatch,
        [fact(FID, "Nelson's Column"), fact(OTHER, "Charing Cross")],
        [],
    )
    tts = StubTTS()
    container = build_container(llm=ScriptedLLM([STORY_RESPONSE]), tts=tts)

    with client_for(container) as client:
        resp = client.post("/api/walk", json={"lat": 51.5074, "lon": -0.1278})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    frames = ndjson(resp)
    texts = spoken(frames)
    assert frames[0]["type"] == "data"
    assert frames[-1]["type"] == "indicator"
    assert frames[-1]["status"] == "walk_complete"
    assert any(f.get("role") == "reminder" for f in frames)

    # The greeting is spoken first and injected in code, not generated: an
    # uncited greeting sentence would be deleted by the validator.
    greeting = texts[0]
    assert frame_text(greeting).startswith("Hello there.")
    assert greeting["citations"] == []
    assert greeting["audio_url"].startswith("/api/audio/")

    first = texts[1]
    assert first["index"] == 1
    assert frame_text(first) == "Nelson's Column rises over the square."
    assert first["citations"] == [FID]
    assert first["source_urls"] == ["https://en.wikipedia.org/wiki/Nelson%27s_Column"]
    assert first["use_native_tts"] is False
    assert first["audio_url"].startswith("/api/audio/")
    assert first["mode"] == "STORY"

    # The pacing hand-off closes the block, also injected in code.
    assert frame_text(texts[3]).startswith("I'll leave you to wander")

    # The tag never reaches the synthesis layer.
    assert "[cite:" not in "".join(tts.spoken)
    assert tts.spoken[1] == "Nelson's Column rises over the square."


def test_walk_greeting_names_the_real_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opener is grounded in the provider's reading, not a guess."""
    from app.models import WeatherInfo

    _stub_sources(monkeypatch, [fact(FID)], [])

    async def conditions(client, lat, lon):
        return WeatherInfo(
            temp=9.0, condition="light rain", sunset="6:42 PM", local_time="3:15 PM"
        )

    monkeypatch.setattr(weather, "fetch_weather", conditions)
    container = build_container(llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), tts=StubTTS())

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 51.507, "lon": -0.127}))

    greeting = frame_text(spoken(frames)[0])
    assert greeting.startswith("Hello there.")
    assert "light rain" in greeting
    # Conditions travel in the done frame for the client too.
    assert frames[-1]["weather"]["condition"] == "light rain"


def test_walk_greeting_uses_the_walkers_clock_not_the_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 7 PM walk in Anchorage must not be greeted as a morning one.

    The daypart used to come from the server's clock, so a walker on another
    meridian heard a greeting about the wrong time of day -- context about them
    that is simply wrong, from the pipeline that exists to avoid exactly that.
    """
    from app.models import WeatherInfo

    _stub_sources(monkeypatch, [fact(FID)], [])

    async def conditions(client, lat, lon):
        return WeatherInfo(
            temp=6.3, condition="light drizzle", sunset="6:59 PM", local_time="7:00 PM"
        )

    monkeypatch.setattr(weather, "fetch_weather", conditions)
    container = build_container(llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), tts=StubTTS())

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 61.218, "lon": -149.9}))

    greeting = frame_text(spoken(frames)[0])
    assert greeting == "Hello there. It's a light drizzle evening here. Let's begin our walk."


def test_walk_mood_reaches_the_story_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_sources(monkeypatch, [fact(FID)], [])
    llm = ScriptedLLM([f"A claim.[cite:{FID}]"])
    container = build_container(llm=llm, tts=StubTTS())

    with client_for(container) as client:
        client.post(
            "/api/walk",
            json={"lat": 51.507, "lon": -0.127, "mood": "sad", "duration": 15},
        )

    prompt = llm.calls[0]["user"]
    assert "mood: sad" in prompt
    assert "resilience" in prompt


def test_walk_stream_carries_conditions_mood_and_five_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acceptance path: a residential coordinate and a walker who is sad.

    Everything the bug report was about, asserted on the NDJSON the client
    actually receives: the conditions and the mood reach Gemma's prompt, five
    cited sentences survive validation, the atmospheric opener survives
    uncited, and nothing in the queue is repeated or empty.
    """
    from app.models import WeatherInfo

    fids = [f"{n:016x}" for n in range(1, 6)]
    facts = [fact(fid, f"Landmark {n}") for n, fid in enumerate(fids, start=1)]
    scripted = "The evening light is soft here. " + " ".join(
        f"Landmark {n} keeps its own quiet story.[cite:{fid}]"
        for n, fid in enumerate(fids, start=1)
    )
    _stub_sources(monkeypatch, facts, [])

    async def conditions(client, lat, lon):
        return WeatherInfo(
            temp=24.0,
            condition="light rain",
            sunset="6:12 PM",
            local_time="5:40 PM",
        )

    monkeypatch.setattr(weather, "fetch_weather", conditions)
    llm = ScriptedLLM([scripted])
    container = build_container(llm=llm, tts=StubTTS())

    with client_for(container) as client:
        resp = client.post(
            "/api/walk",
            json={
                "lat": 12.9638,
                "lon": 77.548,
                "mood": "Sad/Struggling",
            },
        )

    frames = ndjson(resp)
    sentences = spoken(frames)
    cited = [f for f in sentences if f["citations"]]
    texts = [frame_text(f) for f in sentences]

    # 1. At least five grounded sentences, each with a source to show.
    assert len(cited) >= 5
    assert all(f["source_urls"] for f in cited)

    # 2. The conditions reached Gemma, in labelled blocks.
    prompt = llm.calls[0]["user"]
    for block in (
        "ENVIRONMENT CONTEXT",
        "USER MOOD & EMOTIONAL GUIDANCE",
        "AVAILABLE FACTS",
    ):
        assert block in prompt
    assert "light rain" in prompt
    assert "time of day: 5:40 PM" in prompt
    assert "sunset: 6:12 PM" in prompt

    # 3. The mood arrived as guidance to lean on resilience, not as a label.
    assert "mood: Sad/Struggling" in prompt
    assert "uplifting" in prompt and "resilience" in prompt

    # 4. The atmospheric opener survived, uncited and without a source.
    opener = next(
        f for f in sentences if frame_text(f) == "The evening light is soft here."
    )
    assert opener["citations"] == []
    assert opener["source_urls"] == []

    # 5. Nothing repeated, nothing empty, and the walk closes without dropping.
    assert len(texts) == len(set(texts))
    assert all(t.strip() for t in texts)
    assert frames[-1]["type"] == "indicator"
    assert frames[-1]["status"] == "walk_complete"
    assert frames[-1]["rejections"] == []


def test_walk_speaks_the_mood_encouragement_line_uncited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line written for the walker's mood is not stripped for lacking a tag."""
    encouragement = (
        "Whatever you are carrying today, this city has carried it before."
    )
    fids = [f"{n:016x}" for n in range(1, 5)]
    facts = [fact(fid, f"Landmark {n}") for n, fid in enumerate(fids, start=1)]
    scripted = (
        "The evening light is soft here. "
        + " ".join(
            f"Landmark {n} keeps its own quiet story.[cite:{fid}]"
            for n, fid in enumerate(fids, start=1)
        )
        + f" {encouragement}"
    )
    _stub_sources(monkeypatch, facts, [])
    container = build_container(llm=ScriptedLLM([scripted]), tts=StubTTS())

    with client_for(container) as client:
        frames = ndjson(
            client.post(
                "/api/walk",
                json={"lat": 12.9638, "lon": 77.548, "mood": "sad"},
            )
        )

    line = next(
        f for f in spoken(frames) if frame_text(f) == encouragement
    )
    assert line["citations"] == []
    assert frames[-1]["mode"] == "STORY"


def test_walk_weaves_a_local_news_reference_the_validator_keeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The brief's end state: a casually framed news line reaches the audio.

    Three things are checked together, because they fail together: the headline
    arrives in the prompt under its own label, the marked sentence survives
    validation instead of being dropped as uncited content, and the marker
    itself is never spoken.
    """
    fids = [f"{n:016x}" for n in range(1, 5)]
    facts = [fact(fid, f"Landmark {n}") for n, fid in enumerate(fids, start=1)]
    news_line = (
        "With the recent buzz about the tree-poisoning row in the area, "
        "the streets here feel restless."
    )
    scripted = (
        f"{news_line}[news] "
        + " ".join(
            f"Landmark {n} keeps its own quiet story.[cite:{fid}]"
            for n, fid in enumerate(fids, start=1)
        )
        + " The evening is yours to wander through."
    )

    async def headlines(
        client, lat: float, lon: float, *, place_hint: str | None = None
    ) -> list[NewsItem]:
        return [
            NewsItem(
                title="Residents protest against Vijayanagar tree poisoning",
                source="The New Indian Express",
                date="2026-09-28T07:00:00Z",
            )
        ]

    _stub_sources(monkeypatch, facts, [])
    monkeypatch.setattr(news_service, "fetch_local_news", headlines)
    llm = ScriptedLLM([scripted])
    container = build_container(llm=llm, tts=StubTTS())

    with client_for(container) as client:
        frames = ndjson(
            client.post(
                "/api/walk", json={"lat": 12.9638, "lon": 77.548, "mood": "calm"}
            )
        )

    line = next(f for f in spoken(frames) if frame_text(f) == news_line)
    assert line["citations"] == []
    assert "[news]" not in frame_text(line)

    prompt = llm.calls[0]["user"]
    assert "RECENT LOCAL NEWS" in prompt
    assert "Headline: Residents protest against Vijayanagar tree poisoning" in prompt

    # Nothing was dropped to achieve it.
    assert frames[-1]["type"] == "indicator"
    assert frames[-1]["rejections"] == []
    assert len([f for f in spoken(frames) if f["citations"]]) >= 4


def test_walk_stores_the_narration_for_followup_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A follow-up question must see what the walk just narrated."""
    _stub_sources(monkeypatch, [fact(FID, "Nelson's Column")], [])
    container = build_container(llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), tts=StubTTS())

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 51.507, "lon": -0.127}))

    session = container.sessions.get(frames[-1]["session_id"])
    assert session is not None
    assert "A claim." in session.narration_text
    assert session.narration_text.startswith("Hello there.")


def test_walk_appends_the_mandatory_reminder_as_a_static_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_sources(monkeypatch, [fact(FID)], [])
    tts = StubTTS()
    container = build_container(llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), tts=tts)

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 51.507, "lon": -0.127}))

    reminder = next(f for f in frames if f.get("role") == "reminder" and f["type"] == "text")
    assert frame_text(reminder) == PEDESTRIAN_REMINDER
    # `.wav`, and a file that exists: the path used to point at `/static/
    # reminder.mp3`, which was never mounted and never on disk, so the reminder
    # frame promised the client a 404. tests/test_limits.py now fetches it.
    assert reminder["audio_url"] == "/static/reminder.wav"
    # Pre-rendered: the reminder was never synthesised.
    assert PEDESTRIAN_REMINDER not in tts.spoken


def test_walk_done_frame_reports_session_and_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_sources(
        monkeypatch,
        [fact(FID, "Nelson's Column"), fact(OTHER, "Charing Cross")],
        [],
    )
    container = build_container(llm=ScriptedLLM([STORY_RESPONSE]), tts=StubTTS())

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 51.507, "lon": -0.127}))

    done = frames[-1]
    assert done["type"] == "indicator"
    assert done["status"] == "walk_complete"
    assert done["mode"] == "STORY"
    assert done["degraded"] is False
    assert done["used_static_fallback"] is False
    assert done["rejections"] == []
    # The session exists, and it holds the facts the narration was grounded in.
    session = container.sessions.get(done["session_id"])
    assert session is not None
    assert [f.fact_id for f in session.facts] == [FID, OTHER]


def test_walk_goes_to_notice_when_there_are_no_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_sources(monkeypatch, [], [])
    container = build_container(llm=ScriptedLLM([NOTICE_RESPONSE]), tts=StubTTS())

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 0.0, "lon": 0.0}))

    assert frames[-1]["mode"] == "NOTICE"
    assert all(f.get("citations", []) == [] for f in frames)


def test_walk_flags_native_tts_proactively_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client must be told before it asks for audio it will not get."""
    _stub_sources(monkeypatch, [fact(FID)], [])
    container = build_container(llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), tts=None)

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 51.507, "lon": -0.127}))

    sentence = spoken(frames)[0]
    assert sentence["use_native_tts"] is True
    assert "audio_url" not in sentence
    assert sentence["tts_reason"] == "tts_not_configured"


def test_walk_degrades_to_static_copy_without_an_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_sources(monkeypatch, [fact(FID)], [])
    container = build_container(llm=None, tts=StubTTS())

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 51.507, "lon": -0.127}))

    done = frames[-1]
    assert done["used_static_fallback"] is True
    assert done["degraded"] is True
    assert any(f["type"] == "text" for f in frames)


def test_walk_rejects_out_of_range_coordinates() -> None:
    container = build_container()
    with client_for(container) as client:
        resp = client.post("/api/walk", json={"lat": 120.0, "lon": 0.0})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Channel B -- the audio relay
# ---------------------------------------------------------------------------


def _issue(container, *, session_id: str = "session-a") -> str:
    return container.audio_tokens.issue(
        session_id=session_id,
        blob=AudioBlob(data=FAKE_AUDIO, content_type="audio/mpeg", text="Hello."),
    )


def test_audio_is_served_once_then_refused() -> None:
    container = build_container()
    token = _issue(container)

    with client_for(container) as client:
        first = client.get(f"/api/audio/{token}", headers={"x-session-id": "session-a"})
        second = client.get(f"/api/audio/{token}", headers={"x-session-id": "session-a"})

    assert first.status_code == 200
    assert first.headers["content-type"] == "audio/mpeg"
    assert first.headers["cache-control"] == "private, no-store"
    assert first.content == FAKE_AUDIO

    assert second.status_code == 404


def test_audio_rejects_a_foreign_session() -> None:
    container = build_container()
    token = _issue(container, session_id="session-a")

    with client_for(container) as client:
        resp = client.get(f"/api/audio/{token}", headers={"x-session-id": "session-b"})

    assert resp.status_code == 404


def test_audio_never_synthesises_an_unknown_token() -> None:
    """Otherwise the relay is a free text-to-speech service for anyone."""
    tts = StubTTS()
    container = build_container(tts=tts)

    with client_for(container) as client:
        resp = client.get("/api/audio/guessed-token", headers={"x-session-id": "x"})

    assert resp.status_code == 404
    assert tts.spoken == []


def test_walk_issued_token_is_redeemable_by_its_own_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end-to-end path: frame -> audio_url -> bytes."""
    _stub_sources(monkeypatch, [fact(FID)], [])
    container = build_container(llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), tts=StubTTS())

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 51.507, "lon": -0.127}))
        session_id = frames[-1]["session_id"]
        audio_url = spoken(frames)[0]["audio_url"]
        audio = client.get(audio_url, headers={"x-session-id": session_id})

    assert audio.status_code == 200
    assert audio.content == FAKE_AUDIO


def test_walk_issued_token_is_not_redeemable_by_another_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_sources(monkeypatch, [fact(FID)], [])
    container = build_container(llm=ScriptedLLM([f"A claim.[cite:{FID}]"]), tts=StubTTS())

    with client_for(container) as client:
        frames = ndjson(client.post("/api/walk", json={"lat": 51.507, "lon": -0.127}))
        audio_url = spoken(frames)[0]["audio_url"]
        audio = client.get(audio_url, headers={"x-session-id": "someone-else"})

    assert audio.status_code == 404
