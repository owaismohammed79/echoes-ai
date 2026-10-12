"""Tests for the Gemma 4 streaming provider and the narration orchestrator.

Run from ``server/``:  python -m pytest -q

No API key or network access is required: the provider is exercised through
``httpx.MockTransport`` and the orchestrator through a scripted stub, so every
fallback path is genuinely executed rather than assumed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence

import httpx
import pytest

from app.services import prompts
from app.services.gemma import LLMError, GemmaProvider, extract_text_parts
from app.services.narration import (
    MAX_STORY_SENTENCES,
    STATIC_NOTICE_LINES,
    generate_narration,
)
from app.services.prompts import PEDESTRIAN_REMINDER

FID = "9c1f2ab73e0d5144"
OTHER = "4be7c0a91d2f8836"


@dataclass
class FakeFact:
    fact_id: str
    title: str
    snippet: str
    distance_m: float
    source_url: str = "https://example.org/fact"
    kind: str = "monument"


@dataclass
class FakeNews:
    """Structural double for ``app.models.NewsItem``."""

    title: str
    snippet: str = ""
    source: str | None = "The Hindu"
    date: str | None = "2026-09-28T07:00:00Z"
    url: str | None = "https://example.org/story"


def fact(fid: str = FID, title: str = "Nelson's Column") -> FakeFact:
    return FakeFact(
        fact_id=fid,
        title=title,
        snippet=f"{title} is a monument in the square.",
        distance_m=21.6,
    )


def news_item(
    title: str = "Residents protest against Vijayanagar tree poisoning",
    *,
    snippet: str = "",
    source: str | None = "The New Indian Express",
    date: str | None = "2026-09-28T07:00:00Z",
) -> FakeNews:
    return FakeNews(title=title, snippet=snippet, source=source, date=date)


class ScriptedProvider:
    """Yields a scripted response per call; raises when scripted to."""

    def __init__(self, *responses: Sequence[str] | Exception) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, str]] = []

    async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
        index = len(self.calls)
        self.calls.append({"system": system, "user": user})
        response = self._responses[min(index, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        for chunk in response:
            yield chunk


def collect(agen: AsyncIterator[str]) -> list[str]:
    async def run() -> list[str]:
        return [chunk async for chunk in agen]

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# Gemma 4 streaming provider
# ---------------------------------------------------------------------------


def _sse(*frames: dict[str, Any]) -> bytes:
    import json

    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames)
    return body.encode("utf-8")


def _text_frame(text: str, *, thought: bool = False) -> dict[str, Any]:
    part: dict[str, Any] = {"text": text}
    if thought:
        part["thought"] = True
    return {"candidates": [{"content": {"role": "model", "parts": [part]}}]}


def _provider(handler: Any) -> GemmaProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GemmaProvider(api_key="test-key", client=client)


def test_provider_streams_text_from_sse_frames() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert ":streamGenerateContent" in str(request.url)
        return httpx.Response(
            200,
            content=_sse(_text_frame("Nelson's Column "), _text_frame("rises above.")),
        )

    chunks = collect(_provider(handler).stream(system="s", user="u"))
    assert chunks == ["Nelson's Column ", "rises above."]


def test_provider_never_emits_thought_parts() -> None:
    """Reasoning must not be narrated, even if the server ignores thinkingLevel."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                _text_frame("I should mention the column. ", thought=True),
                _text_frame("Nelson's Column rises above."),
            ),
        )

    chunks = collect(_provider(handler).stream(system="s", user="u"))
    assert chunks == ["Nelson's Column rises above."]
    assert all("I should" not in c for c in chunks)


def test_extract_text_parts_skips_thoughts_and_empty_parts() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "hidden"},
                        {"text": "visible"},
                        {"text": ""},
                    ]
                }
            }
        ]
    }
    assert extract_text_parts(payload) == ["visible"]


def test_provider_sends_system_instruction_and_minimal_thinking() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, content=_sse(_text_frame("ok.")))

    collect(_provider(handler).stream(system="SYSTEM", user="USER"))
    assert captured["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
    assert captured["contents"][0]["parts"][0]["text"] == "USER"
    assert captured["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"


def test_provider_raises_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b"quota exceeded")

    with pytest.raises(LLMError, match="HTTP 429"):
        collect(_provider(handler).stream(system="s", user="u"))


def test_provider_raises_on_an_error_frame() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse({"error": {"message": "boom"}}))

    with pytest.raises(LLMError, match="reported an error"):
        collect(_provider(handler).stream(system="s", user="u"))


def test_provider_wraps_transport_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="transport failure"):
        collect(_provider(handler).stream(system="s", user="u"))


def test_provider_requires_an_api_key() -> None:
    with pytest.raises(ValueError):
        GemmaProvider(api_key="")


def test_provider_defaults_to_the_verified_gemma_4_model() -> None:
    provider = GemmaProvider(api_key="k")
    assert provider._model == "gemma-4-26b-a4b-it"
    assert "/v1beta/models/gemma-4-26b-a4b-it:streamGenerateContent" in provider.endpoint


# ---------------------------------------------------------------------------
# Orchestrator: happy path
# ---------------------------------------------------------------------------


def test_story_success_returns_validated_sentences_and_reminder() -> None:
    provider = ScriptedProvider(
        [f"Nelson's Column rises over the square.[cite:{FID}]"]
    )
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )

    assert result.mode == "STORY"
    assert result.degraded is False
    assert result.used_static_fallback is False
    assert len(result.sentences) == 1
    assert f"[cite:{FID}]" not in result.spoken_text  # tags are never spoken
    assert result.spoken_text.endswith(PEDESTRIAN_REMINDER)
    assert result.source_urls == ("https://example.org/fact",)


def test_story_never_speaks_a_tag() -> None:
    provider = ScriptedProvider([f"A rises.[cite:{FID}] B sits near.[cite:{OTHER}]"])
    result = asyncio.run(
        generate_narration(
            mode="STORY",
            provider=provider,
            facts=[fact(), fact(OTHER, "Charing Cross")],
        )
    )
    assert "cite:" not in result.spoken_text
    assert result.spoken_text == (
        f"A rises. B sits near. {PEDESTRIAN_REMINDER}"
    )


def test_story_caps_sentence_count_and_records_overflow() -> None:
    """The cap bounds a runaway model, and reports that it did.

    Six, not three: the prompt asks for 4-6 sentences per chunk so Gemma has
    room to frame the weather and the walker's mood around the cited facts.
    """
    provider = ScriptedProvider(
        [
            f"A rises.[cite:{FID}] B rises.[cite:{FID}] "
            f"C rises.[cite:{FID}] D rises.[cite:{FID}] "
            f"E rises.[cite:{FID}] F rises.[cite:{FID}] G rises.[cite:{FID}]"
        ]
    )
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )
    assert len(result.sentences) == MAX_STORY_SENTENCES == 6
    assert result.rejections.count("overflow") == 1


# ---------------------------------------------------------------------------
# Orchestrator: degradation
# ---------------------------------------------------------------------------


def test_uncited_story_degrades_to_notice() -> None:
    """The core requirement: bad citations fall back to Notice Mode."""
    provider = ScriptedProvider(
        ["The square was laid out in the 1840s."],  # no citation -> dropped
        ["If you're somewhere safe to pause, notice your breathing."],  # NOTICE
    )
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )

    assert result.mode == "NOTICE"
    assert result.degraded is True
    assert result.used_static_fallback is False
    assert len(provider.calls) == 2
    assert provider.calls[0]["system"] == prompts.STORY_SYSTEM
    assert provider.calls[1]["system"] == prompts.NOTICE_SYSTEM
    assert result.rejections == ["no_citation"]


def test_fabricated_fact_id_degrades_to_notice() -> None:
    provider = ScriptedProvider(
        [f"Rises over the square.[cite:{OTHER}]"],
        ["If you're somewhere safe, notice the ground."],
    )
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )
    assert result.mode == "NOTICE"
    assert "unknown_fact_id" in result.rejections


def test_provider_error_degrades_to_notice() -> None:
    provider = ScriptedProvider(
        RuntimeError("gemma exploded"),
        ["If you're somewhere safe, notice the ground."],
    )
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )

    assert result.mode == "NOTICE"
    assert result.degraded is True
    assert result.error is not None and "gemma exploded" in result.error


def test_partial_stream_failure_keeps_validated_sentences() -> None:
    """Sentences cleared before the failure are still speakable."""

    class HalfBroken:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
            self.calls += 1
            yield f"Rises over the square.[cite:{FID}]"
            raise RuntimeError("stream died")

    provider = HalfBroken()
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )

    assert result.mode == "STORY"
    assert provider.calls == 1  # no inline retry
    assert len(result.sentences) == 1
    assert result.error is not None


def test_total_failure_serves_static_copy() -> None:
    provider = ScriptedProvider(RuntimeError("down"), RuntimeError("down again"))
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )

    assert result.mode == "NOTICE"
    assert result.used_static_fallback is True
    assert result.degraded is True
    assert [s.text for s in result.sentences] == list(STATIC_NOTICE_LINES)
    assert result.spoken_text.endswith(PEDESTRIAN_REMINDER)


def test_invalid_notice_output_falls_back_to_static_copy() -> None:
    provider = ScriptedProvider(
        ["The square is old and fine."],
        ["The pavement is grey."],  # not an invitation -> dropped
    )
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )
    assert result.used_static_fallback is True


def test_degradation_is_attempted_only_once() -> None:
    """A strict validator must not become an unbounded retry loop."""
    provider = ScriptedProvider(["uncited prose."], ["The pavement is grey."])
    asyncio.run(generate_narration(mode="STORY", provider=provider, facts=[fact()]))
    assert len(provider.calls) == 2  # STORY once, NOTICE once, then static


# ---------------------------------------------------------------------------
# Orchestrator: mode handling
# ---------------------------------------------------------------------------


def test_notice_mode_skips_the_story_attempt() -> None:
    provider = ScriptedProvider(["If you're somewhere safe, notice your breath."])
    result = asyncio.run(
        generate_narration(mode="NOTICE", provider=provider, facts=[fact()])
    )

    assert len(provider.calls) == 1
    assert provider.calls[0]["system"] == prompts.NOTICE_SYSTEM
    assert result.mode == "NOTICE"
    assert result.degraded is False


def test_story_mode_with_no_facts_goes_straight_to_notice() -> None:
    provider = ScriptedProvider(["If you're somewhere safe, notice your breath."])
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[])
    )

    assert len(provider.calls) == 1
    assert provider.calls[0]["system"] == prompts.NOTICE_SYSTEM
    assert result.degraded is True


def test_notice_prompt_carries_time_but_never_weather() -> None:
    provider = ScriptedProvider(["If you're somewhere safe, notice your breath."])
    notice = prompts.NoticeContext.from_hour(7, 42)
    asyncio.run(generate_narration(mode="NOTICE", provider=provider, notice=notice))

    user = provider.calls[0]["user"]
    assert "local_time: 07:42" in user
    assert "daypart: early morning" in user
    assert "weather" not in user.lower()
    # The system prompt forbids weather outright rather than permitting it.
    assert "Do not mention weather" in prompts.NOTICE_SYSTEM


@pytest.mark.parametrize(
    ("hour", "daypart"),
    [(6, "early morning"), (10, "morning"), (13, "midday"), (15, "afternoon"),
     (18, "evening"), (21, "night"), (2, "late night")],
)
def test_daypart_derivation(hour: int, daypart: str) -> None:
    assert prompts.NoticeContext.from_hour(hour).daypart == daypart


def test_story_prompt_carries_facts_without_source_urls() -> None:
    """source_url is withheld so the model has nothing to mistype."""
    user = prompts.build_story_user_prompt([fact()])
    assert f"id: {FID}" in user
    assert "Nelson's Column is a monument" in user
    assert "https://" not in user


def test_story_prompt_carries_mood_and_conditions() -> None:
    class W:
        temp = 12.0
        condition = "light rain"
        sunset = "6:42 PM"
        local_time = "3:15 PM"

    user = prompts.build_story_user_prompt([fact()], mood="sad", weather=W())

    assert "mood: sad" in user
    assert "resilience" in user
    assert "uplifting" in user
    assert "light rain" in user
    assert "sunset: 6:42 PM" in user


def test_story_prompt_labels_the_three_blocks() -> None:
    """The three blocks are the contract with the model, so pin their names."""

    class W:
        temp = 12.0
        condition = "light rain"
        sunset = "6:42 PM"
        local_time = "3:15 PM"

    user = prompts.build_story_user_prompt([fact()], mood="sad", weather=W())
    for block in (
        "ENVIRONMENT CONTEXT",
        "USER MOOD & EMOTIONAL GUIDANCE",
        "AVAILABLE FACTS",
    ):
        assert block in user
    # The conditions are labelled fields, not a comma-joined line to unpick.
    assert "time of day: 3:15 PM" in user
    # Each fact is tagged with the exact syntax the validator reads, and says
    # what it is, so the narrator can name the kind of place it is passing.
    assert f"tag: [cite:{FID}]" in user
    assert "kind: monument" in user
    # The system prompt asks the model to name the light when the walk is close
    # to sunset; without that, a supplied sunset time is context never used.
    assert "sunset time" in prompts.STORY_SYSTEM


def test_story_prompt_states_the_same_framing_allowance_the_validator_grants() -> None:
    """A prompt that offers more uncited sentences than the validator admits --
    or fewer -- is a prompt that lies about the contract."""
    from app.services import narration

    assert narration.MAX_UNCITED_FRAMING_SENTENCES == (
        prompts.MAX_UNCITED_FRAMING_SENTENCES
    )
    assert (
        f"At most {prompts.MAX_UNCITED_FRAMING_SENTENCES} sentences may carry no citation"
        in prompts.STORY_SYSTEM
    )
    assert prompts.STORY_SENTENCES in prompts.STORY_SYSTEM
    assert narration.MAX_STORY_SENTENCES >= 6


def test_story_prompt_without_a_weather_reading_forbids_guessing() -> None:
    user = prompts.build_story_user_prompt([fact()], mood=None, weather=None)
    assert "do not describe or guess" in user
    assert "mood: none selected" in user


def test_story_prompt_states_the_same_news_allowance_the_validator_grants() -> None:
    """Same contract as the framing allowance, for the same reason."""
    from app.services import narration

    assert narration.MAX_NEWS_SENTENCES == prompts.MAX_NEWS_REFERENCES
    assert f"at most {prompts.MAX_NEWS_REFERENCES} per chunk" in prompts.STORY_SYSTEM


def test_story_prompt_labels_the_news_block_only_when_there_is_news() -> None:
    """No news means no block: an empty one invites comment on its emptiness."""
    assert "RECENT LOCAL NEWS" not in prompts.build_story_user_prompt([fact()])

    user = prompts.build_story_user_prompt(
        [fact()],
        news=[
            news_item(
                "Residents protest against Vijayanagar tree poisoning",
                snippet="Footpath access has narrowed.",
                source="The New Indian Express",
            )
        ],
    )

    assert "RECENT LOCAL NEWS" in user
    assert (
        "Headline: Residents protest against Vijayanagar tree poisoning"
        " - Footpath access has narrowed." in user
    )
    # Without a summary the publisher and date stand in, so the model is never
    # handed a headline with an empty detail.
    fallback = prompts.format_news_block([news_item(snippet="")])
    assert " - The New Indian Express, 2026-09-28T07:00:00Z" in fallback
    assert (
        " - no summary reported"
        in prompts.format_news_block(
            [news_item(snippet="", source=None, date=None)]
        )
    )


def test_build_greeting_uses_condition_and_daypart() -> None:
    assert prompts.build_greeting(daypart="afternoon", condition="Clear") == (
        "Hello there. It's a clear afternoon here. Let's begin our walk."
    )
    assert prompts.build_greeting(daypart="night") == (
        "Hello there. It's night here. Let's begin our walk."
    )
    # The provider's vocabulary has vowel-initial conditions, and the greeting is
    # spoken: "a overcast" is the kind of error a walker notices immediately.
    assert prompts.build_greeting(daypart="evening", condition="overcast") == (
        "Hello there. It's an overcast evening here. Let's begin our walk."
    )


def test_greeting_and_handoff_are_injected_outside_the_validator() -> None:
    """Both are code constants with no citation, so both bypass validation."""
    provider = ScriptedProvider([f"A rises.[cite:{FID}]"])
    result = asyncio.run(
        generate_narration(
            mode="STORY",
            provider=provider,
            facts=[fact()],
            greeting="Hello there. It's a calm afternoon here. Let's begin our walk.",
            handoff="I'll leave you to wander for a few minutes. Just speak if you need me.",
        )
    )
    assert result.sentences[0].text.startswith("Hello there.")
    assert result.sentences[-1].text.startswith("I'll leave you to wander")
    assert result.sentences[0].citations == ()
    assert result.sentences[-1].citations == ()
    assert result.spoken_text.startswith("Hello there.")


# ---------------------------------------------------------------------------
# Orchestrator: framing sentences and repetition
# ---------------------------------------------------------------------------

ATMOSPHERE = "The evening light is soft here."
ENCOURAGEMENT = "Whatever you are carrying today, this city has carried it before."


def test_story_speaks_framing_sentences_uncited() -> None:
    """Atmosphere and mood survive validation instead of being stripped away.

    The regression this pins: an uncited opener and an uncited line of
    encouragement used to be deleted, so a sad walker heard two flat facts and
    nothing that answered how they felt.
    """
    provider = ScriptedProvider(
        [
            f"{ATMOSPHERE} A rises over the square.[cite:{FID}] "
            f"{ENCOURAGEMENT} B sits nearby.[cite:{OTHER}]"
        ]
    )
    result = asyncio.run(
        generate_narration(
            mode="STORY",
            provider=provider,
            facts=[fact(), fact(OTHER, "Charing Cross")],
            mood="sad",
        )
    )

    assert result.mode == "STORY"
    assert [s.text for s in result.sentences] == [
        ATMOSPHERE,
        "A rises over the square.",
        ENCOURAGEMENT,
        "B sits nearby.",
    ]
    assert result.sentences[0].citations == ()
    assert result.sentences[2].citations == ()
    assert result.sentences[1].citations == (FID,)
    assert result.spoken_text.startswith(ATMOSPHERE)


def test_uncited_framing_is_capped_per_chunk() -> None:
    """The exemption is bounded: the third uncited sentence is dropped."""
    provider = ScriptedProvider(
        [
            "The evening light is soft here. The air is cool tonight. "
            f"The breeze is gentle today. A rises.[cite:{FID}]"
        ]
    )
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )

    assert [s.text for s in result.sentences] == [
        "The evening light is soft here.",
        "The air is cool tonight.",
        "A rises.",
    ]
    assert result.rejections == ["no_citation"]


def test_the_framing_allowance_does_not_admit_an_uncited_claim() -> None:
    """The exemption is for framing, not for facts that lost their citation."""
    provider = ScriptedProvider(
        ["The square was laid out in the 1840s. ", f"A rises.[cite:{FID}]"]
    )
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )

    assert result.mode == "STORY"
    assert [s.text for s in result.sentences] == ["A rises."]
    assert result.rejections == ["no_citation"]


def test_a_repeated_sentence_is_dropped_as_a_duplicate() -> None:
    provider = ScriptedProvider(
        [f"A rises.[cite:{FID}] A rises.[cite:{FID}] B sits near.[cite:{FID}]"]
    )
    result = asyncio.run(
        generate_narration(mode="STORY", provider=provider, facts=[fact()])
    )

    assert [s.text for s in result.sentences] == ["A rises.", "B sits near."]
    assert result.rejections == ["duplicate"]


def test_a_repeat_is_caught_across_chunks_and_despite_the_tag() -> None:
    """Same words, different citation: still the same sentence to a listener."""
    provider = ScriptedProvider(
        [f"A rises.[cite:{FID}]", f"A rises .[cite:{OTHER}]"]
    )
    result = asyncio.run(
        generate_narration(
            mode="STORY", provider=provider, facts=[fact(), fact(OTHER, "Charing")]
        )
    )

    assert [s.text for s in result.sentences] == ["A rises."]
    assert result.rejections == ["duplicate"]


#: A line the narrator is told to write: one headline, named once, framed as a
#: passing thought.
NEWS_LINE = (
    "With the recent buzz about the tree-poisoning row in the area, "
    "the streets here feel restless."
)


def test_the_news_line_is_spoken_without_its_marker() -> None:
    """The marked sentence is exempted from citations; the marker is not spoken."""
    provider = ScriptedProvider(
        [f"{NEWS_LINE}[news] A rises over the square.[cite:{FID}]"],
        ["unused"],
    )
    result = asyncio.run(
        generate_narration(
            mode="STORY",
            provider=provider,
            facts=[fact()],
            news=[news_item()],
        )
    )

    assert result.mode == "STORY"
    assert [s.text for s in result.sentences] == [NEWS_LINE, "A rises over the square."]
    assert result.sentences[0].citations == ()
    assert result.sentences[0].news is True
    assert "[news]" not in result.spoken_text
    assert result.rejections == []
    # The headlines reached the model under their own label, not as facts.
    assert "RECENT LOCAL NEWS" in provider.calls[0]["user"]
    assert "Headline: Residents protest" in provider.calls[0]["user"]
    assert len(provider.calls) == 1  # nothing degraded


def test_a_news_line_without_the_marker_is_still_dropped() -> None:
    """Sounding like a bulletin does not earn a sentence the exemption.

    This is the whole difference between the news exemption and the framing
    heuristic: the model has to say *which* ground the sentence rests on, so a
    headline-shaped claim that lost its citation is deleted like any other.
    """
    provider = ScriptedProvider(
        [f"{NEWS_LINE} A rises over the square.[cite:{FID}]"], ["unused"]
    )
    result = asyncio.run(
        generate_narration(
            mode="STORY",
            provider=provider,
            facts=[fact()],
            news=[news_item()],
        )
    )

    assert [s.text for s in result.sentences] == ["A rises over the square."]
    assert result.rejections == ["no_citation"]


def test_the_news_allowance_is_one_sentence_per_chunk() -> None:
    """One event, alluded to once: a second marked sentence is a bulletin."""
    second = (
        "There is more chatter about the banners on the main road this week."
    )
    provider = ScriptedProvider(
        [
            f"{NEWS_LINE}[news] {second}[news] "
            f"A rises over the square.[cite:{FID}]"
        ],
        ["unused"],
    )
    result = asyncio.run(
        generate_narration(
            mode="STORY",
            provider=provider,
            facts=[fact()],
            news=[news_item()],
        )
    )

    assert [s.text for s in result.sentences] == [NEWS_LINE, "A rises over the square."]
    assert result.rejections == ["news_overflow"]


def test_a_news_sentence_spends_a_framing_slot_too() -> None:
    """The chunk's total uncited count is what it was: two."""
    provider = ScriptedProvider(
        [
            f"{NEWS_LINE}[news] {ATMOSPHERE} {ENCOURAGEMENT} "
            f"A rises over the square.[cite:{FID}]"
        ],
        ["unused"],
    )
    result = asyncio.run(
        generate_narration(
            mode="STORY",
            provider=provider,
            facts=[fact()],
            news=[news_item()],
        )
    )

    assert [s.text for s in result.sentences] == [
        NEWS_LINE,
        ATMOSPHERE,
        "A rises over the square.",
    ]
    assert result.rejections == ["no_citation"]


# ---------------------------------------------------------------------------
# End to end: real SSE transport -> assembler -> validator -> fallback
# ---------------------------------------------------------------------------

def test_end_to_end_sse_through_validator_drops_only_bad_sentences() -> None:
    """The whole chain, with a real GemmaProvider reading real SSE frames."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                _text_frame(f"Nelson's Column rises over the square.[cite:{FID}] "),
                _text_frame("It was built long ago. "),  # uncited
                _text_frame(f"Turn left at the corner.[cite:{FID}] "),  # direction
                _text_frame(f"Charing Cross sits nearby.[cite:{OTHER}] "),  # unknown id
            ),
        )

    result = asyncio.run(
        generate_narration(
            mode="STORY", provider=_provider(handler), facts=[fact()]
        )
    )

    assert result.mode == "STORY"
    assert [s.text for s in result.sentences] == [
        "Nelson's Column rises over the square."
    ]
    assert result.rejections == [
        "no_citation",
        "direction_language",
        "unknown_fact_id",
    ]
    assert "cite:" not in result.spoken_text
    assert result.spoken_text.endswith(PEDESTRIAN_REMINDER)


def test_end_to_end_sse_error_degrades_to_notice() -> None:
    """A quota error in the SSE stream must surface as Notice Mode, not a 500."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, content=_sse({"error": {"message": "quota"}}))
        return httpx.Response(
            200,
            content=_sse(
                _text_frame(
                    "If you're somewhere safe to pause, notice the ground underfoot."
                )
            ),
        )

    result = asyncio.run(
        generate_narration(
            mode="STORY", provider=_provider(handler), facts=[fact()]
        )
    )

    assert calls["n"] == 2  # STORY attempt, then the single NOTICE fallback
    assert result.mode == "NOTICE"
    assert result.degraded is True
    assert result.used_static_fallback is False
    assert result.error is not None and "quota" in result.error
    assert "notice" in result.spoken_text.lower()
