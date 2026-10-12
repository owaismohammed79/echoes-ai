"""Tests for sentence assembly and citation validation.

Run from ``server/``:  python -m pytest -q
"""

from __future__ import annotations

import pytest

from app.services import prompts
from app.validation import (
    CITATION_PATTERN,
    RejectionReason,
    SentenceAssembler,
    SentenceRejected,
    extract_citation_ids,
    format_citation_tag,
    format_news_tag,
    normalize,
    strip_citation_tags,
    validate_notice_sentence,
    validate_story_sentence,
)

FID = "9c1f2ab73e0d5144"
OTHER = "4be7c0a91d2f8836"
SOURCES = {FID: "https://en.wikipedia.org/wiki/Nelson%27s_Column"}


# ---------------------------------------------------------------------------
# The prompt/validator contract
# ---------------------------------------------------------------------------


def test_prompt_tag_format_is_validator_readable() -> None:
    """The one failure that would silently drop every sentence in production."""
    tag = format_citation_tag(FID)
    assert CITATION_PATTERN.fullmatch(tag)
    assert extract_citation_ids(tag) == [FID]
    prompts.assert_prompt_matches_validator()


def test_prompt_documents_the_real_tag_syntax() -> None:
    """Guards against the prompt and the parser drifting apart."""
    assert format_citation_tag(FID) in prompts.STORY_SYSTEM
    assert format_news_tag() in prompts.STORY_SYSTEM
    assert "[[cite" not in prompts.STORY_SYSTEM
    assert "[[" not in prompts.STORY_SYSTEM


@pytest.mark.parametrize(
    "tag",
    [
        "[cite:abc123]",
        "[cite:  abc123]",
        "[cite:A_b-c]",
        "[cite:9c1f2ab73e0d5144]",
    ],
)
def test_citation_pattern_accepts_expected_shapes(tag: str) -> None:
    assert CITATION_PATTERN.fullmatch(tag)


@pytest.mark.parametrize(
    "tag",
    ["[cite:]", "[cite:abc def]", "cite:abc", "[cite:abc", "(cite:abc)"],
)
def test_citation_pattern_rejects_malformed_shapes(tag: str) -> None:
    assert CITATION_PATTERN.fullmatch(tag) is None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_folds_fullwidth_and_curly_quotes() -> None:
    assert normalize("\uff3bcite:abc\uff3d") == "[cite:abc]"
    assert normalize("Walker\u2019s rest") == "Walker's rest"
    assert normalize("  lots   of\tspace ") == "lots of space"


# ---------------------------------------------------------------------------
# Sentence assembly
# ---------------------------------------------------------------------------


def test_assembler_splits_on_terminators() -> None:
    asm = SentenceAssembler()
    assert asm.push("One. Two! Three? ") == ["One.", "Two!", "Three?"]


def test_assembler_splits_when_a_tag_follows_the_period() -> None:
    """The case a naive 'terminator + whitespace' splitter gets wrong.

    Tags normally sit directly after the period, so without explicit handling
    every sentence in a response would be glued into a single unit.
    """
    asm = SentenceAssembler()
    out = asm.push(f"Nelson rises.[cite:{FID}] Charing sits nearby.[cite:{OTHER}]")
    assert out == [
        f"Nelson rises.[cite:{FID}]",
        f"Charing sits nearby.[cite:{OTHER}]",
    ]


def test_assembler_tolerates_a_space_before_the_tag() -> None:
    asm = SentenceAssembler()
    assert asm.push(f"Ends here. [cite:{FID}] Next one. ") == [
        f"Ends here. [cite:{FID}]",
        "Next one.",
    ]


def test_assembler_does_not_split_abbreviations() -> None:
    asm = SentenceAssembler()
    assert asm.push("St. James's Church is nearby. ") == [
        "St. James's Church is nearby."
    ]


def test_assembler_does_not_split_decimals() -> None:
    asm = SentenceAssembler()
    assert asm.push("The column stands 21.6m high. ") == [
        "The column stands 21.6m high."
    ]


def test_assembler_does_not_split_initials() -> None:
    asm = SentenceAssembler()
    assert asm.push("J. Smith built it. ") == ["J. Smith built it."]


def test_assembler_reassembles_a_tag_split_across_chunks() -> None:
    asm = SentenceAssembler()
    assert asm.push("Rises.[cite:9c1f2ab") == []
    assert asm.push("73e0d5144] Next") == [f"Rises.[cite:{FID}]"]


def test_assembler_reassembles_a_sentence_split_across_chunks() -> None:
    asm = SentenceAssembler()
    assert asm.push("Hello wor") == []
    assert asm.push("ld. Bye. ") == ["Hello world.", "Bye."]


def test_assembler_flush_returns_the_unterminated_tail() -> None:
    asm = SentenceAssembler()
    assert asm.push("No terminator here") == []
    assert asm.flush() == ["No terminator here"]
    # flush is one-shot
    assert asm.flush() == []


def test_assembler_handles_ellipsis() -> None:
    asm = SentenceAssembler()
    assert asm.push("Wait\u2026 and listen. ") == ["Wait\u2026", "and listen."]


# ---------------------------------------------------------------------------
# Tag stripping
# ---------------------------------------------------------------------------


def test_strip_removes_tags_without_stranding_punctuation() -> None:
    assert strip_citation_tags(f"Rises over the square.[cite:{FID}]") == (
        "Rises over the square."
    )


def test_strip_keeps_interior_punctuation_intact() -> None:
    text = f"Built in 1682, it stands.[cite:{FID}]"
    assert strip_citation_tags(text) == "Built in 1682, it stands."


# ---------------------------------------------------------------------------
# STORY validation
# ---------------------------------------------------------------------------


def test_story_accepts_a_cited_sentence_and_resolves_sources() -> None:
    out = validate_story_sentence(
        f"The monument rises over the square.[cite:{FID}]",
        valid_fact_ids=SOURCES,
    )
    assert out.text == "The monument rises over the square."
    assert out.citations == (FID,)
    assert out.source_urls == ("https://en.wikipedia.org/wiki/Nelson%27s_Column",)


def test_story_rejects_a_sentence_with_no_citation() -> None:
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(
            "The square was laid out in the 1840s.", valid_fact_ids=SOURCES
        )
    assert exc.value.reason is RejectionReason.NO_CITATION


def test_story_rejects_a_fabricated_fact_id() -> None:
    """Syntactic validity is worthless on its own; membership is the real check."""
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(
            f"The square dates from 1840.[cite:{OTHER}]", valid_fact_ids=SOURCES
        )
    assert exc.value.reason is RejectionReason.UNKNOWN_FACT_ID


def test_story_accepts_an_iterable_of_ids() -> None:
    out = validate_story_sentence(f"Rises.[cite:{FID}]", valid_fact_ids=[FID])
    assert out.citations == (FID,)
    assert out.source_urls == ()


@pytest.mark.parametrize(
    ("sentence", "reason"),
    [
        (
            f"Turn left at the corner.[cite:{FID}]",
            RejectionReason.DIRECTION_LANGUAGE,
        ),
        (
            f"Head north past the square.[cite:{FID}]",
            RejectionReason.DIRECTION_LANGUAGE,
        ),
        (
            f"There is a fountain in front of you.[cite:{FID}]",
            RejectionReason.PHYSICAL_ASSERTION,
        ),
        (
            f"You can see the column from here.[cite:{FID}]",
            RejectionReason.PHYSICAL_ASSERTION,
        ),
        (
            f"Mind the traffic as you go.[cite:{FID}]",
            RejectionReason.TRAFFIC_LANGUAGE,
        ),
    ],
)
def test_story_safety_deny_lists(sentence: str, reason: RejectionReason) -> None:
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(sentence, valid_fact_ids=SOURCES)
    assert exc.value.reason is reason


def test_story_multiple_citations_are_preserved_in_order() -> None:
    out = validate_story_sentence(
        f"A rises here.[cite:{FID}] B sits near.[cite:{OTHER}]",
        valid_fact_ids={FID: "u1", OTHER: "u2"},
    )
    assert out.citations == (FID, OTHER)
    assert out.source_urls == ("u1", "u2")


# ---------------------------------------------------------------------------
# NOTICE validation
# ---------------------------------------------------------------------------


def test_notice_accepts_an_invitation() -> None:
    out = validate_notice_sentence(
        "If you're somewhere safe to pause, notice what your feet are resting on."
    )
    assert out.text.startswith("If you're somewhere safe")


@pytest.mark.parametrize(
    ("sentence", "reason"),
    [
        (
            f"Notice the park here.[cite:{FID}]",
            RejectionReason.UNEXPECTED_CITATION,
        ),
        ("The pavement is grey.", RejectionReason.NOT_INVITATIONAL),
        (
            "Notice how the rain feels on your skin.",
            RejectionReason.UNGROUNDED_WEATHER,
        ),
        ("Notice the bench to your left.", RejectionReason.PHYSICAL_ASSERTION),
        ("Notice the road as you cross the road.", RejectionReason.DIRECTION_LANGUAGE),
        (
            "Notice the church that was built in 1682.",
            RejectionReason.FACTUAL_CLAIM,
        ),
    ],
)
def test_notice_rules(sentence: str, reason: RejectionReason) -> None:
    with pytest.raises(SentenceRejected) as exc:
        validate_notice_sentence(sentence)
    assert exc.value.reason is reason


def test_notice_rejects_weather_even_with_a_valid_invitation() -> None:
    """N6 must fire before N3, or a weather guess rides in on 'notice'."""
    with pytest.raises(SentenceRejected) as exc:
        validate_notice_sentence("If you're somewhere safe, notice how sunny it is.")
    assert exc.value.reason is RejectionReason.UNGROUNDED_WEATHER


# ---------------------------------------------------------------------------
# STORY: the bounded framing exemption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "The evening light is soft here.",
        "The air is cool tonight, and the walk can be slow.",
        "Whatever you are carrying today, this city has carried it before.",
        "The light is gentle and the sky is quiet.",
    ],
)
def test_story_admits_framing_sentences_only_when_the_caller_allows(
    sentence: str,
) -> None:
    """Off by default -- the exemption is spent by the chunk, not assumed."""
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(sentence, valid_fact_ids=SOURCES)
    assert exc.value.reason is RejectionReason.NO_CITATION

    out = validate_story_sentence(
        sentence, valid_fact_ids=SOURCES, allow_uncited_framing=True
    )
    assert out.text == sentence
    assert out.citations == ()
    assert out.source_urls == ()


@pytest.mark.parametrize(
    "sentence",
    [
        "The fort was built here.",  # claim vocabulary
        "It has stood for 200 years.",  # a number
        "The light over Vijayanagar is soft tonight.",  # a proper noun
        "The pavement is grey.",  # not atmospheric at all
        "Water finds its way down the hill.",  # no framing vocabulary
    ],
)
def test_story_framing_exemption_does_not_admit_a_claim(sentence: str) -> None:
    """Uncited atmosphere is allowed; uncited *content* is not."""
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(
            sentence, valid_fact_ids=SOURCES, allow_uncited_framing=True
        )
    assert exc.value.reason is RejectionReason.NO_CITATION


def test_framing_exemption_still_enforces_the_safety_denylists() -> None:
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(
            "The evening light falls to your left.",
            valid_fact_ids=SOURCES,
            allow_uncited_framing=True,
        )
    assert exc.value.reason is RejectionReason.PHYSICAL_ASSERTION


def test_framing_exemption_leaves_cited_sentences_alone() -> None:
    """A cited sentence about a place is still exactly as valid as before."""
    out = validate_story_sentence(
        f"The evening light falls over Nelson's Column.[cite:{FID}]",
        valid_fact_ids=SOURCES,
        allow_uncited_framing=True,
    )
    assert out.citations == (FID,)
    assert out.source_urls == ("https://en.wikipedia.org/wiki/Nelson%27s_Column",)


# ---------------------------------------------------------------------------
# STORY: the bounded news exemption
# ---------------------------------------------------------------------------

#: A line the narrator is told to write: one headline, named once, framed as a
#: passing thought. Note the proper nouns -- the framing heuristic can never
#: admit this, which is the whole reason the marker exists.
NEWS_LINE = (
    "With the recent buzz about the tree-poisoning row in Vijayanagar, "
    "the streets here feel restless."
)


def test_a_marked_news_sentence_needs_the_callers_allowance() -> None:
    """Off by default, like the framing allowance -- and it must be marked."""
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(NEWS_LINE + "[news]", valid_fact_ids=SOURCES)
    assert exc.value.reason is RejectionReason.NEWS_OVERFLOW

    # The framing allowance alone does not admit it: an unmarked version of the
    # same sentence is still uncited content, not atmosphere.
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(
            NEWS_LINE + "[news]",
            valid_fact_ids=SOURCES,
            allow_uncited_framing=True,
        )
    assert exc.value.reason is RejectionReason.NEWS_OVERFLOW

    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(NEWS_LINE, valid_fact_ids=SOURCES)
    assert exc.value.reason is RejectionReason.NO_CITATION


def test_an_allowed_news_sentence_speaks_without_its_marker() -> None:
    out = validate_story_sentence(
        NEWS_LINE + "[news]",
        valid_fact_ids=SOURCES,
        allow_uncited_framing=True,
        allow_news_reference=True,
    )

    assert out.text == NEWS_LINE
    assert "[news]" not in out.text
    assert out.citations == ()
    assert out.source_urls == ()
    # The flag is how the orchestrator spends the news allowance rather than the
    # framing one against this sentence.
    assert out.news is True


def test_a_news_sentence_still_spends_the_chunk_s_uncited_allowance() -> None:
    """News may not become a third uncited sentence of a chunk."""
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(
            NEWS_LINE + "[news]",
            valid_fact_ids=SOURCES,
            allow_uncited_framing=False,
            allow_news_reference=True,
        )
    assert exc.value.reason is RejectionReason.NO_CITATION


def test_the_news_exemption_still_enforces_the_safety_denylists() -> None:
    with pytest.raises(SentenceRejected) as exc:
        validate_story_sentence(
            "With all this news, turn left at the corner.[news]",
            valid_fact_ids=SOURCES,
            allow_uncited_framing=True,
            allow_news_reference=True,
        )
    assert exc.value.reason is RejectionReason.DIRECTION_LANGUAGE


def test_a_cited_sentence_that_also_carries_the_marker_is_still_cited() -> None:
    """The citation is the stronger ground, and the marker never reaches TTS."""
    out = validate_story_sentence(
        f"Nelson's Column rises over the square.[cite:{FID}][news]",
        valid_fact_ids=SOURCES,
        allow_uncited_framing=True,
        allow_news_reference=False,
    )

    assert out.citations == (FID,)
    assert out.news is False
    assert "[news]" not in out.text


def test_strip_citation_tags_removes_the_news_marker() -> None:
    assert strip_citation_tags("The buzz is real. [news]") == "The buzz is real."
    assert strip_citation_tags(f"A rises .[cite:{FID}]") == "A rises."


def test_notice_rejects_a_news_marker() -> None:
    """A mode with nothing to mark must not speak the machinery."""
    with pytest.raises(SentenceRejected) as exc:
        validate_notice_sentence(
            "If you're somewhere safe to pause, notice the news round here.[news]"
        )
    assert exc.value.reason is RejectionReason.UNEXPECTED_CITATION


def test_the_assembler_splits_after_a_news_marker() -> None:
    """The marker sits where a citation does, and must not glue sentences."""
    assembler = SentenceAssembler()
    sentences = assembler.push(
        "The buzz is real.[news] The evening light is soft. And the air is cool."
    )
    assert sentences == [
        "The buzz is real.[news]",
        "The evening light is soft.",
        "And the air is cool.",
    ]
    assert assembler.flush() == []


# ---------------------------------------------------------------------------
# Static fallback copy
# ---------------------------------------------------------------------------


def test_static_notice_lines_pass_notice_validation() -> None:
    """The last-resort copy must satisfy the rules it bypasses at runtime."""
    from app.services.narration import STATIC_NOTICE_LINES

    assert STATIC_NOTICE_LINES
    for line in STATIC_NOTICE_LINES:
        out = validate_notice_sentence(line)
        assert out.text == line


# ---------------------------------------------------------------------------
# Contract with the request and with /api/context
# ---------------------------------------------------------------------------


def test_shipped_regex_is_byte_identical_to_the_specified_pattern() -> None:
    assert CITATION_PATTERN.pattern == r"\[cite:\s*([a-zA-Z0-9_-]+)\]"


def test_real_fact_ids_from_context_round_trip_through_the_parser() -> None:
    """The seam between /api/context (Part 2) and this validator (Part 3).

    ``make_fact_id`` produces 16 hex characters, which must satisfy the
    citation pattern without the pattern having to encode that shape.
    """
    from app.services.geo import make_fact_id

    for title, lat, lon in [
        ("Nelson's Column", 51.507, -0.128),
        ("Hyde Park", 51.507, -0.165),
    ]:
        fact_id = make_fact_id(title, lat, lon)
        assert extract_citation_ids(format_citation_tag(fact_id)) == [fact_id]
        validated = validate_story_sentence(
            f"Something nearby.[cite:{fact_id}]", valid_fact_ids={fact_id: "u"}
        )
        assert validated.citations == (fact_id,)
