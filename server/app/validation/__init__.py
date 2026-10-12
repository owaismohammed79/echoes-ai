"""Sentence assembly and citation validation -- the project's factual moat.

This package is the single entry point for "may this sentence be spoken?".
Importing from here rather than from the individual modules means callers never
need to know how the grammar, the splitter and the rule sets are divided up::

    from app.validation import SentenceAssembler, validate_story_sentence

Implements docs/llm-orchestration.md sections 7.1-7.7.
"""

from __future__ import annotations

from app.validation.assembler import SentenceAssembler
from app.validation.citations import (
    CITATION_PATTERN,
    NEWS_TAG_PATTERN,
    RejectionReason,
    SentenceRejected,
    ValidatedSentence,
    extract_citation_ids,
    format_citation_tag,
    format_news_tag,
    has_news_tag,
    normalize,
    strip_citation_tags,
)
from app.validation.rules import (
    DIRECTION_LANGUAGE,
    FACTUAL_CLAIM,
    INVITATION_MARKERS,
    PHYSICAL_ASSERTION,
    TRAFFIC_LANGUAGE,
    WEATHER_PREDICATES,
    validate_notice_sentence,
    validate_qa_sentence,
    validate_story_sentence,
)

__all__ = [
    "CITATION_PATTERN",
    "DIRECTION_LANGUAGE",
    "FACTUAL_CLAIM",
    "INVITATION_MARKERS",
    "NEWS_TAG_PATTERN",
    "PHYSICAL_ASSERTION",
    "RejectionReason",
    "SentenceAssembler",
    "SentenceRejected",
    "TRAFFIC_LANGUAGE",
    "ValidatedSentence",
    "WEATHER_PREDICATES",
    "extract_citation_ids",
    "format_citation_tag",
    "format_news_tag",
    "has_news_tag",
    "normalize",
    "strip_citation_tags",
    "validate_notice_sentence",
    "validate_qa_sentence",
    "validate_story_sentence",
]
