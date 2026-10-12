"""Citation tags, text normalisation, and the validated-sentence contract.

Implements sections 7.2-7.3 and 7.7 of docs/llm-orchestration.md.

This module owns the citation grammar. It is deliberately the *only* place the
pattern is written down: ``services.prompts`` renders tags through
:func:`format_citation_tag`, so the syntax the model is told to emit is, by
construction, the syntax the rules read. A drift between the two would
silently reject every sentence, which would look like a model failure rather
than a configuration one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum, auto

# ---------------------------------------------------------------------------
# Citation tag -- SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------

#: The one and only citation-tag pattern.
CITATION_PATTERN = re.compile(r"\[cite:\s*([a-zA-Z0-9_-]+)\]")

#: The marker a sentence carries when it rests on the RECENT LOCAL NEWS block
#: instead of on a fact. It is a *marker*, not a citation, and deliberately
#: carries no id: news is never citable (``app/services/news.py``), and an id in
#: it would invite the model to invent one. It exists so that a news sentence
#: can be admitted by an explicit, bounded rule rather than by loosening the
#: framing heuristic -- the headline's own place name and figures would fail
#: that heuristic every time (``rules._is_framing_sentence``).
NEWS_TAG_PATTERN = re.compile(r"\[news\]")


#: Every marker the walk writes and the validator reads. The splitter walks this
#: too, or a marker sitting directly after a period would glue two sentences
#: into one (``assembler._sentence_boundary_end``).
TAG_PATTERNS = (CITATION_PATTERN, NEWS_TAG_PATTERN)


def format_citation_tag(fact_id: str) -> str:
    """Render a tag exactly as :data:`CITATION_PATTERN` expects to read it."""
    return f"[cite:{fact_id}]"


def format_news_tag() -> str:
    """Render the news marker, for the same reason the citation is rendered here.

    The prompt shows the model exactly the spelling :data:`NEWS_TAG_PATTERN`
    matches, so the two cannot drift.
    """
    return "[news]"


def has_news_tag(sentence: str) -> bool:
    """Whether ``sentence`` claims the news block as its ground."""
    return NEWS_TAG_PATTERN.search(sentence) is not None


def match_tag(text: str, position: int) -> int | None:
    """End index of a marker starting at ``position``, or None if none does."""
    for pattern in TAG_PATTERNS:
        match = pattern.match(text, position)
        if match:
            return match.end()
    return None


def extract_citation_ids(sentence: str) -> list[str]:
    """All fact_ids cited by ``sentence``, in order of appearance."""
    return CITATION_PATTERN.findall(sentence)


def strip_citation_tags(sentence: str) -> str:
    """Remove tags and tidy the whitespace they leave behind.

    Citation tags and the news marker are both removed: a marker is metadata
    for the validator, and a walker who heard one spoken would hear the
    machinery (``services.narration`` speaks only stripped text).

    A tag sits between the period and the sentence boundary, so naive removal
    yields ``"...rises ."``. Punctuation is re-attached before collapsing.
    """
    without = sentence
    for pattern in TAG_PATTERNS:
        without = pattern.sub("", without)
    # A removed tag can strand a space before terminal punctuation.
    without = re.sub(r"\s+([,.;:!?])", r"\1", without)
    return re.sub(r"\s{2,}", " ", without).strip()


# ---------------------------------------------------------------------------
# Rejection vocabulary
# ---------------------------------------------------------------------------


class RejectionReason(StrEnum):
    """Why a sentence was dropped. Logged, never spoken."""

    NO_CITATION = auto()
    UNKNOWN_FACT_ID = auto()
    DIRECTION_LANGUAGE = auto()
    PHYSICAL_ASSERTION = auto()
    TRAFFIC_LANGUAGE = auto()
    OVERFLOW = auto()
    UNEXPECTED_CITATION = auto()
    #: A news marker on a sentence the chunk has no room for: one allusive line
    #: per chunk, and the allowance is spent by the chunk, not the sentence.
    NEWS_OVERFLOW = auto()
    NOT_INVITATIONAL = auto()
    UNGROUNDED_WEATHER = auto()
    FACTUAL_CLAIM = auto()
    #: The same sentence, again. A model loop rather than narration, and the one
    #: rejection that is about the chunk as a whole rather than the sentence.
    DUPLICATE = auto()


class SentenceRejected(Exception):
    """Raised by a validator when a sentence must not be spoken."""

    def __init__(self, reason: RejectionReason, sentence: str) -> None:
        super().__init__(f"{reason.value}: {sentence!r}")
        self.reason = reason
        self.sentence = sentence


@dataclass(frozen=True)
class ValidatedSentence:
    """A sentence cleared for TTS, with its provenance preserved."""

    text: str  # tag-free, ready to speak
    citations: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    #: True when the sentence was marked as resting on the news block rather
    #: than on a fact. It has no citations by construction, so the caller needs
    #: this flag to spend the news allowance against the right sentence.
    news: bool = False


# ---------------------------------------------------------------------------
# Normalisation (7.2)
# ---------------------------------------------------------------------------

_FULLWIDTH_BRACKETS = str.maketrans({"［": "[", "］": "]", "：": ":"})


def normalize(text: str) -> str:
    """Canonicalise so formatting cannot smuggle text past a rule."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_FULLWIDTH_BRACKETS)
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return re.sub(r"[ \t]+", " ", text).strip()


__all__ = [
    "CITATION_PATTERN",
    "NEWS_TAG_PATTERN",
    "TAG_PATTERNS",
    "RejectionReason",
    "SentenceRejected",
    "ValidatedSentence",
    "extract_citation_ids",
    "format_citation_tag",
    "format_news_tag",
    "has_news_tag",
    "match_tag",
    "normalize",
    "strip_citation_tags",
]
