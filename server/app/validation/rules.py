"""Mode-specific validation rules.

Implements sections 7.4-7.6 of docs/llm-orchestration.md and Q1-Q4 of
docs/audio-streaming.md section 10.4.

This is where the project's central promise is kept: AGENTS.md section 1 forbids
relying on prompt instructions to prevent hallucination, so every sentence that
reaches TTS has passed through one of the functions below.

The three rule sets share their safety deny-lists but differ in what counts as
success:

* **STORY** presumes every declarative sentence is a claim, so an uncited
  sentence is dropped -- with two bounded exemptions: a sentence that reads as
  atmosphere, mood or transition and cannot be read as a claim about any
  particular place may be spoken uncited, at most
  ``narration.MAX_UNCITED_FRAMING_SENTENCES`` times per chunk (see
  :func:`_is_framing_sentence`); and a sentence carrying the *news marker*
  (``[news]``) may allude to one headline from the RECENT LOCAL NEWS block, at
  most ``prompts.MAX_NEWS_REFERENCES`` times per chunk. The second exemption is
  explicit rather than heuristic: a headline's own place name and figures would
  fail the framing test every time, and loosening that test to admit them would
  admit any uncited claim.
* **NOTICE** inverts: with nothing to cite, a citation tag is evidence the
  model went looking for something to claim.
* **QA** keeps STORY's grounding contract, but the *consequence* of failure is
  a canned refusal rather than Notice Mode.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

from app.validation.citations import (
    RejectionReason,
    SentenceRejected,
    ValidatedSentence,
    extract_citation_ids,
    has_news_tag,
    normalize,
    strip_citation_tags,
)

# ---------------------------------------------------------------------------
# Deny-lists (7.6)
# ---------------------------------------------------------------------------

DIRECTION_LANGUAGE = re.compile(
    r"""
    \b(?:
      turn\s+(?:left|right)|cross\s+the\s+(?:road|street|main\s+road)|
      head\s+(?:north|south|east|west)|continue\s+along|
      go\s+straight|follow\s+the\s+(?:road|path|signs)|walk\s+towards?
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

PHYSICAL_ASSERTION = re.compile(
    r"""
    \b(?:
      in\s+front\s+of\s+you|ahead\s+of\s+you|behind\s+you|
      (?:to|on)\s+your\s+(?:left|right)|
      you\s+can\s+(?:see|hear)\b|you(?:'ll|\s+will)\s+see|
      there\s+is\s+a\s+\w+\s+(?:to\s+your|beside|next\s+to)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

TRAFFIC_LANGUAGE = re.compile(
    r"""\b(?:traffic|crossing|roadworks|hazard|lane\s+closure|junction\s+ahead)\b""",
    re.IGNORECASE | re.VERBOSE,
)

INVITATION_MARKERS = re.compile(
    r"""
    \b(?:
      notice|noticing|listen|feel|observe|sense|
      if\s+you'?re\s+somewhere\s+safe|you\s+might|you\s+could|
      try\s+to|let\s+your|take\s+a\s+moment|when\s+you'?re\s+ready
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Weather is time-only by decision (docs/llm-orchestration.md 6), so any
#: mention of conditions is a fabrication by construction. If a weather
#: provider is ever added without a new rule, this catches the resulting guess
#: in code rather than in production.
WEATHER_PREDICATES = re.compile(
    r"""
    \b(?:
      rain|rains|raining|rainy|drizzle|drizzling|snow|snowing|sunny|sunshine|
      wind|windy|cloud|clouds|cloudy|storm|stormy|humid|forecast|temperature|
      degrees|fog|foggy|mist|overcast
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

FACTUAL_CLAIM = re.compile(
    r"""
    (?:
      \b(?:was|were)\s+(?:built|constructed|founded|established|designed)\b|
      \bdates?\s+from\b|\bhistor(?:y|ic|ical)\b|\bcentury\b|
      \b(?:1[0-9]{3}|20[0-9]{2})\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Vocabulary that marks a sentence as *framing* rather than claim: atmosphere,
#: the hour, the weather (which reaches STORY from a named provider, so naming it
#: is grounded there), how the walker feels, and the transitions between facts.
FRAMING_MARKERS = re.compile(
    r"""
    \b(?:
      today|tonight|this\s+(?:morning|afternoon|evening|hour)|
      morning|afternoon|evening|dusk|sunset|night|
      light|sky|air|breeze|wind|shade|warmth|warm|cool|cooler|quiet|gentle|
      softly|slowly|soft|
      rain|rainy|drizzle|cloud|clouds|cloudy|overcast|sunshine|sunny|humid|fog|
      breathe|breathing|breath|carry|carrying|shoulders|heart|
      hope|hopeful|resilience|resilient|endure|enduring|steady|steadily|
      courage|brave|bravely|kind|kindness|proud|proudly|
      meanwhile|perhaps|maybe|somewhere|for\s+a\s+moment
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Anything numeric: years, distances, counts. The shape of a specific claim.
_DIGIT = re.compile(r"\d")

#: Capitalised words a framing sentence may contain after its first word. First
#: person is not a proper noun, and spoken prose opens with it constantly.
_PROPER_NOUN_ALLOWED = frozenset(
    {"I", "I'm", "I've", "I'll", "I'd", "AM", "PM"}
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_sources(
    valid_fact_ids: Mapping[str, str] | Iterable[str],
) -> dict[str, str]:
    """Normalise the caller's fact-id input into ``{fact_id: source_url}``."""
    if isinstance(valid_fact_ids, Mapping):
        return dict(valid_fact_ids)
    return {fid: "" for fid in valid_fact_ids}


def _check_safety_denylists(text: str, sentence: str) -> None:
    """Raise if ``text`` violates a shared safety rule."""
    if DIRECTION_LANGUAGE.search(text):
        raise SentenceRejected(RejectionReason.DIRECTION_LANGUAGE, sentence)
    if PHYSICAL_ASSERTION.search(text):
        raise SentenceRejected(RejectionReason.PHYSICAL_ASSERTION, sentence)
    if TRAFFIC_LANGUAGE.search(text):
        raise SentenceRejected(RejectionReason.TRAFFIC_LANGUAGE, sentence)


def _has_proper_noun(text: str) -> bool:
    """True if any word after the first is capitalised ("Vijayanagar", "Kempe").

    A named place or person is what "a claim about *this* place" means in
    practice: it is the shape a hallucination takes, and it is the part of a
    sentence that cannot be waved through uncited.
    """
    for index, word in enumerate(re.findall(r"[A-Za-z][A-Za-z']*", text)):
        if index and word[0].isupper() and word not in _PROPER_NOUN_ALLOWED:
            return True
    return False


def _is_atmosphere_sentence(text: str) -> bool:
    """Whether an uncited sentence is weather, mood or time-of-day.

    Atmosphere cycles have no FactObject to cite, but they *are* grounded: the
    weather provider, the walker's clock, and the mood they chose. Digits are
    allowed so "6:42 PM" and "9°C" survive. Historical or geographical claims
    are still rejected -- those belong to a wiki/park cycle with a citation.
    """
    if FACTUAL_CLAIM.search(text):
        return False
    if _has_proper_noun(text):
        return False
    return bool(
        FRAMING_MARKERS.search(text)
        or WEATHER_PREDICATES.search(text)
        or INVITATION_MARKERS.search(text)
    )


def _is_framing_sentence(text: str) -> bool:
    """Whether an uncited STORY sentence is framing rather than a claim.

    R1 presumes a claim, which is the right default -- but applied absolutely it
    also deletes the two sentences that make a walk feel written for the walker:
    the atmospheric opener and the line that answers their mood. So a *bounded*
    exemption admits a sentence that can be argued not to claim anything in
    particular:

    * no digits -- years, distances, dates and counts are the shape of a
      specific claim;
    * no claim vocabulary (:data:`FACTUAL_CLAIM`);
    * no proper noun past the first word;
    * and it reads as framing: atmosphere, the hour, the weather, a feeling, or
      a transition between facts.

    This is a heuristic where the rest of this module is deterministic. That is
    the cost of admitting feeling at all, and it is contained: the orchestrator
    spends the exemption at most
    ``narration.MAX_UNCITED_FRAMING_SENTENCES`` times per chunk, so an
    unevidenced sentence is the exception the walk logs, never the default it
    narrates from.
    """
    if _DIGIT.search(text) or FACTUAL_CLAIM.search(text):
        return False
    if _has_proper_noun(text):
        return False
    return FRAMING_MARKERS.search(text) is not None


def _cited_result(
    text: str, cited: list[str], id_to_url: Mapping[str, str]
) -> ValidatedSentence:
    """Build the validated sentence, preserving citation order and sources."""
    ordered = tuple(dict.fromkeys(cited))
    return ValidatedSentence(
        text=strip_citation_tags(text),
        citations=ordered,
        source_urls=tuple(url for url in (id_to_url[cid] for cid in ordered) if url),
    )


# ---------------------------------------------------------------------------
# Mode-specific validators (7.4 / 7.5 / 10.4)
# ---------------------------------------------------------------------------


def validate_story_sentence(
    sentence: str,
    *,
    valid_fact_ids: Mapping[str, str] | Iterable[str],
    allow_uncited_framing: bool = False,
    allow_news_reference: bool = False,
    allow_atmosphere: bool = False,
) -> ValidatedSentence:
    """Validate one STORY sentence. Raises ``SentenceRejected`` if unspeakable.

    ``valid_fact_ids`` maps ``fact_id -> source_url`` for the facts that were
    actually placed in the prompt. Membership here is what makes a citation
    real: syntactic validity alone is worthless, since a model can emit a
    perfectly-formed tag wrapping an id it invented.

    ``allow_uncited_framing`` is the bounded R1 exemption for atmosphere and
    mood (see :func:`_is_framing_sentence`). It answers "may this sentence skip
    a citation?"; "how many may?" is the orchestrator's answer, and it spends
    the allowance per chunk. A sentence that fails either test is rejected as
    ``NO_CITATION``, which is what it is: uncited, and not the kind of sentence
    that may be.

    ``allow_news_reference`` is the exemption for a sentence the model marked
    with the news tag, and it is deliberately *narrower* than the framing one:
    it must be granted explicitly, and the sentence still has to be inside the
    chunk's uncited allowance, so an allusive news line can never be the third
    uncited sentence of a chunk. A marked sentence with no allowance left is
    rejected as ``NEWS_OVERFLOW`` -- the rejection that says the tag was
    understood and the budget was the problem.

    ``allow_atmosphere`` is the weather/mood/time cycle: uncited sentences that
    name the sky, the hour or the feeling are admitted even when they carry a
    clock time or a temperature. Wiki and SerpApi claims still need a citation;
    this flag does not waive :data:`FACTUAL_CLAIM`.
    """
    id_to_url = _resolve_sources(valid_fact_ids)

    text = normalize(sentence)
    cited = extract_citation_ids(text)

    # R1 -- every declarative sentence is presumed to be a claim, unless it is
    # one of the few framing sentences the walk is allowed to spend, or it is an
    # attested reference to the news block.
    if not cited:
        if has_news_tag(text):
            if not allow_news_reference:
                raise SentenceRejected(RejectionReason.NEWS_OVERFLOW, sentence)
            if not allow_uncited_framing:
                raise SentenceRejected(RejectionReason.NO_CITATION, sentence)
            spoken = strip_citation_tags(text)
            _check_safety_denylists(spoken, sentence)
            return ValidatedSentence(text=spoken, news=True)
        if allow_atmosphere and _is_atmosphere_sentence(text):
            _check_safety_denylists(text, sentence)
            return ValidatedSentence(text=text)
        if not (allow_uncited_framing and _is_framing_sentence(text)):
            raise SentenceRejected(RejectionReason.NO_CITATION, sentence)
        _check_safety_denylists(text, sentence)
        return ValidatedSentence(text=text)

    # R2 -- the id must exist in the context we supplied.
    if any(cid not in id_to_url for cid in cited):
        raise SentenceRejected(RejectionReason.UNKNOWN_FACT_ID, sentence)

    _check_safety_denylists(text, sentence)
    return _cited_result(text, cited, id_to_url)


def validate_notice_sentence(sentence: str) -> ValidatedSentence:
    """Validate one NOTICE sentence. Raises ``SentenceRejected`` if unspeakable.

    The rule set inverts: with no facts to cite, a citation tag is not evidence
    of good behaviour but evidence the model went looking for something to
    claim.
    """
    text = normalize(sentence)

    # N2 -- no citations in a mode that has nothing to cite. The news marker
    # counts here for the same reason: a mode with no material has nothing to
    # mark, and a walker must never hear the machinery (``[news]``) spoken.
    if extract_citation_ids(text) or has_news_tag(text):
        raise SentenceRejected(RejectionReason.UNEXPECTED_CITATION, sentence)

    _check_safety_denylists(text, sentence)

    # N6 -- weather is never supplied, so mentioning it is invention.
    if WEATHER_PREDICATES.search(text):
        raise SentenceRejected(RejectionReason.UNGROUNDED_WEATHER, sentence)

    # N7 -- no claims at all in this mode.
    if FACTUAL_CLAIM.search(text):
        raise SentenceRejected(RejectionReason.FACTUAL_CLAIM, sentence)

    # N3 -- must be phrased as an invitation.
    if not INVITATION_MARKERS.search(text):
        raise SentenceRejected(RejectionReason.NOT_INVITATIONAL, sentence)

    return ValidatedSentence(text=text)


def validate_qa_sentence(
    sentence: str,
    *,
    valid_fact_ids: Mapping[str, str] | Iterable[str],
) -> ValidatedSentence:
    """Validate one Push-To-Talk answer sentence (docs/audio-streaming.md Q1-Q4).

    Same grounding contract as Story Mode: an answer is a claim, so it needs an
    in-context citation. The difference is the *consequence* of failure --
    Q3 routes to a canned refusal rather than to Notice Mode.

    The caller handles Q1 (take only the first sentence) and Q5 (replace the
    ``INSUFFICIENT`` sentinel with the canned refusal).
    """
    id_to_url = _resolve_sources(valid_fact_ids)

    text = normalize(sentence)
    cited = extract_citation_ids(text)

    # Q2 / Q3 -- unwrapped by the caller as "refuse", but still a rejection here.
    if not cited:
        raise SentenceRejected(RejectionReason.NO_CITATION, sentence)
    if any(cid not in id_to_url for cid in cited):
        raise SentenceRejected(RejectionReason.UNKNOWN_FACT_ID, sentence)

    # Q4 -- safety deny-lists.
    _check_safety_denylists(text, sentence)
    return _cited_result(text, cited, id_to_url)


__all__ = [
    "DIRECTION_LANGUAGE",
    "FACTUAL_CLAIM",
    "FRAMING_MARKERS",
    "INVITATION_MARKERS",
    "PHYSICAL_ASSERTION",
    "TRAFFIC_LANGUAGE",
    "WEATHER_PREDICATES",
    "validate_notice_sentence",
    "validate_qa_sentence",
    "validate_story_sentence",
]
