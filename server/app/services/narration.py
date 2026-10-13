"""Narration orchestration: prompt -> stream -> validate -> speak.

Implements sections 5-7.8 of docs/llm-orchestration.md.

The load-bearing guarantee: **no unvalidated sentence ever reaches TTS.** If the
model produces nothing usable -- whether because every sentence failed citation
validation or because the provider threw -- this module degrades to NOTICE mode,
and if that fails too it returns pre-approved static copy. It never raises to
the caller and never speaks raw model output, with a single deliberate
exception: :class:`LLMTemporaryOutage`, which means the whole provider cascade
is down rather than one sentence being bad, and must reach the API so it can
answer with the structured retryable error instead of pretending static copy is
the walk.

Two deliberate exemptions sit alongside that guarantee, and both are bounded:
sentences injected in code (the greeting and the hand-off, which carry no
citation the validator could check), and up to ``MAX_UNCITED_FRAMING_SENTENCES``
uncited sentences per chunk -- admitted either as framing, which the validator
allows only when the sentence cannot be read as a claim about anything in
particular (``validation.rules._is_framing_sentence``), or as a marked
reference to the RECENT LOCAL NEWS block, of which the chunk may have at most
``MAX_NEWS_REFERENCES`` and which spends one of those framing slots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from app.services.prompts import (
    MAX_NEWS_REFERENCES,
    MAX_UNCITED_FRAMING_SENTENCES,
    NOTICE_SYSTEM,
    PEDESTRIAN_REMINDER,
    STORY_SYSTEM,
    NoticeContext,
    build_cycle_user_prompt,
    build_notice_user_prompt,
    build_story_user_prompt,
)
from app.services.providers import LLMProvider, LLMTemporaryOutage
from app.validation import (
    RejectionReason,
    SentenceAssembler,
    SentenceRejected,
    ValidatedSentence,
    validate_notice_sentence,
    validate_story_sentence,
)

logger = logging.getLogger("echoes.narration")

#: Last-resort copy. Fixed constants, so they cannot hallucinate; they are
#: validated against the NOTICE rules in the test suite.
STATIC_NOTICE_LINES: tuple[str, ...] = (
    "If you're somewhere safe to pause, notice what your feet are resting on.",
    "When you're ready, listen for the nearest sound and let it come to you.",
)

#: How many sentences one STORY chunk may speak. Sized for the 4-6 the prompt
#: asks for, with headroom: the cap exists to bound what a runaway model can
#: cost, not to trim a well-behaved one.
MAX_STORY_SENTENCES = 6
MAX_NOTICE_SENTENCES = 2

#: How many of those sentences may go uncited: the atmospheric opener and the
#: line that answers the walker's mood. ``MAX_UNCITED_FRAMING_SENTENCES`` is
#: imported from ``prompts`` rather than defined here, because the model is told
#: the same number and two definitions of it would drift. What may spend the
#: allowance is decided in ``validation.rules``.
#:
#: ``MAX_NEWS_REFERENCES`` is the same arrangement for the news marker, with one
#: extra rule the validator enforces: a marked sentence spends a framing slot as
#: well, so the news cannot lift the chunk's total uncited count. Re-exported
#: under this module's name so callers read it beside the other sentence caps.
MAX_NEWS_SENTENCES = MAX_NEWS_REFERENCES


@dataclass
class NarrationResult:
    """What the client should say, and how it got there."""

    mode: str
    sentences: list[ValidatedSentence]
    degraded: bool = False
    used_static_fallback: bool = False
    rejections: list[RejectionReason] = field(default_factory=list)
    error: str | None = None

    @property
    def spoken_text(self) -> str:
        """Speakable text: validated sentences, then the mandatory reminder."""
        body = " ".join(s.text for s in self.sentences if s.text)
        return f"{body} {PEDESTRIAN_REMINDER}".strip()

    @property
    def source_urls(self) -> tuple[str, ...]:
        """Deduplicated source links for the client panel, in citation order."""
        seen: dict[str, None] = {}
        for sentence in self.sentences:
            for url in sentence.source_urls:
                seen.setdefault(url, None)
        return tuple(seen)


@dataclass
class _StreamOutcome:
    kept: list[ValidatedSentence]
    rejections: list[RejectionReason]
    error: str | None


async def _stream_and_validate(
    provider: LLMProvider,
    *,
    system: str,
    user: str,
    validate: Callable[[str], ValidatedSentence],
    sentence_limit: int,
) -> _StreamOutcome:
    """Stream a completion, keeping only sentences that pass ``validate``.

    A stream failure is deliberately not propagated: sentences validated before
    the failure are still good, and the caller's job is to degrade, not to
    retry inline (docs/llm-orchestration.md 7.8). The failure is captured in
    ``error`` and logged so it stays observable.

    The one exception is :class:`LLMTemporaryOutage`: that is not a stream that
    failed, it is the whole cascade failing inside its retry budget, and
    degrading to static copy would hide a systemic outage from the walker. It is
    re-raised so the API can answer with the structured retryable error
    (``LLM_TEMPORARY_OUTAGE``).

    A sentence the chunk already contains is dropped as a duplicate. Repeating a
    sentence is the failure mode a walker notices immediately and the one the
    cap cannot catch: it costs nothing to say, and it arrives as a full-length
    chunk of the same words.
    """
    assembler = SentenceAssembler()
    kept: list[ValidatedSentence] = []
    seen: set[str] = set()
    rejections: list[RejectionReason] = []
    error: str | None = None

    def handle(raw: str) -> None:
        if len(kept) >= sentence_limit:
            # R5 / N1: cap how much can be spoken, regardless of what arrives.
            rejections.append(RejectionReason.OVERFLOW)
            return
        try:
            sentence = validate(raw)
        except SentenceRejected as rejected:
            rejections.append(rejected.reason)
            logger.info("dropped sentence (%s): %.120s", rejected.reason, raw)
            return

        fingerprint = sentence.text.casefold()
        if fingerprint in seen:
            rejections.append(RejectionReason.DUPLICATE)
            logger.info("dropped repeated sentence: %.120s", raw)
            return

        seen.add(fingerprint)
        kept.append(sentence)

    try:
        async for chunk in provider.stream(system=system, user=user):
            for sentence in assembler.push(chunk):
                handle(sentence)
        for sentence in assembler.flush():
            handle(sentence)
    except LLMTemporaryOutage:
        # Every provider is down: this is an outage, not a bad sentence. Let it
        # reach the API layer, which knows how to say so to the client.
        raise
    except Exception as exc:  # noqa: BLE001 - designed degradation, see docstring
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("narration stream failed: %s", error)

    return _StreamOutcome(kept=kept, rejections=rejections, error=error)


async def generate_narration(
    *,
    mode: str,
    provider: LLMProvider,
    facts: Sequence[Any] = (),
    notice: NoticeContext | None = None,
    greeting: str | None = None,
    handoff: str | None = None,
    mood: str | None = None,
    weather: Any = None,
    news: Sequence[Any] = (),
    cycle_kind: str = "full",
    cycle_words: int | None = None,
    already_said: str = "",
    max_story_sentences: int = MAX_STORY_SENTENCES,
    max_notice_sentences: int = MAX_NOTICE_SENTENCES,
) -> NarrationResult:
    """Produce validated narration for one walk step.

    Never raises, except for :class:`LLMTemporaryOutage`: a stream that fails
    degrades, but a cascade that has exhausted every provider is a systemic
    outage the caller is told about. Degradation to NOTICE happens at most once,
    so a prompt or model change that makes the validator reject everything
    cannot become an unbounded loop.

    ``greeting`` and ``handoff`` are injected in code, after validation: both
    are citation-free, so a model that produced them would have them deleted
    (see ``prompts.STORY_SYSTEM``). ``mood``, ``weather`` and ``news`` are
    per-request context folded into the STORY prompt; ``mood`` is the one the
    brief calls out, so it is passed through in the model's own words rather
    than mapped away, and ``news`` reaches the model as headlines it may allude
    to once -- never as anything it may state.
    """
    rejections: list[RejectionReason] = []
    degraded = False
    last_error: str | None = None

    if cycle_kind == "atmosphere" or (cycle_kind == "local" and not facts):
        news_left_atm = MAX_NEWS_REFERENCES if news else 0
        # Weather/mood/time: no FactObject, so STORY would otherwise degrade to
        # NOTICE. The atmosphere flag lets uncited clock/temp sentences through
        # while still stripping historical claims (validation.rules).
        def validate_atmosphere(raw: str) -> ValidatedSentence:
            nonlocal news_left_atm
            sentence = validate_story_sentence(
                raw,
                valid_fact_ids={},
                allow_uncited_framing=True,
                allow_news_reference=news_left_atm > 0,
                allow_atmosphere=True,
            )
            if sentence.news:
                news_left_atm -= 1
            return sentence

        outcome = await _stream_and_validate(
            provider,
            system=STORY_SYSTEM,
            user=build_cycle_user_prompt(
                "atmosphere",
                mood=mood,
                weather=weather,
                news=news,
                words=cycle_words,
                sentences=max_story_sentences if cycle_words else None,
                already_said=already_said,
            ),
            validate=validate_atmosphere,
            sentence_limit=max_story_sentences,
        )
        rejections.extend(outcome.rejections)
        last_error = outcome.error
        if outcome.kept:
            return _decorate(
                NarrationResult(
                    mode="STORY",
                    sentences=outcome.kept,
                    rejections=rejections,
                    error=outcome.error,
                ),
                greeting=greeting,
                handoff=handoff,
            )
        logger.info(
            "atmosphere cycle produced no valid sentences (%s); degrading to NOTICE",
            [r.value for r in rejections],
        )
        degraded = True
    elif mode == "STORY" and facts:
        fact_sources = {
            fact.fact_id: getattr(fact, "source_url", "") for fact in facts
        }
        # The uncited allowance belongs to the chunk, not the sentence: it is
        # spent in the order the model uses it and never renewed, so a model
        # that answers with nothing but atmosphere is rejected from the third
        # uncited sentence onwards. The state lives here rather than in the
        # validator for the same reason -- the validator sees one sentence.
        framing_left = MAX_UNCITED_FRAMING_SENTENCES
        # The news allowance is tracked separately as well as spending a framing
        # slot, so a second marked sentence is refused for what it is (a
        # bulletin, ``NEWS_OVERFLOW``) rather than for what it looked like.
        news_left = MAX_NEWS_REFERENCES

        def validate_framing_aware(raw: str) -> ValidatedSentence:
            nonlocal framing_left, news_left
            sentence = validate_story_sentence(
                raw,
                valid_fact_ids=fact_sources,
                allow_uncited_framing=framing_left > 0,
                allow_news_reference=news_left > 0,
            )
            if sentence.news:
                news_left -= 1
                framing_left -= 1
                logger.info("speaking a news sentence uncited: %.120s", raw)
            elif not sentence.citations:
                framing_left -= 1
                logger.info("speaking a framing sentence uncited: %.120s", raw)
            return sentence

        user_prompt = (
            build_cycle_user_prompt(
                cycle_kind,
                facts=facts,
                mood=mood,
                weather=weather,
                news=news,
                words=cycle_words,
                sentences=max_story_sentences if cycle_words else None,
                already_said=already_said,
            )
            if cycle_kind in {"wiki", "local"}
            else build_story_user_prompt(
                facts, mood=mood, weather=weather, news=news
            )
        )
        outcome = await _stream_and_validate(
            provider,
            system=STORY_SYSTEM,
            user=user_prompt,
            validate=validate_framing_aware,
            sentence_limit=max_story_sentences,
        )
        rejections.extend(outcome.rejections)
        last_error = outcome.error

        if outcome.kept:
            return _decorate(
                NarrationResult(
                    mode="STORY",
                    sentences=outcome.kept,
                    rejections=rejections,
                    error=outcome.error,
                ),
                greeting=greeting,
                handoff=handoff,
            )

        logger.info(
            "STORY produced no valid sentences (%s); degrading to NOTICE",
            [r.value for r in rejections],
        )
        degraded = True
    elif mode != "NOTICE":
        # STORY requested with no facts is not a valid story state.
        degraded = True

    # Single NOTICE attempt -- the brake on degradation.
    outcome = await _stream_and_validate(
        provider,
        system=NOTICE_SYSTEM,
        user=build_notice_user_prompt(notice),
        validate=validate_notice_sentence,
        sentence_limit=max_notice_sentences,
    )
    rejections.extend(outcome.rejections)
    last_error = outcome.error or last_error

    if outcome.kept:
        return _decorate(
            NarrationResult(
                mode="NOTICE",
                sentences=outcome.kept,
                degraded=degraded,
                rejections=rejections,
                error=last_error,
            ),
            greeting=greeting,
        )

    logger.warning("NOTICE also produced nothing; serving static copy")
    return _decorate(
        NarrationResult(
            mode="NOTICE",
            sentences=[ValidatedSentence(text=line) for line in STATIC_NOTICE_LINES],
            degraded=True,
            used_static_fallback=True,
            rejections=rejections,
            error=last_error,
        ),
        greeting=greeting,
    )


def _decorate(
    result: NarrationResult,
    *,
    greeting: str | None = None,
    handoff: str | None = None,
) -> NarrationResult:
    """Prepend the greeting and append the hand-off, outside the validator.

    Both are code constants with no citation, so a model that produced them
    would have them deleted before TTS (see ``prompts.STORY_SYSTEM``). Injecting
    them here -- after every sentence has cleared validation -- is the only place
    they survive to be spoken.
    """
    if not greeting and not handoff:
        return result
    sentences: list[ValidatedSentence] = []
    if greeting:
        sentences.append(ValidatedSentence(text=greeting))
    sentences.extend(result.sentences)
    if handoff:
        sentences.append(ValidatedSentence(text=handoff))
    return replace(result, sentences=sentences)


__all__ = [
    "MAX_NEWS_SENTENCES",
    "MAX_NOTICE_SENTENCES",
    "MAX_STORY_SENTENCES",
    "MAX_UNCITED_FRAMING_SENTENCES",
    "NarrationResult",
    "STATIC_NOTICE_LINES",
    "generate_narration",
]
