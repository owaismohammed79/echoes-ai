"""Push-To-Talk question answering.

Implements sections 10.2-10.4 of docs/audio-streaming.md.

Transcription lives in ``services.stt`` and synthesis in ``services.tts``; this
module owns the middle: turning a question plus the walk's cached facts into
**one grounded, validated sentence**, or an honest refusal.

The refusal path is the interesting part. A model told to always cite cannot
answer an out-of-scope question, and the validator will correctly delete
whatever it invents -- leaving silence. So an ungrounded answer is replaced by a
*constant*, never by model output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.services.prompts import (
    CANNED_REFUSAL,
    QA_INSUFFICIENT,
    QA_SYSTEM,
    build_qa_user_prompt,
)
from app.services.providers import LLMProvider
from app.validation import (
    RejectionReason,
    SentenceAssembler,
    SentenceRejected,
    ValidatedSentence,
    validate_qa_sentence,
)

logger = logging.getLogger("echoes.qa")

#: Cap on past turns handed back to the model. Unbounded history is a latency
#: and cost leak, and a walk's useful follow-up window is short.
MAX_HISTORY_TURNS = 4


@dataclass
class QAResult:
    """The outcome of one Push-To-Talk exchange."""

    transcript: str
    answer: ValidatedSentence | None
    refused: bool = False
    reason: str | None = None
    rejections: list[RejectionReason] = field(default_factory=list)
    error: str | None = None

    @property
    def answer_text(self) -> str:
        """What should be spoken -- the validated answer, or the refusal."""
        if self.answer is not None and self.answer.text:
            return self.answer.text
        return CANNED_REFUSAL

    @property
    def citations(self) -> tuple[str, ...]:
        return self.answer.citations if self.answer else ()

    @property
    def source_urls(self) -> tuple[str, ...]:
        return self.answer.source_urls if self.answer else ()


def trim_history(
    history: Sequence[tuple[str, str]], *, max_turns: int = MAX_HISTORY_TURNS
) -> list[tuple[str, str]]:
    """Keep only the most recent ``max_turns`` (role, text) pairs."""
    if max_turns <= 0:
        return []
    return list(history)[-max_turns:]


def _refusal(
    transcript: str,
    *,
    reason: str,
    rejections: list[RejectionReason] | None = None,
    error: str | None = None,
) -> QAResult:
    return QAResult(
        transcript=transcript,
        answer=None,
        refused=True,
        reason=reason,
        rejections=rejections or [],
        error=error,
    )


async def answer_question(
    *,
    question: str,
    facts: Sequence[Any],
    provider: LLMProvider,
    history: Sequence[tuple[str, str]] = (),
    narration: str | None = None,
) -> QAResult:
    """Answer ``question`` from ``facts``, or refuse. Never raises.

    ``facts`` is the walk's cached context. An empty list means the walk is in
    Notice Mode, where no question can be grounded -- refusing immediately
    avoids paying for a model call that cannot produce a speakable sentence.

    ``narration`` is the validated narration the walk just spoke, so a
    follow-up that refers back to it ("that temple") has the landmarks in front
    of the model rather than an empty window.
    """
    question = (question or "").strip()
    if not question:
        return _refusal("", reason="empty_transcript")
    if not facts:
        logger.info("PTT with no cached facts; refusing without an LLM call")
        return _refusal(question, reason="no_context")

    id_to_url = {fact.fact_id: getattr(fact, "source_url", "") for fact in facts}

    raw_parts: list[str] = []
    error: str | None = None
    try:
        async for chunk in provider.stream(
            system=QA_SYSTEM,
            user=build_qa_user_prompt(
                facts=facts,
                question=question,
                history=trim_history(history),
                narration=narration,
            ),
        ):
            raw_parts.append(chunk)
    except Exception as exc:  # noqa: BLE001 - a failed answer must not 500 a walk
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("QA generation failed: %s", error)

    raw = "".join(raw_parts).strip()

    # Q5 -- the model's structured escape hatch. Case/whitespace tolerant,
    # because a sentinel that only matches exactly is a sentinel that fails.
    if QA_INSUFFICIENT in raw.upper():
        return _refusal(question, reason="insufficient_facts")

    if not raw:
        return _refusal(question, reason="empty_answer", error=error)

    # Q1 -- one sentence only.
    assembler = SentenceAssembler()
    sentences = assembler.push(raw) + assembler.flush()
    if not sentences:
        return _refusal(question, reason="empty_answer", error=error)

    try:
        validated = validate_qa_sentence(sentences[0], valid_fact_ids=id_to_url)
    except SentenceRejected as rejected:
        logger.info("QA answer rejected (%s): %.120s", rejected.reason, sentences[0])
        return _refusal(
            question,
            reason="ungrounded_answer",
            rejections=[rejected.reason],
            error=error,
        )

    return QAResult(transcript=question, answer=validated, error=error)


__all__ = ["MAX_HISTORY_TURNS", "QAResult", "answer_question", "trim_history"]
