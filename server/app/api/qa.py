"""``POST /api/qa`` -- Push-To-Talk question answering.

Implements section 10 of docs/audio-streaming.md.

Pipeline: mic clip -> Groq Whisper -> grounded one-sentence answer -> audio.

The response shape carries **both** an audio body (success) and a JSON body
(success-with-native-fallback or refusal), because the two failures are
different events. A refusal means the facts could not answer the question; a
fallback means ElevenLabs could not say an answer that *is* grounded. Both set
``use_native_tts`` so the client's action is the same either way: speak the text
it already holds.

Error handling is explicit about what is pre-stream and what is post-stream:
guards and transcription failures raise proper HTTP statuses, because nothing
has been promised to the client yet. A TTS failure after a grounded answer
exists cannot do that -- the answer is good and must be delivered -- so it
returns 200 with ``use_native_tts: true`` in the header and the text in the body.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.api.headers import fallback_headers, qa_metadata_headers
from app.api.limits import DemoAdmissionDep, demo_headers
from app.dependencies import ContainerDep
from app.services.budgets import Admission
from app.services.demo import DEMO_ANSWER
from app.validation import ValidatedSentence
from app.services.providers import NullLLMProvider
from app.services.qa import QAResult, answer_question
from app.services.sessions import context_is_stale
from app.services.stt import (
    MAX_AUDIO_BYTES,
    STTError,
    STTRateLimited,
    is_supported_content_type,
)

logger = logging.getLogger("echoes.api.qa")

router = APIRouter()


async def _read_capped(upload: UploadFile, *, limit: int) -> bytes:
    """Read an upload, refusing anything over ``limit`` without buffering it all."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(64 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"audio exceeds {limit} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _demo_answer(admission: Admission, started: float) -> JSONResponse:
    """The kill-switch, for push-to-talk.

    A recorded line in the transport shape the client already handles, plus the
    demo header. No Groq call, no model call, no ElevenLabs call -- push-to-talk
    is a paid path twice over, so unless the cap stops it here it stops nothing.

    The answer is an assertion about the *system*, never about the walker's
    surroundings: Notice Mode's rules do not lapse because the budget did.
    """
    # A real ValidatedSentence, not a bare string: it is what QAResult carries,
    # and it is what makes `answer_text`, `citations` and `source_urls` behave
    # exactly as they do on the live path.
    result = QAResult(
        transcript="",
        answer=ValidatedSentence(text=DEMO_ANSWER),
        refused=False,
        reason=None,
    )
    return JSONResponse(
        status_code=200,
        content={
            "transcript": "",
            "answer": result.answer_text,
            "refused": False,
            "refusal_reason": None,
            "use_native_tts": True,
            "demo": True,
            "reason": "demo_mode",
        },
        headers={
            **fallback_headers(use_native_tts=True, reason="demo_mode"),
            **demo_headers(True, reason=admission.reason),
            **qa_metadata_headers(
                result,
                timings={"qa_total": int((time.monotonic() - started) * 1000)},
            ),
        },
    )


@router.post("/api/qa", summary="Push-To-Talk question answering")
async def api_qa(
    container: ContainerDep,
    admission: DemoAdmissionDep,
    audio: UploadFile = File(...),
    session_id: str | None = Form(None),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
) -> Response:
    """Transcribe a clip, answer it from cached facts, stream back audio."""
    started = time.monotonic()

    # --- Guards. None of these may cost a paid API call. -------------------
    # Checked before the upload is even buffered: under the kill-switch there is
    # nothing useful to do with the clip, so reading it would be pure cost.
    if admission.demo:
        return _demo_answer(admission, started)

    if not is_supported_content_type(audio.content_type):
        raise HTTPException(
            status_code=415,
            detail=f"unsupported audio type: {audio.content_type}",
        )
    clip = await _read_capped(audio, limit=MAX_AUDIO_BYTES)
    if not clip:
        raise HTTPException(status_code=422, detail="empty audio upload")

    if container.stt is None:
        raise HTTPException(status_code=503, detail="stt_not_configured")

    session = container.sessions.get(session_id)

    def refuse(reason: str, *, remember: bool = False) -> JSONResponse:
        """Refuse in the transport shape the client already handles.

        Text plus the flag, never audio: there is nothing grounded to say, and
        the client will speak the canned refusal with speechSynthesis.
        """
        result = QAResult(transcript="", answer=None, refused=True, reason=reason)
        if remember and session is not None:
            session.remember("assistant", result.answer_text)
        return JSONResponse(
            status_code=200,
            content={
                "transcript": result.transcript,
                "answer": result.answer_text,
                "refused": True,
                "refusal_reason": reason,
                "use_native_tts": True,
            },
            headers={
                **fallback_headers(use_native_tts=True, reason=reason),
                **qa_metadata_headers(
                    result,
                    timings={"qa_total": int((time.monotonic() - started) * 1000)},
                ),
            },
        )

    if session is None:
        # No cached facts means nothing can be grounded at all.
        return refuse("no_session")

    if context_is_stale(session, lat, lon):
        return refuse("context_stale")

    # --- Transcribe --------------------------------------------------------
    transcribe_started = time.monotonic()
    try:
        transcript = await container.stt.transcribe(
            audio=clip,
            filename=audio.filename or "clip.m4a",
            content_type=audio.content_type or "audio/mp4",
            vocabulary=[f.title for f in session.facts],
        )
    except STTRateLimited as exc:
        logger.warning("Groq rate limited: %s", exc)
        raise HTTPException(
            status_code=429, detail="transcription_rate_limited"
        ) from exc
    except STTError as exc:
        logger.warning("transcription failed: %s", exc)
        raise HTTPException(status_code=502, detail="transcription_failed") from exc
    transcribe_ms = int((time.monotonic() - transcribe_started) * 1000)

    # --- Answer ------------------------------------------------------------
    answer_started = time.monotonic()
    result = await answer_question(
        question=transcript,
        facts=session.facts,
        provider=container.llm or NullLLMProvider(),
        history=session.history,
        narration=session.narration_text,
    )
    answer_ms = int((time.monotonic() - answer_started) * 1000)

    # Approximate spend: characters of question and answer stand in for tokens,
    # because the provider seam does not surface usage. Recorded only when a
    # model was actually configured.
    if container.llm is not None:
        container.guardrails.ledger.record_llm(
            len(transcript) + len(result.answer_text)
        )

    # --- Synthesize --------------------------------------------------------
    # Only the first chunk is awaited: enough to know whether synthesis is
    # possible, without paying for the whole clip before speaking. That is what
    # keeps the response streamable *and* lets the fallback flag be accurate at
    # header-flush time.
    tts_started = time.monotonic()
    attempt = await container.tts_queue.stream_sentence(result.answer_text)
    tts_ms = int((time.monotonic() - tts_started) * 1000)

    if attempt.ok:
        # Exact: the characters ElevenLabs was asked to say.
        container.guardrails.ledger.record_tts(len(result.answer_text))

    timings = {
        "transcribe": transcribe_ms,
        "answer": answer_ms,
        # Time to first audio byte from the TTS stage, not the full clip.
        "tts": tts_ms,
        "qa_total": int((time.monotonic() - started) * 1000),
    }

    session.remember("user", result.transcript)
    session.remember("assistant", result.answer_text)
    metadata = qa_metadata_headers(result, timings=timings)

    if not attempt.ok:
        # Post-answer failure: the flag goes out in a header, and the body is
        # JSON carrying the text the client will speak with speechSynthesis.
        logger.info("QA falling back to native TTS: %s", attempt.reason)
        return JSONResponse(
            status_code=200,
            content={
                "transcript": result.transcript,
                "answer": result.answer_text,
                "refused": result.refused,
                "refusal_reason": result.reason,
                "citations": list(result.citations),
                "source_urls": list(result.source_urls),
                "use_native_tts": True,
                "reason": attempt.reason,
            },
            headers={
                **fallback_headers(
                    use_native_tts=True, reason=attempt.reason or "tts_unavailable"
                ),
                **metadata,
            },
        )

    # Success: stream the audio through as ElevenLabs produces it. The client
    # already holds the answer text in the headers, so if the provider dies
    # mid-clip the client can still finish the answer with speechSynthesis.
    return StreamingResponse(
        attempt.chunks,
        media_type=attempt.content_type or "audio/mpeg",
        headers={
            **fallback_headers(use_native_tts=False),
            **metadata,
            "Cache-Control": "private, no-store",
        },
    )


__all__ = ["router"]
