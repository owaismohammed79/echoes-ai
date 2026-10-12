"""``POST /api/walk`` -- narrate a walk step end to end.

Implements sections 5, 6 and 9 of docs/audio-streaming.md.

Channel A (this endpoint) streams NDJSON metadata frames; Channel B
(``GET /api/audio/{token}``) serves the audio each frame points at. The two are
separate because audio wanted a cacheable, prefetchable, independently
abortable resource, while text stayed tiny and readable.

The gating rule is that the dispatcher only ever receives sentences the
validator already cleared. That costs single-digit milliseconds today and is the
first thing a latency-obsessed optimisation would try to remove, so it is stated
as a constraint rather than a preference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.limits import (
    AdmittedWalk,
    AdmittedWalkDep,
    demo_headers,
    rate_headers,
)
from app.dependencies import ContainerDep
from app.models import WalkRequest, WeatherInfo
from app.services.audio_tokens import AudioBlob
from app.services.context import aggregate_context
from app.services.demo import DEMO_FACTS, REMINDER_CLIP, demo_walk_frames
from app.services.narration import generate_narration
from app.services.prompts import (
    PEDESTRIAN_REMINDER,
    WALK_HANDOFF,
    NoticeContext,
    build_greeting,
)
from app.services.providers import LLMTemporaryOutage, NullLLMProvider
from app.services.weather import hour_from_clock

logger = logging.getLogger("echoes.api.walk")

router = APIRouter()


def _greeting_for(weather: WeatherInfo | None) -> str:
    """Build the spoken opener from the walker's local hour and real conditions.

    The hour is the provider's local time when there is a reading, and the
    server's clock only as a fallback. Those are different clocks: a walk at
    7 PM in Anchorage was greeted as a morning one because the deployment is on
    another meridian, which is the same mistake as inventing the weather --
    context about the walker that the walker can hear is wrong.

    The condition is whatever Open-Meteo reported; when there is no reading the
    greeting names only the daypart rather than inventing a sky (see
    ``prompts.build_greeting``).
    """
    hour = hour_from_clock(weather.local_time) if weather is not None else None
    daypart = NoticeContext.from_hour(
        hour if hour is not None else datetime.now().hour
    ).daypart
    return build_greeting(
        daypart=daypart,
        condition=weather.condition if weather is not None else None,
    )


def _demo_response(
    payload: WalkRequest,
    container: ContainerDep,
    admitted: AdmittedWalk,
    quota: dict[str, str],
) -> StreamingResponse:
    """The kill-switch: a recorded walk, labelled, costing nothing.

    A session is still created, over the demo facts, so a push-to-talk question
    asked during a demo walk has something grounded to answer from. The header is
    on the response, and each frame carries ``demo: true``, because a client that
    cannot tell a recorded walk from a generated one cannot label it.
    """
    admitted.guardrails.ledger.record_demo()
    session = container.sessions.create(
        facts=list(DEMO_FACTS),
        anchor=(payload.lat, payload.lon),
        session_id=payload.session_id,
    )

    async def demo_frames() -> AsyncIterator[str]:
        for frame in demo_walk_frames(session_id=session.session_id):
            yield json.dumps(frame) + "\n"

    logger.info("serving recorded demo walk: %s", admitted.admission.reason)
    return StreamingResponse(
        demo_frames(),
        media_type="application/x-ndjson",
        headers={
            **quota,
            **demo_headers(True, reason=admitted.admission.reason),
        },
    )


@router.post("/api/walk", summary="Narrate a walk step end to end")
async def api_walk(
    payload: WalkRequest,
    request: Request,
    container: ContainerDep,
    admitted: AdmittedWalkDep,
) -> Response:
    """Aggregate context, narrate, synthesize, and stream NDJSON frames."""
    lat = round(payload.lat, 3)
    lon = round(payload.lon, 3)

    limit = container.guardrails.limiter.limit

    # --- Guards. Neither of these may cost a paid API call. ---------------
    if not admitted.admission.allowed:
        raise HTTPException(
            status_code=429,
            detail="walk_limit_reached",
            headers={
                **rate_headers(admitted.admission, limit=limit),
                **demo_headers(False),
            },
        )

    if admitted.admission.demo:
        # A demo walk is free, so the allowance the client arrived with is still
        # the allowance they have.
        return _demo_response(
            payload,
            container,
            admitted,
            rate_headers(admitted.admission, limit=limit),
        )

    # Charged only here, once the walk is certain to be generated: a request
    # refused above, or served from the demo, never spends the allowance. The
    # quota headers are then built from the *post-charge* decision, so
    # X-RateLimit-Remaining reports what the client has left rather than what
    # they arrived with.
    charged = admitted.guardrails.begin_walk(admitted.identity)
    quota = rate_headers(
        replace(admitted.admission, walks_remaining=charged.remaining),
        limit=limit,
    )

    context = await aggregate_context(container.http, lat=lat, lon=lon)
    session = container.sessions.create(
        facts=context.facts,
        anchor=(lat, lon),
        session_id=payload.session_id,
    )

    try:
        narration = await generate_narration(
            mode=context.mode,
            provider=container.llm or NullLLMProvider(),
            facts=context.facts,
            greeting=_greeting_for(context.weather),
            # The hand-off belongs to a story block: Notice Mode has no story to
            # leave, so it is only appended when the walk is narrating facts.
            handoff=WALK_HANDOFF if context.mode == "STORY" else None,
            mood=payload.mood,
            weather=context.weather,
            # Headlines the narrator may allude to once (never state). Empty when
            # no news provider answered, in which case the prompt has no news
            # block at all and the model is told to ignore the section.
            news=context.news,
        )
    except LLMTemporaryOutage as exc:
        # Every provider in the cascade failed inside its retry budget. This is
        # the one narration failure the walker is told about rather than being
        # silently degraded to static copy: the structured error is a signal to
        # tap Retry, not a broken walk. Nothing has been streamed yet, so the
        # status can still be a 503 with a JSON body.
        logger.warning("narration cascade exhausted: %s", exc.detail or exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "code": exc.code,
                "message": exc.message,
            },
            headers={**quota, **demo_headers(False)},
        )

    # The validated narration the walk will speak, kept on the session so a
    # follow-up question can resolve what it refers back to.
    session.narration_text = narration.spoken_text

    # Spend is estimated, and recorded only when a model was actually configured:
    # the null provider makes no call and therefore costs nothing. The provider
    # seam does not surface token usage, so prompt and completion characters stand
    # in for it -- an approximation that is labelled as one wherever it is
    # reported, rather than dressed up as a bill.
    if container.llm is not None:
        prompt_chars = sum(len(f.title) + len(f.snippet) for f in context.facts)
        container.guardrails.ledger.record_llm(
            prompt_chars + len(narration.spoken_text)
        )

    async def frames() -> AsyncIterator[str]:
        for index, sentence in enumerate(narration.sentences):
            outcome = await container.tts_queue.synthesize(sentence.text)
            if outcome.ok:
                # Characters actually sent to ElevenLabs, counted per sentence as
                # it is synthesised rather than per walk: a barge-in closes this
                # stream early, and the sentences nobody heard must not inflate
                # the day's spend (docs/audio-streaming.md 12).
                container.guardrails.ledger.record_tts(len(sentence.text))
            frame: dict[str, Any] = {
                "type": "sentence",
                "index": index,
                "text": sentence.text,
                "citations": list(sentence.citations),
                "source_urls": list(sentence.source_urls),
                # Proactive fallback: the client is told before it asks for
                # audio it would not get.
                "use_native_tts": not outcome.ok,
                "mode": narration.mode,
            }
            if outcome.ok:
                token = container.audio_tokens.issue(
                    session_id=session.session_id,
                    blob=AudioBlob(
                        data=outcome.audio or b"",
                        content_type=outcome.content_type or "audio/mpeg",
                        text=sentence.text,
                    ),
                )
                frame["audio_url"] = f"/api/audio/{token}"
            else:
                frame["tts_reason"] = outcome.reason
                logger.info(
                    "TTS unavailable for sentence %d: %s", index, outcome.reason
                )
            yield json.dumps(frame) + "\n"

            # A barge-in closes the text stream early; stop paying for audio
            # nobody will hear (docs/audio-streaming.md 12).
            if await request.is_disconnected():
                logger.info("walk client disconnected; stopping dispatch")
                return

        # Pre-rendered asset: the reminder never touches the TTS provider, so
        # the last-resort path has no dependency on ElevenLabs at all.
        # The file has to exist for this to be worth anything: it is served by
        # the static mount in app.main, and /api/health reports when the recorded
        # clips are still placeholders.
        yield json.dumps(
            {
                "type": "reminder",
                "text": PEDESTRIAN_REMINDER,
                "audio_url": f"/static/{REMINDER_CLIP}",
            }
        ) + "\n"
        yield json.dumps(
            {
                "type": "done",
                "mode": narration.mode,
                "degraded": narration.degraded,
                "used_static_fallback": narration.used_static_fallback,
                "session_id": session.session_id,
                "rejections": [str(r) for r in narration.rejections],
                # Conditions the walk was narrated under, so the client can show
                # them alongside the facts. None when no provider answered.
                "weather": (
                    context.weather.model_dump()
                    if context.weather is not None
                    else None
                ),
            }
        ) + "\n"

    return StreamingResponse(
        frames(),
        media_type="application/x-ndjson",
        # Sent on the success path too: the client needs the quota readout, and
        # an explicit `False` is the only way it can tell "live" from "a proxy
        # stripped the header".
        headers={**quota, **demo_headers(False)},
    )


__all__ = ["router"]
