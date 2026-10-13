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
from app import config as app_config
from app.dependencies import ContainerDep
from app.models import ContextResponse, FactObject, WalkRequest, WeatherInfo
from app.services.audio_tokens import AudioBlob
from app.services.context import aggregate_context
from app.services.demo import DEMO_FACTS, REMINDER_CLIP, demo_walk_frames
from app.services.narration import generate_narration
from app.services.pacing import (
    QA_GRACE_MS,
    SERP_REFRESH_SECONDS,
    count_words,
    cycle_kind,
    fit_to_budget,
    is_expired,
    news_key,
    plan_cycle,
    plan_pad_ms,
    plan_pause_ms,
    remaining_seconds,
    select_cycle_context,
)
from app.services.prompts import (
    PEDESTRIAN_REMINDER,
    WALK_HANDOFF,
    NoticeContext,
    build_greeting,
)
from app.services.providers import LLMTemporaryOutage, NullLLMProvider
from app.services.sessions import WalkSession
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


def _dumps(frame: dict[str, Any]) -> str:
    return json.dumps(frame) + "\n"


def _text_frame(
    *,
    index: int,
    content: str,
    citations: list[str],
    source_urls: list[str],
    mode: str,
    use_native_tts: bool,
    tts_reason: str | None = None,
    audio_url: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """A playable sentence. ``text`` aliases ``content`` for older clients."""
    frame: dict[str, Any] = {
        "type": "text",
        "content": content,
        "text": content,
        "index": index,
        "citations": citations,
        "source_urls": source_urls,
        "use_native_tts": use_native_tts,
        "mode": mode,
    }
    if audio_url:
        frame["audio_url"] = audio_url
    if tts_reason:
        frame["tts_reason"] = tts_reason
    if role:
        frame["role"] = role
    return frame


def _audio_frame(
    *,
    index: int,
    url: str | None,
    use_native_tts: bool,
    tts_reason: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "type": "audio",
        "index": index,
        "use_native_tts": use_native_tts,
    }
    if url:
        frame["url"] = url
        frame["audio_url"] = url
    if tts_reason:
        frame["tts_reason"] = tts_reason
    if role:
        frame["role"] = role
    return frame


def _complete_frames(
    session: WalkSession, *, mode: str, extra: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Reminder plus the terminal indicator. The client plays, then stops."""
    reminder_url = f"/static/{REMINDER_CLIP}"
    body = extra or {}
    return [
        _text_frame(
            index=session.sentence_index,
            content=PEDESTRIAN_REMINDER,
            citations=[],
            source_urls=[],
            mode=mode,
            use_native_tts=False,
            audio_url=reminder_url,
            role="reminder",
        ),
        _audio_frame(
            index=session.sentence_index,
            url=reminder_url,
            use_native_tts=False,
            role="reminder",
        ),
        {
            "type": "indicator",
            "status": "walk_complete",
            "complete": True,
            "session_id": session.session_id,
            "mode": mode,
            "grace_ms": QA_GRACE_MS,
            **body,
        },
    ]


def _prior_context(session: WalkSession) -> ContextResponse:
    return ContextResponse(
        mode="STORY" if session.facts else "NOTICE",
        facts=list(session.facts),
        weather=session.weather,
        news=list(session.news),
    )


def _refresh_facts(
    fresh: list[FactObject], previous: list[FactObject]
) -> list[FactObject]:
    """Facts for the next cycle: this location's, topped up from the last one.

    A refresh from the walker's new position replaces the pool, because it
    describes where they are now. When the upstream answered thinly (a slow
    Wikipedia is how a walk ends up with two facts), the earlier facts are kept
    behind the new ones rather than lost, so one bad lookup cannot empty a walk.
    """
    from app.config import MIN_CONTEXT_FACTS

    if len(fresh) >= MIN_CONTEXT_FACTS:
        return list(fresh)
    seen = {fact.fact_id for fact in fresh}
    return list(fresh) + [fact for fact in previous if fact.fact_id not in seen]


@router.post("/api/walk", summary="Narrate a walk step end to end")
async def api_walk(
    payload: WalkRequest,
    request: Request,
    container: ContainerDep,
    admitted: AdmittedWalkDep,
) -> Response:
    """One cycle of a walk, or the terminal complete indicator.

    The client is a dumb player: it posts ``session_id`` (and fresh GPS), plays
    the frames, then asks again when its local ambient bed ends. This handler
    owns the state machine -- duration expiry, round-robin context, whether
    the walk is done -- and never holds the HTTP connection across a pause.
    """
    lat = round(payload.lat, 3)
    lon = round(payload.lon, 3)

    limit = container.guardrails.limiter.limit
    session = container.sessions.get(payload.session_id)
    continuing = session is not None

    # Continuations are the same walk: they must not spend another walk credit
    # and must not 429 a walker who used their last credit on cycle one.
    if not continuing and not admitted.admission.allowed:
        raise HTTPException(
            status_code=429,
            detail="walk_limit_reached",
            headers={
                **rate_headers(admitted.admission, limit=limit),
                **demo_headers(False),
            },
        )

    if not continuing and admitted.admission.demo:
        return _demo_response(
            payload,
            container,
            admitted,
            rate_headers(admitted.admission, limit=limit),
        )

    if not continuing:
        charged = admitted.guardrails.begin_walk(admitted.identity)
        quota = rate_headers(
            replace(admitted.admission, walks_remaining=charged.remaining),
            limit=limit,
        )
    else:
        quota = rate_headers(admitted.admission, limit=limit)
        container.sessions.touch(session)

    if continuing:
        duration_min = (
            session.duration_min
            if session.duration_min is not None
            else payload.duration
        )
        if is_expired(
            started_at=session.created_at,
            duration_min=duration_min,
            now=container.sessions.now(),
        ):
            mode = "STORY" if session.facts else "NOTICE"

            async def done_frames() -> AsyncIterator[str]:
                for frame in _complete_frames(session, mode=mode):
                    yield _dumps(frame)

            return StreamingResponse(
                done_frames(),
                media_type="application/x-ndjson",
                headers={**quota, **demo_headers(False)},
            )

    # Size this cycle from the time left. ``None`` for the one-shot walk (no
    # duration), which keeps the legacy 4-6 sentence shape.
    duration_for_plan = session.duration_min if continuing else payload.duration
    plan = None
    if duration_for_plan is not None:
        started_at = session.created_at if continuing else container.sessions.now()
        remaining_s = remaining_seconds(
            started_at=started_at,
            duration_min=duration_for_plan,
            now=container.sessions.now(),
        )
        plan = plan_cycle(remaining_s) if remaining_s is not None else None
        if plan is None and continuing:
            # Less than a cycle's worth of time is left: the walk is over.
            mode = "STORY" if session.facts else "NOTICE"

            async def finished_frames() -> AsyncIterator[str]:
                for frame in _complete_frames(session, mode=mode):
                    yield _dumps(frame)

            return StreamingResponse(
                finished_frames(),
                media_type="application/x-ndjson",
                headers={**quota, **demo_headers(False)},
            )

    paid = True
    prior = _prior_context(session) if continuing else None
    if continuing and session is not None:
        paid = (
            container.sessions.now() - session.last_serp_at
        ) >= SERP_REFRESH_SECONDS

    context = await aggregate_context(
        container.http,
        lat=lat,
        lon=lon,
        paid=paid,
        prior=prior,
        # The walker is waiting for the first word on cycle one, but sitting in
        # an ambient gap on every cycle after it: be patient, lose fewer facts.
        budget=app_config.EXTERNAL_TIMEOUT_SECONDS * 2 if continuing else None,
    )

    if not continuing:
        session = container.sessions.create(
            facts=context.facts,
            anchor=(lat, lon),
            session_id=payload.session_id,
            duration_min=payload.duration,
            mood=payload.mood,
            news=context.news,
            weather=context.weather,
        )
    else:
        session.facts = _refresh_facts(context.facts, session.facts)
        session.news = list(context.news)
        session.weather = context.weather
        session.anchor = (lat, lon)
        if paid:
            session.last_serp_at = container.sessions.now()

    kind = cycle_kind(session.cycle_index, duration_min=session.duration_min)
    slice_ = select_cycle_context(
        kind,
        facts=session.facts,
        news=session.news,
        weather=session.weather,
        used_fact_ids=session.used_fact_ids,
    )
    oneshot = session.duration_min is None
    first_cycle = session.cycle_index == 0
    # A duration walk with no wiki/park facts for this slice still has weather.
    effective_mode = (
        "STORY"
        if slice_.facts
        or slice_.news
        or slice_.kind == "atmosphere"
        or context.mode == "STORY"
        else context.mode
    )
    if slice_.kind == "atmosphere":
        effective_mode = "STORY"

    # The hand-off ("I'll leave you to wander...") is said once, at the end of
    # the first cycle. Repeating it every cycle was the line walkers kept
    # hearing; a final cycle has nobody left to wander off.
    if oneshot:
        handoff = WALK_HANDOFF if context.mode == "STORY" else None
    else:
        handoff = (
            WALK_HANDOFF
            if first_cycle and not (plan is not None and plan.final)
            else None
        )

    try:
        narration = await generate_narration(
            mode=effective_mode if slice_.kind != "full" else context.mode,
            provider=container.llm or NullLLMProvider(),
            facts=slice_.facts,
            greeting=_greeting_for(slice_.weather or context.weather)
            if first_cycle
            else None,
            handoff=handoff,
            mood=session.mood or payload.mood,
            weather=slice_.weather or context.weather,
            news=slice_.news,
            cycle_kind=slice_.kind,
            cycle_words=plan.words if plan is not None else None,
            already_said=session.narration_text if not first_cycle else "",
            **(
                {"max_story_sentences": plan.sentences}
                if plan is not None
                else {}
            ),
        )
    except LLMTemporaryOutage as exc:
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

    sentences = list(narration.sentences)
    if not oneshot and narration.used_static_fallback:
        # Every model rung failed and the generic copy is all there is. Say it
        # once; after that, silence is better than hearing it on a loop.
        if session.static_served:
            sentences = []
        session.static_served = True
    if plan is not None and sentences:
        keep = fit_to_budget([count_words(s.text) for s in sentences], plan.words)
        sentences = [sentences[i] for i in keep]
    if sentences != list(narration.sentences):
        narration = replace(narration, sentences=sentences)
    spoken_words = sum(count_words(s.text) for s in sentences)

    body = " ".join(s.text for s in sentences if s.text)
    session.narration_text = (
        f"{session.narration_text} {body}".strip() if session.narration_text else body
    )
    for fact in slice_.facts:
        if fact.fact_id not in session.used_fact_ids:
            session.used_fact_ids.append(fact.fact_id)
    for item in slice_.news:
        key = news_key(item)
        if key not in session.used_fact_ids:
            session.used_fact_ids.append(key)
    session.cycle_index += 1

    if container.llm is not None:
        prompt_chars = sum(len(f.title) + len(f.snippet) for f in slice_.facts)
        container.guardrails.ledger.record_llm(
            prompt_chars + len(narration.spoken_text)
        )

    async def frames() -> AsyncIterator[str]:
        yield _dumps(
            {
                "type": "data",
                "cycle": session.cycle_index - 1,
                "context": slice_.kind,
                "mode": narration.mode,
                "session_id": session.session_id,
                "plan": (
                    {
                        "words": plan.words,
                        "sentences": plan.sentences,
                        "final": plan.final,
                        "remaining_s": round(plan.remaining_s, 1),
                    }
                    if plan is not None
                    else None
                ),
                "weather": (
                    context.weather.model_dump()
                    if context.weather is not None
                    else None
                ),
            }
        )

        for sentence in sentences:
            index = session.sentence_index
            session.sentence_index += 1
            outcome = await container.tts_queue.synthesize(sentence.text)
            if outcome.ok:
                container.guardrails.ledger.record_tts(len(sentence.text))
            audio_url: str | None = None
            if outcome.ok:
                token = container.audio_tokens.issue(
                    session_id=session.session_id,
                    blob=AudioBlob(
                        data=outcome.audio or b"",
                        content_type=outcome.content_type or "audio/mpeg",
                        text=sentence.text,
                    ),
                )
                audio_url = f"/api/audio/{token}"
            else:
                logger.info(
                    "TTS unavailable for sentence %d: %s", index, outcome.reason
                )
            yield _dumps(
                _text_frame(
                    index=index,
                    content=sentence.text,
                    citations=list(sentence.citations),
                    source_urls=list(sentence.source_urls),
                    mode=narration.mode,
                    use_native_tts=not outcome.ok,
                    tts_reason=None if outcome.ok else outcome.reason,
                    audio_url=audio_url,
                )
            )
            yield _dumps(
                _audio_frame(
                    index=index,
                    url=audio_url,
                    use_native_tts=not outcome.ok,
                    tts_reason=None if outcome.ok else outcome.reason,
                )
            )
            if await request.is_disconnected():
                logger.info("walk client disconnected; stopping dispatch")
                return

        meta = {
            "mode": narration.mode,
            "degraded": narration.degraded,
            "used_static_fallback": narration.used_static_fallback,
            "session_id": session.session_id,
            "rejections": [str(r) for r in narration.rejections],
            "weather": (
                context.weather.model_dump()
                if context.weather is not None
                else None
            ),
        }
        if oneshot or (plan is not None and plan.final):
            extra = dict(meta)
            if plan is not None:
                # A final cycle that came in short (the model wrote fewer words
                # than asked, or the validator cut some) is padded with ambient
                # silence so the walk still ends on its chosen duration.
                extra["pad_ms"] = plan_pad_ms(plan.remaining_s, spoken_words)
            for frame in _complete_frames(
                session, mode=narration.mode, extra=extra
            ):
                yield _dumps(frame)
            return

        assert plan is not None
        yield _dumps(
            {
                "type": "indicator",
                "status": "cycle_complete",
                "complete": False,
                "pause_ms": plan_pause_ms(plan.remaining_s, spoken_words),
                "cycle": session.cycle_index - 1,
                "context": slice_.kind,
                **meta,
            }
        )

    return StreamingResponse(
        frames(),
        media_type="application/x-ndjson",
        headers={**quota, **demo_headers(False)},
    )


__all__ = ["router"]
