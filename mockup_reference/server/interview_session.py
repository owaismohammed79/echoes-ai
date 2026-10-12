from __future__ import annotations
import json
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services.turn_pipeline import run_ai_turn
from ..services.transcription_service import transcribe_bytes, is_no_speech
from ..services.vad_service import contains_speech

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/interview")
async def interview_ws(websocket: WebSocket):
    await websocket.accept()
    audio_chunks = []
    ai_turn_task = None

    try:
        while True:
            msg = await websocket.receive()

            if msg.get("bytes"):
                audio_chunks.append(msg["bytes"])
                logger.debug("Audio chunk received (%d bytes)", len(msg["bytes"]))

            elif msg.get("text"):
                data = json.loads(msg["text"])

                if data["type"] == "barge_in":
                    if ai_turn_task and not ai_turn_task.done():
                        ai_turn_task.cancel()
                        try:
                            await ai_turn_task
                        except asyncio.CancelledError:
                            pass
                    audio_chunks = []

                elif data["type"] == "end":
                    if not audio_chunks:
                        logger.warning("Chat end received with no audio chunks ,ignored")
                        continue

                    full_audio = b"".join(audio_chunks)
                    print("Complete audio found")
                    #clear the audio chunks so that when the turn starts, you don't have audio chunks of prev one
                    audio_chunks = []

                    try:
                        has_speech, max_vad_prob = await asyncio.to_thread(contains_speech, full_audio)
                    except ValueError as e:
                        logger.warning("vad Audio decode failed: %s", e)
                        has_speech = False
                        max_vad_prob = 0.0

                    if not has_speech:
                        logger.info("vad rejected audio (max_prob=%.3f), sending no_speech", max_vad_prob,)
                        await websocket.send_json({"type": "no_speech"})
                        continue

                    logger.info("vad passed, sending to stt", max_vad_prob)

                    result = await asyncio.to_thread(transcribe_bytes, full_audio)
                    if is_no_speech(result):
                        logger.info("stt rejected transcript, sending no_speech")
                        await websocket.send_json({"type": "no_speech"})
                        continue

                    transcript= result.text.strip()
                    logger.info("stt transcript: %r", transcript)

                    await websocket.send_json({"type": "transcript", "text": transcript})

                    ai_turn_task = asyncio.create_task(run_ai_turn(websocket, transcript))

                else:
                    logger.debug("Unknown msg type: %r", data["type"])

    except WebSocketDisconnect:
        print("Client disconnected")

    except asyncio.CancelledError:
        print("Server is shutting down, killing websocket...")
        raise

    except Exception as e:
        print("Error:", e)
        await websocket.close(code=1011)
    
    finally:
        if ai_turn_task and not ai_turn_task.done():
            ai_turn_task.cancel()
