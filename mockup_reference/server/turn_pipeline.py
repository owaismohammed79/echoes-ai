from __future__ import annotations
import asyncio
import logging
from fastapi import WebSocket
from .ai_service import stream_ai_response
from .tts_service import synthesize_and_send

logger = logging.getLogger(__name__)


async def run_ai_turn(websocket: WebSocket, transcript: str):
    tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def tts_consumer():
        try:
            while True:
                sentence = await tts_queue.get()
                if sentence is None:
                    break
                await synthesize_and_send(sentence, websocket)
                tts_queue.task_done()
        except asyncio.CancelledError: # this is because we need to handoff the control back to uvicorn to end the server
            raise

        except Exception as e:
            logger.exception(f"tts consumer error: {e}")

    consumer_task = asyncio.create_task(tts_consumer())

    try:
        buffer = ""
        async for token in stream_ai_response(transcript):
            #Cancelled error can be raised at any of these awaits
            await websocket.send_json({ "type": "llm_text", "text": token })
            buffer += token
            if any(p in buffer for p in [".", "?", "!"]):
                await tts_queue.put(buffer)
                buffer = ""

        if buffer.strip():
            await tts_queue.put(buffer)

        await tts_queue.put(None)
        await consumer_task

        await websocket.send_json({ "type": "ai_turn_complete", "text": None })

    except asyncio.CancelledError:
        #user interrupted, cancel tts
        consumer_task.cancel()

        while not tts_queue.empty():
            tts_queue.get_nowait()

        #wait for consumer to finish cancelling
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        logger.info("AI turn cancelled by barge in")
        raise  #so that we know that the tts was cancelled rather than completing successfully
