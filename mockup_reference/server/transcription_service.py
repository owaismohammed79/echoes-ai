from __future__ import annotations
from app.config.config import groq_transcription_client
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)
 
_NO_SPEECH_PROB_THRESHOLD = 0.6

def transcribe_bytes(audio_bytes: bytes) -> dict:
    try:
        return groq_transcription_client.audio.transcriptions.create(
            file=("audio.webm", audio_bytes),
            model="whisper-large-v3-turbo",
            temperature=0,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
    except Exception as e:
        logger.exception("Groq Whisper transcription failed")
        raise HTTPException(status_code=500, detail=str(e))

def is_no_speech(transcription_result) -> bool:
    if not transcription_result.text.strip():
        logger.debug("blank transcript, treating as no speech")
        return True
 
    segments = getattr(transcription_result, "segments", None)
    if not segments:
        return False
 
    avg_no_speech_prob = sum(s.no_speech_prob for s in segments)/len(segments)
 
    if avg_no_speech_prob > _NO_SPEECH_PROB_THRESHOLD:
        logger.debug("STT: avg no_speech_prob=%.3f > threshold=%.2f, skipping LLM call", avg_no_speech_prob, _NO_SPEECH_PROB_THRESHOLD)
        return True
 
    return False