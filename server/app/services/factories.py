"""Provider factories.

The only place the application decides whether a dependency is configured. Each
factory returns ``None`` rather than raising, because an unconfigured provider
must degrade a walk (to Notice Mode, or to native TTS), never fail it.

Keys are read from the environment here and nowhere else, so "is this
configured?" has exactly one answer.
"""

from __future__ import annotations

import logging
import os

import httpx

from app.config import FORCE_NATIVE_TTS, TTS_MAX_CONCURRENCY
from app.services.gemma import DEFAULT_GROQ_MODEL, DEFAULT_MODEL as LLM_DEFAULT_MODEL
from app.services.gemma import FailoverLLMProvider, GemmaProvider, GroqGemmaProvider
from app.services.providers import LLMProvider
from app.services.stt import DEFAULT_MODEL as STT_DEFAULT_MODEL
from app.services.stt import GroqWhisper
from app.services.tts import DEFAULT_BASE_URL as TTS_DEFAULT_BASE_URL
from app.services.tts import DEFAULT_MODEL as TTS_DEFAULT_MODEL
from app.services.tts import DEFAULT_OUTPUT_FORMAT, ElevenLabsTTS
from app.services.tts_queue import TTSQueue

logger = logging.getLogger("echoes.factories")

__all__ = [
    "build_llm_provider",
    "build_stt_provider",
    "build_tts_provider",
    "build_tts_queue",
]


def build_llm_provider() -> LLMProvider | None:
    """The narration cascade, or None when nothing is configured.

    Primary is Google AI Studio (Gemma 4); secondary is Groq's ``gemma2-9b-it``,
    reached under a separate key when AI Studio returns 5xx or times out. Both
    are wrapped in a :class:`FailoverLLMProvider` so even a single configured
    provider gets the retry budget and the structured outage signal, rather than
    failing straight to static copy.

    Each key is read here and nowhere else, so "is this rung configured?" has
    exactly one answer. No keys at all still returns ``None`` -- the walk stays
    available, narrated from static copy, instead of 500ing.
    """
    providers: list[LLMProvider] = []

    api_key = os.getenv("GEMMA_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    if api_key:
        providers.append(
            GemmaProvider(
                api_key=api_key, model=os.getenv("GEMMA_MODEL", LLM_DEFAULT_MODEL)
            )
        )
    else:
        logger.warning("GEMMA_API_KEY unset; the primary narration provider is off")

    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        providers.append(
            GroqGemmaProvider(
                api_key=groq_key,
                model=os.getenv("GROQ_LLM_MODEL", DEFAULT_GROQ_MODEL),
            )
        )
    else:
        logger.warning("GROQ_API_KEY unset; the narration fallback is off")

    if not providers:
        logger.warning("no narration provider configured; narration degrades to static copy")
        return None
    return FailoverLLMProvider(providers)


def build_tts_provider(client: httpx.AsyncClient) -> ElevenLabsTTS | None:
    """ElevenLabs, or None -- which routes the client to native TTS.

    A configured key is not enough: ``ECHOES_FORCE_NATIVE_TTS`` disables the
    remote voice outright, so a local run never spends the free-tier credits
    even when the keys are present in ``.env`` (AGENTS.md 4).
    """
    if FORCE_NATIVE_TTS:
        logger.warning(
            "ECHOES_FORCE_NATIVE_TTS is on; clients will be told use_native_tts"
        )
        return None
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
    if not api_key or not voice_id:
        logger.warning(
            "ELEVENLABS_API_KEY/ELEVENLABS_VOICE_ID unset; "
            "clients will be told use_native_tts"
        )
        return None
    return ElevenLabsTTS(
        api_key=api_key,
        voice_id=voice_id,
        model=os.getenv("ELEVENLABS_MODEL", TTS_DEFAULT_MODEL),
        output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT") or DEFAULT_OUTPUT_FORMAT,
        base_url=os.getenv("ELEVENLABS_BASE_URL", TTS_DEFAULT_BASE_URL),
        client=client,
    )


def build_stt_provider(client: httpx.AsyncClient) -> GroqWhisper | None:
    """Groq Whisper, or None when Push-To-Talk cannot be served."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY unset; /api/qa will return 503")
        return None
    return GroqWhisper(
        api_key=api_key,
        model=os.getenv("GROQ_MODEL", STT_DEFAULT_MODEL),
        client=client,
    )


def build_tts_queue(client: httpx.AsyncClient) -> TTSQueue:
    """A bounded synthesis queue, configured or not."""
    return TTSQueue(
        build_tts_provider(client),
        max_concurrency=int(
            os.getenv("TTS_MAX_CONCURRENCY", str(TTS_MAX_CONCURRENCY))
        ),
    )
