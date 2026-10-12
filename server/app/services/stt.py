"""Groq Whisper transcription for Push-To-Talk.

Implements section 10.1 of docs/audio-streaming.md.

Groq exposes an OpenAI-compatible endpoint at
``POST {base}/audio/transcriptions``. Verified limits: 25MB (free tier) /
100MB (dev tier), minimum billed length 10 seconds, and a ``prompt`` of up to
224 tokens that can bias the model's spelling of unfamiliar words.

The guards here are the reason this is a separate module from the route: the
upload is attacker-controlled and every rejected clip is a paid API call that
never happens.

NOT verified against the live service: no API key was available.
"""

from __future__ import annotations

import logging
from typing import Final

import httpx

logger = logging.getLogger("echoes.speech.stt")

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

#: whisper-large-v3 (10.3% WER) over turbo (12%) because PTT questions are
#: short, so the cost delta is negligible while a mis-heard place name
#: degrades every downstream stage. Configurable via ``GROQ_MODEL``.
DEFAULT_MODEL = "whisper-large-v3"

DEFAULT_TIMEOUT_SECONDS = 20.0

#: Groq bills a minimum of 10 seconds per request regardless of clip length.
MIN_BILLED_SECONDS = 10

#: A PTT question is a few seconds of speech. Anything larger is a client bug
#: or abuse, and must be refused before it reaches a paid endpoint.
MAX_AUDIO_BYTES = 10 * 1024 * 1024

#: Format list confirmed against GroqDocs.
SUPPORTED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/mpeg",
        "audio/mpga",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/flac",
        "video/mp4",
    }
)

#: Whisper's documented prompt ceiling, in tokens. Approximated by characters,
#: which is conservative for short proper nouns.
VOCABULARY_PROMPT_LIMIT = 224


class STTError(RuntimeError):
    """Transcription failed."""


class STTRateLimited(STTError):
    """HTTP 429 from Groq."""


def is_supported_content_type(content_type: str | None) -> bool:
    """True when Groq will accept this upload type."""
    if not content_type:
        return False
    return content_type.split(";")[0].strip().lower() in SUPPORTED_CONTENT_TYPES


def build_vocabulary_prompt(titles: list[str], *, limit: int = VOCABULARY_PROMPT_LIMIT) -> str:
    """Bias Whisper toward the names the walker is standing next to.

    The cheapest available accuracy win for a place-based app: the landmark
    being asked about is already in the fact context, so its correct spelling
    can be seeded rather than guessed from phonetics.
    """
    seen: list[str] = []
    for title in titles:
        title = title.strip()
        if title and title not in seen:
            seen.append(title)
    if not seen:
        return ""
    prompt = ", ".join(seen)
    return prompt[:limit]


class GroqWhisper:
    """Transcribes short PTT clips."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        language: str = "en",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GroqWhisper requires an API key")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._language = language
        self._client = client

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/audio/transcriptions"

    async def transcribe(
        self,
        *,
        audio: bytes,
        filename: str = "clip.m4a",
        content_type: str = "audio/mp4",
        vocabulary: list[str] | None = None,
    ) -> str:
        """Return the transcript, or raise ``STTError``/``STTRateLimited``.

        Audio is posted straight through and never written to disk
        (docs/audio-streaming.md 10.6).
        """
        data = {
            "model": self._model,
            # Documented to improve both accuracy and latency.
            "language": self._language,
            "response_format": "json",
            "temperature": "0",
        }
        if vocabulary:
            if prompt := build_vocabulary_prompt(vocabulary):
                data["prompt"] = prompt

        headers = {"Authorization": f"Bearer {self._api_key}"}
        files = {"file": (filename, audio, content_type)}

        if self._client is not None:
            return await self._post(self._client, data, files, headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._post(client, data, files, headers)

    async def _post(
        self,
        client: httpx.AsyncClient,
        data: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
        headers: dict[str, str],
    ) -> str:
        try:
            response = await client.post(
                self.endpoint, data=data, files=files, headers=headers
            )
        except httpx.HTTPError as exc:
            raise STTError(f"Groq transport failure: {exc}") from exc

        if response.status_code == 429:
            raise STTRateLimited("Groq rate limited the transcription request")
        if response.status_code != 200:
            raise STTError(
                f"Groq returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise STTError("Groq returned a non-JSON response") from exc

        if error := payload.get("error"):
            raise STTError(f"Groq reported an error: {error}")

        return str(payload.get("text", "")).strip()


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "MAX_AUDIO_BYTES",
    "MIN_BILLED_SECONDS",
    "SUPPORTED_CONTENT_TYPES",
    "STTError",
    "STTRateLimited",
    "GroqWhisper",
    "build_vocabulary_prompt",
    "is_supported_content_type",
]
