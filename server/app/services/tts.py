"""ElevenLabs speech synthesis.

Implements section 7 of docs/audio-streaming.md.

HTTP streaming (``/v1/text-to-speech/{voice_id}/stream``) rather than batch or
WebSocket. The WebSocket path has the lowest time-to-first-byte, but it exists
for text that arrives incrementally; our unit of work is a *complete validated
sentence*, which is precisely the case HTTP streaming is documented for. The
trade-off to revisit later is concurrency accounting: over HTTP every in-flight
request holds a provider slot for its full duration, over WebSocket only active
generation counts.

NOT verified against the live service: no API key was available. Endpoint path,
output-format string, model id and the 429 body shape come from documentation.
Confirm each against the live API before building on it.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger("echoes.speech.tts")

DEFAULT_BASE_URL = "https://api.elevenlabs.io"

#: Lowest-latency multilingual model (~75ms inference). The docs describe
#: ``eleven_multilingual_v2`` as the *narration* model, so this is a deliberate
#: latency-over-fidelity default (docs/audio-streaming.md 7), overridable by env.
DEFAULT_MODEL = "eleven_flash_v2_5"

#: MP3: what ``<audio>`` plays progressively on iOS, at a fraction of PCM's
#: bandwidth. ~64kbps mono is ample for voice.
DEFAULT_OUTPUT_FORMAT = "mp3_44100_64"

DEFAULT_TIMEOUT_SECONDS = 3.0


class TTSError(RuntimeError):
    """Speech synthesis failed for a reason that may or may not be permanent."""


class TTSRateLimited(TTSError):
    """HTTP 429 from ElevenLabs."""


class ElevenLabsTTS:
    """Streams synthesized audio for one piece of text."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = DEFAULT_MODEL,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("ElevenLabsTTS requires an API key")
        if not voice_id:
            raise ValueError("ElevenLabsTTS requires a voice id")
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._output_format = output_format
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/v1/text-to-speech/{self._voice_id}/stream"

    @property
    def content_type(self) -> str:
        # Every supported output format is MP3 today; kept as a property so a
        # switch to PCM or mu-law only has to change this mapping.
        return "audio/mpeg"

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        """Yield audio chunks for ``text`` as ElevenLabs produces them.

        ``text`` must already be tag-stripped by the validator. A ``[cite:...]``
        tag sent here would be narrated as literal punctuation.
        """
        params = {"output_format": self._output_format}
        payload = {"text": text, "model_id": self._model}
        headers = {"xi-api-key": self._api_key, "Accept": "audio/mpeg"}

        if self._client is not None:
            async for chunk in self._stream_with(
                self._client, params, payload, headers
            ):
                yield chunk
            return

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async for chunk in self._stream_with(client, params, payload, headers):
                yield chunk

    async def _stream_with(
        self,
        client: httpx.AsyncClient,
        params: dict[str, str],
        payload: dict[str, str],
        headers: dict[str, str],
    ) -> AsyncIterator[bytes]:
        try:
            async with client.stream(
                "POST",
                self.endpoint,
                params=params,
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code == 429:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise TTSRateLimited(f"ElevenLabs rate limited: {body[:200]}")
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise TTSError(
                        f"ElevenLabs returned HTTP {response.status_code}: {body[:200]}"
                    )
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as exc:
            raise TTSError(f"ElevenLabs transport failure: {exc}") from exc


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_FORMAT",
    "ElevenLabsTTS",
    "TTSError",
    "TTSRateLimited",
]
