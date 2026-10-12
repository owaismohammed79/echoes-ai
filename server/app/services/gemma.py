"""Gemma 4 streaming provider.

Talks to the hosted Gemini API's Gemma endpoint over SSE, using ``httpx``
(already a project dependency) rather than adding a vendor SDK.

Verified against the official model list: ``gemma-4-31b-it`` and
``gemma-4-26b-a4b-it`` are the hosted Gemma 4 models, and the API accepts a
``systemInstruction`` plus a ``thinkingConfig``.

NOT verified against the live service: no API key was available, so the
``:streamGenerateContent`` path, the SSE frame shape, and the exact
``thinkingLevel`` spelling are implemented from documentation. Section 3 of
docs/llm-orchestration.md flags these as needing confirmation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.services.redact import redact

logger = logging.getLogger("echoes.llm.gemma")

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL = "gemma-4-26b-a4b-it"

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_TEMPERATURE = 0.7
#: Enough tokens for the 4-6 sentences a story chunk asks for, including the
#: citations they carry. A cap that lands mid-sentence is worse than a slower
#: one: a truncated tail is still a sentence as far as the validator is
#: concerned, and it would be spoken.
DEFAULT_MAX_OUTPUT_TOKENS = 320


class LLMError(RuntimeError):
    """The model could not be reached, or replied with an error."""


def extract_text_parts(payload: dict[str, Any]) -> list[str]:
    """Pull speakable text out of one streamed Gemini response object.

    Thought parts are dropped. Gemma 4 has an internal reasoning process that
    can be emitted as parts flagged ``thought: true``; speaking those would
    narrate the model's deliberation to a walker. They are filtered here so
    that even if the ``thinkingLevel`` request is ignored by the server, the
    reasoning never reaches TTS.
    """
    texts: list[str] = []

    candidates = payload.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if part.get("thought") is True:
                continue
            text = part.get("text")
            if text:
                texts.append(text)

    return texts


class GemmaProvider:
    """Streams narration tokens from the hosted Gemma 4 endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GemmaProvider requires an API key")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._client = client

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/v1beta/models/{self._model}:streamGenerateContent"

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_output_tokens,
                # Reasoning tokens are generated before any spoken text, which
                # spends the 2.5s time-to-first-audio budget directly.
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }

    async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
        """Yield narration text chunks as they arrive."""
        params = {"alt": "sse", "key": self._api_key}
        payload = self._payload(system, user)

        if self._client is not None:
            async for chunk in self._stream_with(self._client, params, payload):
                yield chunk
            return

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async for chunk in self._stream_with(client, params, payload):
                yield chunk

    async def _stream_with(
        self,
        client: httpx.AsyncClient,
        params: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncIterator[str]:
        try:
            async with client.stream(
                "POST", self.endpoint, params=params, json=payload
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    # Redacted for the same reason the transport failure is: this
                    # request carries the API key in its query string.
                    raise LLMError(
                        f"Gemma returned HTTP {response.status_code}: "
                        f"{redact(body[:300])}"
                    )

                async for line in response.aiter_lines():
                    chunk = self._parse_sse_line(line)
                    if chunk:
                        yield chunk
        except httpx.HTTPError as exc:
            # ``exc`` carries the request URL, and this URL holds the API key.
            raise LLMError(f"Gemma transport failure: {redact(exc)}") from exc

    @staticmethod
    def _parse_sse_line(line: str) -> str | None:
        """Decode one SSE line into speakable text, or None if it carries none."""
        if not line or not line.startswith("data:"):
            return None
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            return None

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("skipping unparseable SSE frame: %.200s", data)
            return None

        if error := payload.get("error"):
            raise LLMError(f"Gemma reported an error: {error}")

        texts = extract_text_parts(payload)
        return "".join(texts) if texts else None


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "GemmaProvider",
    "LLMError",
    "extract_text_parts",
]
