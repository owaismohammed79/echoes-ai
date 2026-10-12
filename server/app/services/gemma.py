"""Gemma 4 streaming providers and the narration failover cascade.

Talks to the hosted Gemini API's Gemma endpoint over SSE, using ``httpx``
(already a project dependency) rather than adding a vendor SDK. This module owns
both rungs of the cascade: :class:`GemmaProvider` (Google AI Studio) and
:class:`GroqGemmaProvider` (Groq's OpenAI-compatible endpoint), composed by
:class:`FailoverLLMProvider` under a single retry/time budget.

Verified against the official model list: ``gemma-4-31b-it`` and
``gemma-4-26b-a4b-it`` are the hosted Gemma 4 models, and the API accepts a
``systemInstruction`` plus a ``thinkingConfig``.

NOT verified against the live service: no API key was available, so the
``:streamGenerateContent`` path, the SSE frame shape, and the exact
``thinkingLevel`` spelling are implemented from documentation. Section 3 of
docs/llm-orchestration.md flags these as needing confirmation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Sequence

import httpx

from app.services.providers import (
    LLMError,
    LLMOutageError,
    LLMProvider,
    LLMTemporaryOutage,
)
from app.services.redact import redact

logger = logging.getLogger("echoes.llm.gemma")

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL = "gemma-4-26b-a4b-it"

#: The second rung of the cascade: the same Gemma family, reached through Groq's
#: OpenAI-compatible endpoint under a separate key, when Google AI Studio is
#: returning 5xx or timing out.
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "gemma2-9b-it"
DEFAULT_GROQ_TIMEOUT_SECONDS = 10.0

#: Statuses that mean "the provider is having a moment" rather than "this
#: request was wrong": the cascade fails over and retries on these, and treats
#: every other non-200 (a 400, a 403) as terminal.
TRANSIENT_STATUS_CODES = frozenset({500, 502, 503, 504})

#: Cumulative wall-clock budget for the whole failover cascade, and the backoff
#: ladder between successive attempts (the first failover is immediate).
DEFAULT_FAILOVER_BUDGET_SECONDS = 15.0
DEFAULT_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0)

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_TEMPERATURE = 0.7
#: Enough tokens for the 4-6 sentences a story chunk asks for, including the
#: citations they carry. A cap that lands mid-sentence is worse than a slower
#: one: a truncated tail is still a sentence as far as the validator is
#: concerned, and it would be spoken.
DEFAULT_MAX_OUTPUT_TOKENS = 320


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
                    message = (
                        f"Gemma returned HTTP {response.status_code}: "
                        f"{redact(body[:300])}"
                    )
                    if response.status_code in TRANSIENT_STATUS_CODES:
                        raise LLMOutageError(message)
                    raise LLMError(message)

                async for line in response.aiter_lines():
                    chunk = self._parse_sse_line(line)
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as exc:
            # A timed-out request is the other half of "the provider is having
            # a moment": the cascade fails over rather than degrading.
            raise LLMOutageError(f"Gemma timed out: {redact(exc)}") from exc
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


class GroqGemmaProvider:
    """Streams narration from Groq's OpenAI-compatible chat endpoint.

    The cascade's second rung: the same Gemma family is reachable through Groq
    under a separate key, so an AI Studio outage does not drop the walk to
    static copy. Groq speaks OpenAI's SSE dialect rather than Gemini's, so the
    frame shape differs from :class:`GemmaProvider` while the interface -- the
    ``stream(system=..., user=...)`` generator -- is identical.

    NOT verified against the live service: no Groq key was available, so the
    endpoint path and the streaming chunk shape come from Groq's published
    OpenAI compatibility documentation.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_GROQ_MODEL,
        base_url: str = DEFAULT_GROQ_BASE_URL,
        timeout: float = DEFAULT_GROQ_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GroqGemmaProvider requires an API key")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._client = client

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "stream": True,
        }

    async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
        """Yield narration text chunks as they arrive."""
        payload = self._payload(system, user)
        headers = {"Authorization": f"Bearer {self._api_key}"}

        if self._client is not None:
            async for chunk in self._stream_with(
                self._client, payload, headers
            ):
                yield chunk
            return

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async for chunk in self._stream_with(client, payload, headers):
                yield chunk

    async def _stream_with(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[str]:
        try:
            async with client.stream(
                "POST", self.endpoint, json=payload, headers=headers
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    message = (
                        f"Groq returned HTTP {response.status_code}: "
                        f"{redact(body[:300])}"
                    )
                    if response.status_code in TRANSIENT_STATUS_CODES:
                        raise LLMOutageError(message)
                    raise LLMError(message)

                async for line in response.aiter_lines():
                    chunk = self._parse_sse_line(line)
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise LLMOutageError(f"Groq timed out: {redact(exc)}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Groq transport failure: {redact(exc)}") from exc

    @staticmethod
    def _parse_sse_line(line: str) -> str | None:
        """Decode one OpenAI-style SSE line into text, or None if it carries none."""
        if not line or not line.startswith("data:"):
            return None
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            return None

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("skipping unparseable Groq frame: %.200s", data)
            return None

        if error := payload.get("error"):
            raise LLMError(f"Groq reported an error: {error}")

        texts: list[str] = []
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                texts.append(content)
        return "".join(texts) if texts else None


class FailoverLLMProvider:
    """Runs the provider cascade under a single cumulative time budget.

    Each attempt streams from the next provider in turn; the first failover is
    immediate (a Google AI Studio 500 hands straight over to Groq), and later
    retries back off on the ``1s, 2s`` ladder while the 15-second budget holds.
    When the budget is spent, the walk is told the truth: an
    :class:`LLMTemporaryOutage` the API turns into a structured error.

    A provider that fails *after* yielding text is not retried: the walk has
    already heard those sentences, and re-running would repeat them. That
    matches the orchestrator's rule that a mid-stream error degrades rather
    than replays (docs/llm-orchestration.md 7.8).
    """

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        budget: float = DEFAULT_FAILOVER_BUDGET_SECONDS,
        backoff: Sequence[float] = DEFAULT_BACKOFF_SECONDS,
        clock: Any = time.monotonic,
        sleep: Any = asyncio.sleep,
    ) -> None:
        if not providers:
            raise ValueError("FailoverLLMProvider needs at least one provider")
        self._providers = list(providers)
        self._budget = budget
        self._backoff = list(backoff) or [0.0]
        self._clock = clock
        self._sleep = sleep

    async def stream(self, *, system: str, user: str) -> AsyncIterator[str]:
        deadline = self._clock() + self._budget
        last_error: Exception | None = None
        attempt = 0

        while True:
            provider = self._providers[attempt % len(self._providers)]
            started = False
            try:
                async for chunk in provider.stream(system=system, user=user):
                    started = True
                    yield chunk
                return
            except LLMError as exc:
                if started:
                    # Text is already on its way; replaying it would repeat
                    # sentences the walker has heard.
                    raise
                last_error = exc
                logger.warning(
                    "narration provider %d failed (%s); considering failover",
                    attempt,
                    exc,
                )

            attempt += 1
            # The first hand-off is immediate; every retry after it waits.
            delay = (
                0.0
                if attempt == 1
                else self._backoff[min(attempt - 2, len(self._backoff) - 1)]
            )
            if self._clock() + delay >= deadline:
                break
            if delay:
                await self._sleep(delay)

        raise LLMTemporaryOutage(
            str(last_error) if last_error is not None else None
        )


__all__ = [
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_BASE_URL",
    "DEFAULT_FAILOVER_BUDGET_SECONDS",
    "DEFAULT_GROQ_BASE_URL",
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_MODEL",
    "FailoverLLMProvider",
    "GemmaProvider",
    "GroqGemmaProvider",
    "LLMError",
    "TRANSIENT_STATUS_CODES",
    "extract_text_parts",
]
