"""Response headers for the audio and PTT endpoints.

The fallback flag travels in a **header**, not the body, because these endpoints
return audio: a JSON flag inside an MP3 stream is not readable by the client,
and headers are flushed before the body so the client knows what is coming
before it starts decoding.

The flag is emitted under two names on purpose. nginx -- and therefore most PaaS
front ends, Render included -- silently drops headers containing underscores
unless ``underscores_in_headers`` is enabled, so the hyphenated spelling is the
one that survives a real deployment and the underscored one is kept for clients
that already read it.
"""

from __future__ import annotations

from typing import Mapping
from urllib.parse import quote

from app.services.qa import QAResult

#: The canonical flag the client should read.
FALLBACK_HEADER = "X-Use-Native-Tts"

#: The legacy spelling, kept because proxies drop it rather than erroring.
FALLBACK_HEADER_LEGACY = "use_native_tts"


def fallback_headers(
    *, use_native_tts: bool, reason: str | None = None
) -> dict[str, str]:
    """The fallback flag, in both spellings (see :data:`FALLBACK_HEADER`)."""
    flag = "true" if use_native_tts else "false"
    headers = {FALLBACK_HEADER: flag, FALLBACK_HEADER_LEGACY: flag}
    if reason:
        headers["X-Fallback-Reason"] = reason
    return headers


def qa_metadata_headers(result: QAResult, *, timings: Mapping[str, int]) -> dict[str, str]:
    """Everything the client needs about a PTT answer, success or refusal alike.

    Carried in headers rather than the body because the body may be audio. Per-
    stage timings are included deliberately: PTT is the path users perceive as
    interactive, and per-stage numbers are the only way to tell which dependency
    is responsible when it feels slow (docs/audio-streaming.md 10.5).
    """
    # Every value is percent-encoded exactly ONCE. HTTP headers are latin-1, so
    # a transcript containing a place name outside that range would otherwise
    # raise at response-construction time.
    #
    # Encoding must not be applied twice: re-encoding an already-encoded value
    # turns `%27` into `%2527`, and the client then needs two decode passes to
    # recover the URL. Sources are encoded per item and joined verbatim so a
    # literal comma inside a URL cannot be mistaken for the separator.
    headers = {
        "X-Transcript": quote(result.transcript, safe=""),
        "X-Answer-Text": quote(result.answer_text, safe=""),
        "X-Answer-Citations": ",".join(result.citations),
        "X-Answer-Sources": ",".join(
            quote(url, safe="") for url in result.source_urls
        ),
        "X-Answer-Refused": "true" if result.refused else "false",
    }
    if result.reason:
        headers["X-Refusal-Reason"] = result.reason
    headers["X-Timings-Ms"] = ";".join(f"{k}={v}" for k, v in timings.items())
    return headers


__all__ = [
    "FALLBACK_HEADER",
    "FALLBACK_HEADER_LEGACY",
    "fallback_headers",
    "qa_metadata_headers",
]
