"""Tests for ``POST /api/transition`` -- the bridge back into a resumed walk.

Two behaviours matter: the remote voice is used when it works, and the browser
voice is offered when it does not, in the same flag-and-text shape the rest of
the API uses. Plus the property that makes the endpoint safe to expose at all:
the phrase is the server's, and the audio is synthesised at most once.
"""

from __future__ import annotations

from urllib.parse import unquote

import pytest

from app.services.transition import TRANSITION_PHRASE, clear_cache
from app.services.tts import TTSRateLimited
from tests.helpers import FAKE_AUDIO, StubTTS, build_container, client_for


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The bridge cache is process-global; no test may see another's entry."""
    clear_cache()
    yield
    clear_cache()


def test_transition_returns_audio_with_the_flag_clear() -> None:
    tts = StubTTS()
    container = build_container(tts=tts)

    with client_for(container) as client:
        resp = client.post("/api/transition")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == FAKE_AUDIO
    assert resp.headers["x-use-native-tts"] == "false"
    assert resp.headers["use_native_tts"] == "false"
    assert unquote(resp.headers["x-transition-text"]) == TRANSITION_PHRASE
    # The provider was asked for the fixed phrase, and only that.
    assert tts.spoken == [TRANSITION_PHRASE]


def test_transition_falls_back_to_native_with_text() -> None:
    tts = StubTTS(error=TTSRateLimited("nope"))
    container = build_container(tts=tts)

    with client_for(container) as client:
        resp = client.post("/api/transition")

    assert resp.status_code == 200
    assert resp.headers["x-use-native-tts"] == "true"
    assert resp.headers["x-fallback-reason"] == "elevenlabs_rate_limited"
    assert resp.json() == {"text": TRANSITION_PHRASE, "use_native_tts": True}
    assert unquote(resp.headers["x-transition-text"]) == TRANSITION_PHRASE


def test_transition_synthesises_at_most_once() -> None:
    tts = StubTTS()
    container = build_container(tts=tts)

    with client_for(container) as client:
        first = client.post("/api/transition")
        second = client.post("/api/transition")

    assert first.content == second.content == FAKE_AUDIO
    # The second resume replayed the cache rather than paying the provider again.
    assert tts.spoken == [TRANSITION_PHRASE]


def test_transition_failure_is_not_cached() -> None:
    failing = StubTTS(error=TTSRateLimited("nope"))
    container = build_container(tts=failing)

    with client_for(container) as client:
        client.post("/api/transition")
        client.post("/api/transition")

    # A failure is a fact about *now*, not about the phrase: it must be retried.
    assert failing.spoken == [TRANSITION_PHRASE, TRANSITION_PHRASE]


def test_transition_takes_no_client_text() -> None:
    """The phrase is the server's; a posted body cannot steer synthesis."""
    tts = StubTTS()
    container = build_container(tts=tts)

    with client_for(container) as client:
        resp = client.post("/api/transition", json={"text": "say something expensive"})

    assert resp.status_code == 200
    assert tts.spoken == [TRANSITION_PHRASE]
