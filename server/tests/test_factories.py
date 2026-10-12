"""Tests for the provider factories -- specifically the local voice switch.

``ECHOES_FORCE_NATIVE_TTS`` exists so a local run never spends the ElevenLabs
free tier even when the keys are present in ``.env``. These tests pin both
directions: forced builds no provider (clients get ``use_native_tts``), and
unforced builds the real one.
"""

from __future__ import annotations

import httpx

from app.services import factories


def _client() -> httpx.AsyncClient:
    # Never used for a request -- the factory only stores it -- so it is not
    # opened or closed here.
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )


def test_force_native_tts_builds_no_provider(monkeypatch) -> None:
    monkeypatch.setattr(factories, "FORCE_NATIVE_TTS", True)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")

    client = _client()
    assert factories.build_tts_provider(client) is None
    # The queue reports itself unconfigured, which is what drives
    # use_native_tts on the wire.
    assert factories.build_tts_queue(client).configured is False


def test_remote_provider_built_when_not_forced(monkeypatch) -> None:
    monkeypatch.setattr(factories, "FORCE_NATIVE_TTS", False)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")

    client = _client()
    assert factories.build_tts_provider(client) is not None
    assert factories.build_tts_queue(client).configured is True
