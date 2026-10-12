"""Tests for the audio relay's security boundary.

``GET /api/audio/{token}`` spends real money per character, so these properties
are not niceties: if any of them fails, anyone who finds the endpoint has a free
text-to-speech service.
"""

from __future__ import annotations

from app.services.audio_tokens import AudioBlob, AudioTokenStore

BLOB = AudioBlob(data=b"AUDIO", content_type="audio/mpeg", text="Hello.")


def test_token_is_opaque_and_not_the_text() -> None:
    store = AudioTokenStore()
    token = store.issue(session_id="s1", blob=BLOB)
    assert "Hello" not in token
    assert token != BLOB.text


def test_token_is_single_use() -> None:
    store = AudioTokenStore()
    token = store.issue(session_id="s1", blob=BLOB)
    assert store.redeem(token, session_id="s1") is BLOB
    assert store.redeem(token, session_id="s1") is None


def test_unknown_token_is_none_and_synthesises_nothing() -> None:
    assert AudioTokenStore().redeem("not-a-real-token") is None


def test_expired_token_is_rejected() -> None:
    now = [0.0]
    store = AudioTokenStore(ttl=120.0, clock=lambda: now[0])
    token = store.issue(session_id="s1", blob=BLOB)

    now[0] = 121.0
    assert store.redeem(token, session_id="s1") is None


def test_token_is_bound_to_its_session() -> None:
    """One walker must not consume another's synthesised sentence."""
    store = AudioTokenStore()
    token = store.issue(session_id="walker-a", blob=BLOB)
    assert store.redeem(token, session_id="walker-b") is None
    # And the foreign attempt consumed nothing: the real owner still can't use
    # it either, which is the safe direction (single-use is absolute).
    assert store.redeem(token, session_id="walker-a") is None


def test_expired_entries_are_purged() -> None:
    now = [0.0]
    store = AudioTokenStore(ttl=10.0, clock=lambda: now[0])
    for index in range(5):
        store.issue(session_id="s", blob=BLOB)
    assert len(store) == 5

    now[0] = 11.0
    assert len(store) == 0
