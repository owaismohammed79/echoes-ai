"""Tests for the cost guardrails: per-IP walk limits, the daily cap, demo mode.

The claims worth pinning down, in the order they can go wrong:

* the allowance belongs to the *client*, and the client is the address the
  trusted proxy reported -- not whatever a caller prepends to ``X-Forwarded-For``,
  which would otherwise make the limit a formality;
* a refused walk spends nothing, so a refusal cannot drain the allowance it is
  enforcing;
* crossing the cap serves the recorded demo, with the header on the response,
  without calling a single provider -- a kill-switch that still spends is not a
  kill-switch;
* the demo's audio URLs resolve to files that exist and are served, because one
  that fires into 404s leaves the walker listening to silence with no way to know
  why.
"""

from __future__ import annotations

import pytest

from app.api.limits import DEMO_HEADER
from app.services.budgets import (
    Guardrails,
    PerIPWalkLimiter,
    Prices,
    SpendLedger,
)
from app.services.demo import DEMO_FACTS, DEMO_SEGMENTS, missing_clips
from tests.helpers import (
    FID,
    FakeClock,
    FakeDay,
    ScriptedLLM,
    StubSTT,
    StubTTS,
    build_container,
    build_guardrails,
    client_for,
    ndjson,
)

BODY = {"lat": 51.5074, "lon": -0.1278}
CLIENT_IP = "203.0.113.7"
OTHER_IP = "198.51.100.4"
GROUNDED = [f"Grounded claim.[cite:{FID}]"]


def _post_walk(client, *, ip: str, forwarded: str | None = None):
    """One walk generation, from a chosen client."""
    headers = {"X-Forwarded-For": forwarded or ip}
    return client.post("/api/walk", json=BODY, headers=headers)


# ---------------------------------------------------------------------------
# Per-IP walk limit
# ---------------------------------------------------------------------------


def test_sixth_walk_from_one_ip_is_refused() -> None:
    """Five per hour means the sixth is a 429, not a sixth walk."""
    container = build_container(
        llm=ScriptedLLM(GROUNDED), tts=StubTTS(), guardrails=build_guardrails(walk_limit=5)
    )

    with client_for(container) as client:
        for _ in range(5):
            assert _post_walk(client, ip=CLIENT_IP).status_code == 200
        sixth = _post_walk(client, ip=CLIENT_IP)

    assert sixth.status_code == 429
    assert sixth.json()["detail"] == "walk_limit_reached"
    assert int(sixth.headers["retry-after"]) > 0
    assert sixth.headers["x-ratelimit-limit"] == "5"
    assert sixth.headers["x-ratelimit-remaining"] == "0"
    # A refusal still states the demo flag, so the client can tell "refused" from
    # "a proxy ate the header".
    assert sixth.headers[DEMO_HEADER] == "False"


def test_the_allowance_is_per_client_not_global() -> None:
    container = build_container(
        llm=ScriptedLLM(GROUNDED), tts=StubTTS(), guardrails=build_guardrails(walk_limit=1)
    )

    with client_for(container) as client:
        assert _post_walk(client, ip=CLIENT_IP).status_code == 200
        assert _post_walk(client, ip=CLIENT_IP).status_code == 429
        # A different walker is unaffected by the first one's exhaustion.
        assert _post_walk(client, ip=OTHER_IP).status_code == 200


def test_a_forged_forwarded_prefix_does_not_buy_a_new_allowance() -> None:
    """The client is the entry the trusted hop appended, not the one a caller made up.

    With one trusted proxy the real address is last, so prepending a fresh bogus
    address to every request lands in the *same* bucket instead of a new one.
    """
    container = build_container(
        llm=ScriptedLLM(GROUNDED), tts=StubTTS(), guardrails=build_guardrails(walk_limit=2)
    )

    with client_for(container) as client:
        assert (
            _post_walk(
                client, ip=CLIENT_IP, forwarded=f"1.1.1.1, {CLIENT_IP}"
            ).status_code
            == 200
        )
        assert (
            _post_walk(
                client, ip=CLIENT_IP, forwarded=f"2.2.2.2, {CLIENT_IP}"
            ).status_code
            == 200
        )
        forged = _post_walk(client, ip=CLIENT_IP, forwarded=f"3.3.3.3, {CLIENT_IP}")

    assert forged.status_code == 429


def test_a_refusal_does_not_consume_the_allowance() -> None:
    """The limiter must not count the requests it rejected, or it can never free up."""
    container = build_container(
        llm=ScriptedLLM(GROUNDED), tts=StubTTS(), guardrails=build_guardrails(walk_limit=1)
    )
    limiter = container.guardrails.limiter

    with client_for(container) as client:
        assert _post_walk(client, ip=CLIENT_IP).status_code == 200
        assert limiter.peek(CLIENT_IP) == 1
        for _ in range(3):
            assert _post_walk(client, ip=CLIENT_IP).status_code == 429
        assert limiter.peek(CLIENT_IP) == 1


def test_the_window_slides_rather_than_resetting_on_the_hour() -> None:
    """A fixed bucket lets a client spend two allowances either side of the hour."""
    clock = FakeClock()
    limiter = PerIPWalkLimiter(limit=2, window_s=3600.0, clock=clock)

    assert limiter.consume("a").allowed
    clock.advance(100)
    assert limiter.consume("a").allowed

    assert not limiter.check("a").allowed

    # 3_500 later the first hit is 3_600 old and ages out, while the second is
    # only 3_500 old and does not. One slot frees -- not both at once, which is
    # the failure a fixed hourly bucket is prone to.
    clock.advance(3_500)
    assert limiter.check("a").allowed
    assert limiter.consume("a").allowed
    assert not limiter.check("a").allowed


def test_with_no_trusted_proxy_the_forwarded_header_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directly exposed, a forwarded header is just something a client made up."""
    monkeypatch.setattr("app.api.limits.TRUSTED_PROXY_HOPS", 0)
    container = build_container(
        llm=ScriptedLLM(GROUNDED), tts=StubTTS(), guardrails=build_guardrails(walk_limit=1)
    )

    with client_for(container) as client:
        # Same socket, two fabricated addresses: one bucket, so the second is
        # refused. Trusting the header here would hand out unlimited walks.
        assert _post_walk(client, ip="1.1.1.1").status_code == 200
        assert _post_walk(client, ip="2.2.2.2").status_code == 429


# ---------------------------------------------------------------------------
# Daily spend cap
# ---------------------------------------------------------------------------


def test_a_live_walk_records_what_it_spent() -> None:
    """Spend accrues against the real units: characters sent, per path."""
    guardrails = build_guardrails(cap_usd=100.0)
    container = build_container(
        llm=ScriptedLLM(GROUNDED), tts=StubTTS(), guardrails=guardrails
    )

    with client_for(container) as client:
        assert _post_walk(client, ip=CLIENT_IP).status_code == 200

    snapshot = guardrails.ledger.snapshot()
    assert snapshot["tts_chars"] > 0
    assert snapshot["llm_chars"] > 0
    assert snapshot["spent_usd"] > 0
    assert snapshot["walks"] == 1
    assert snapshot["exhausted"] is False


def test_the_ledger_rolls_over_at_the_utc_day_boundary() -> None:
    """A daily cap that never resets is a one-off cap."""
    day = FakeDay()
    ledger = SpendLedger(cap_usd=1.0, prices=Prices(configured=True), clock=day)
    ledger.record_tts(10_000)

    assert ledger.exhausted is True
    assert ledger.spent_usd > 1.0

    day.advance()
    assert ledger.exhausted is False
    assert ledger.spent_usd == 0.0
    assert ledger.snapshot()["day"] == day.day.isoformat()


def test_a_zero_cap_disables_the_ceiling() -> None:
    """Zero means "no cap", not "refuse everything" -- the safe reading for a dev box."""
    ledger = SpendLedger(cap_usd=0.0, prices=Prices(configured=True))
    ledger.record_tts(1_000_000)
    assert ledger.exhausted is False


def test_exhausting_the_cap_serves_the_demo_without_calling_anyone() -> None:
    """The kill-switch, and the assertion that matters: nothing was spent."""
    tts = StubTTS()
    llm = ScriptedLLM(GROUNDED)
    guardrails = build_guardrails(cap_usd=1.0)
    guardrails.ledger.record_tts(10_000)  # spend past the cap
    container = build_container(llm=llm, tts=tts, guardrails=guardrails)

    with client_for(container) as client:
        resp = _post_walk(client, ip=CLIENT_IP)
        frames = ndjson(resp)

    assert resp.status_code == 200
    assert resp.headers[DEMO_HEADER] == "True"
    assert resp.headers["x-demo-reason"] == "spend_cap"
    assert [f["type"] for f in frames][-1] == "indicator"
    assert frames[-1]["status"] == "walk_complete"
    assert all(f.get("demo") is True for f in frames)
    # The whole point: no ElevenLabs, no model.
    assert tts.spoken == []
    assert llm.calls == []


def test_the_demo_is_labelled_and_its_citations_resolve() -> None:
    """A recorded walk must not read as a generated one, and must still be coherent."""
    container = build_container(
        llm=ScriptedLLM(GROUNDED),
        tts=StubTTS(),
        guardrails=build_guardrails(force_demo=True),
    )

    with client_for(container) as client:
        frames = ndjson(_post_walk(client, ip=CLIENT_IP))

    assert all(f.get("demo") is True for f in frames)
    sentences = [
        f for f in frames if f["type"] in {"text", "sentence"} and f.get("role") != "reminder"
    ]
    assert len(sentences) == len(DEMO_SEGMENTS)

    fact_ids = {f.fact_id for f in DEMO_FACTS}
    for sentence in sentences:
        # Every demo sentence cites a demo fact that exists. The validator's
        # contract holds in the demo too, or the demo is teaching the wrong rules.
        assert sentence["citations"], sentence
        assert set(sentence["citations"]) <= fact_ids
        assert sentence["source_urls"]

    done = frames[-1]
    assert done["mode"] == "STORY"
    assert done["used_static_fallback"] is True
    # A session over the demo facts, so a question asked mid-demo can be served.
    assert done["session_id"]
    assert container.sessions.get(done["session_id"]) is not None


def test_the_demo_reports_no_quota_trouble_on_a_live_walk() -> None:
    """The happy path states the flag explicitly rather than omitting it."""
    container = build_container(
        llm=ScriptedLLM(GROUNDED), tts=StubTTS(), guardrails=build_guardrails(walk_limit=3)
    )

    with client_for(container) as client:
        resp = _post_walk(client, ip=CLIENT_IP)

    assert resp.status_code == 200
    assert resp.headers[DEMO_HEADER] == "False"
    assert "x-demo-reason" not in resp.headers
    assert resp.headers["x-ratelimit-remaining"] == "2"


# ---------------------------------------------------------------------------
# The prerecorded assets have to actually be there
# ---------------------------------------------------------------------------


def test_every_demo_clip_is_on_disk_and_served() -> None:
    """A kill-switch that fires into 404s is worse than no kill-switch."""
    assert missing_clips() == ()

    container = build_container(
        llm=ScriptedLLM(GROUNDED),
        tts=StubTTS(),
        guardrails=build_guardrails(force_demo=True),
    )

    with client_for(container) as client:
        frames = ndjson(_post_walk(client, ip=CLIENT_IP))
        urls = [f["audio_url"] for f in frames if f.get("audio_url")]
        assert urls, "demo frames must point at audio"

        for url in urls:
            asset = client.get(url)
            assert asset.status_code == 200, url
            assert len(asset.content) > 0, url
            assert asset.headers["content-type"].startswith("audio/"), url


# ---------------------------------------------------------------------------
# Push-to-talk under the cap
# ---------------------------------------------------------------------------


def test_demo_mode_answers_push_to_talk_without_calling_a_provider() -> None:
    """PTT is a paid path twice over, so the cap has to stop it too."""
    stt = StubSTT()
    llm = ScriptedLLM(GROUNDED)
    container = build_container(
        llm=llm, stt=stt, tts=StubTTS(), guardrails=build_guardrails(force_demo=True)
    )

    with client_for(container) as client:
        resp = client.post(
            "/api/qa",
            files={"audio": ("question.m4a", b"fake-clip", "audio/mp4")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["demo"] is True
    assert body["use_native_tts"] is True
    assert resp.headers[DEMO_HEADER] == "True"
    assert "demonstration" in body["answer"]
    # Groq and the model were never asked.
    assert stt.uploads == []
    assert llm.calls == []


def test_push_to_talk_does_not_spend_the_walk_allowance() -> None:
    """Asking five questions is not starting five walks."""
    container = build_container(
        llm=ScriptedLLM(GROUNDED),
        stt=StubSTT(),
        tts=StubTTS(),
        guardrails=build_guardrails(walk_limit=5),
    )
    limiter = container.guardrails.limiter

    with client_for(container) as client:
        for _ in range(5):
            client.post(
                "/api/qa",
                files={"audio": ("q.m4a", b"clip", "audio/mp4")},
                headers={"X-Forwarded-For": CLIENT_IP},
            )
        # The allowance is untouched, so a walk is still available.
        assert limiter.peek(CLIENT_IP) == 0
        assert _post_walk(client, ip=CLIENT_IP).status_code == 200


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_reports_the_budget_and_the_switch() -> None:
    guardrails = build_guardrails(cap_usd=2.0, walk_limit=5)
    guardrails.ledger.record_tts(1_000)
    container = build_container(
        llm=ScriptedLLM(GROUNDED), tts=StubTTS(), guardrails=guardrails
    )

    with client_for(container) as client:
        health = client.get("/api/health").json()

    assert health["walk_limit_per_hour"] == 5
    assert health["daily_spend_cap_usd"] == 2.0
    # Process-local, stated for the same reason the session store is.
    assert health["guardrails"] == "process_local"
    assert health["spend"]["exhausted"] is False
    assert health["spend"]["tts_chars"] == 1_000
    assert health["demo"]["active"] is False
    assert health["demo"]["reason"] is None
    # The placeholders are declared as placeholders.
    assert health["demo"]["clips_recorded"] is False
    assert health["demo"]["clips_missing"] == []


def test_health_says_why_the_switch_fired() -> None:
    container = build_container(
        llm=ScriptedLLM(GROUNDED),
        tts=StubTTS(),
        guardrails=build_guardrails(force_demo=True),
    )

    with client_for(container) as client:
        health = client.get("/api/health").json()

    assert health["demo"]["active"] is True
    assert health["demo"]["reason"] == "demo_forced"
    assert health["demo"]["forced"] is True


def test_a_forced_switch_outranks_a_healthy_budget() -> None:
    """The manual kill-switch is for a demo day or an incident, not only for spend."""
    guardrails = Guardrails(
        limiter=PerIPWalkLimiter(limit=5, window_s=3600.0),
        ledger=SpendLedger(cap_usd=100.0, prices=Prices(configured=True)),
        force_demo=True,
    )

    admission = guardrails.admit(CLIENT_IP)
    assert admission.allowed is True
    assert admission.demo is True
    assert admission.reason == "demo_forced"
