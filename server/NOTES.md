# Backend Notes

Decisions and live-API findings behind the Echoes backend. Each entry was
verified by calling the real endpoint, not assumed from documentation.

## 0. Layout

The backend is an `app/` package with three layers, so no file has to be read
whole to find one behaviour:

| Path                  | Contains                                                            |
| --------------------- | ------------------------------------------------------------------- |
| `main.py`             | ASGI entrypoint (`uvicorn main:app`). A shim, ~30 lines.            |
| `app/api/`            | One router per concern: context, health, walk, audio, qa, headers.  |
| `app/services/`       | Behaviour: cataloged below.                                         |
| `app/validation/`     | Citations, sentence assembly, rule sets. The factual moat.          |
| `app/models.py`       | `FactObject` and the request/response schemas.                      |
| `app/config.py`       | Every `os.getenv`. Nothing else reads the environment.              |
| `app/dependencies.py` | The typed `Container` and the FastAPI dependency that serves it.    |
| `tests/`              | Test suite; no network or API key required.                         |

`app/services/` is further divided: `context.py` (aggregation policy),
`wikipedia.py` / `parks.py` (sources), `denylist.py`, `geo.py`, `sessions.py`,
`narration.py`, `qa.py`, `prompts.py`, `providers.py` (the LLM seam),
`gemma.py`, and the speech group `tts.py` / `tts_queue.py` / `audio_tokens.py` /
`stt.py`. `factories.py` is the only place that decides whether a provider is
configured.

Handlers depend on the container rather than reaching into `app.state`, which is
what lets a test swap one object instead of patching module globals.

## 1. Wikipedia no longer accepts `gsrlat` / `gsrlon`

The Part 1 design used `list=geosearch&gsrlat=…&gsrlon=…`. Against the live
MediaWiki (1.47.0-wmf.23) that returns HTTP 200 with an error body:

```json
{"error":{"code":"missingparam",
          "info":"One of the parameters \"gscoord\", \"gspage\" and \"gsbbox\" is required."}}
```

Coordinates are now passed as a single tuple, and the generator form uses a
`ggs` prefix:

| Old (broken)            | Current                |
| ----------------------- | ---------------------- |
| `gsrlat` + `gsrlon`     | `gscoord=lat\|lon`     |
| `gsrradius`             | `gsradius`             |
| `gsrlimit`              | `gslimit`              |
| `ggsrlat` + `ggsrlon`   | `ggscoord=lat\|lon`    |

Because this fails *silently* with a 200, a naive client would treat every
lookup as "no facts" and pin the user in NOTICE mode forever.
`services/wikipedia.py` now detects `payload["error"]` explicitly and logs it.

## 2. A descriptive `User-Agent` is mandatory

Requests without one are rejected with **HTTP 403** by Wikimedia. The client
sends `ECHOES_USER_AGENT` (overridable via env) on every request.

## 3. One round-trip, not two

`generator=geosearch` combined with `prop=extracts|coordinates&exintro=1`
returns the title, intro extract **and** coordinates in a single response.
`list=geosearch` returns a distance but no text, and fetching extracts
afterwards would serialise two network calls and blow the latency budget
(PRD §5: TTFA < 2.5s). Distance is therefore computed locally with a
haversine helper from the returned coordinates.

## 4. Deny-list reconciliation (spec conflict)

The two governing documents disagree:

- Part 1 spec: `metro`, `station`, `constituency`, `district`, `wholesale`, `cutting`
- AGENTS.md §4: `metro`, `station`, `constituency`, `district`, `assembly`

`DENY_LIST` is the **union** of both. Over-filtering costs at most one fact and
falls back to the safe NOTICE path; under-filtering leaks infrastructure or
commercial results into narrated audio, which AGENTS.md treats as a hard
failure. `assembly` and `wholesale`/`cutting` are consequently all active.

## 5. `prop=coordinates` is not a reliable source of distance

Reading distance from `prop=coordinates` looks correct and silently fails.
Measured live at Trafalgar Square, **10 of 20** geosearch hits returned no
`coordinates` at all -- geosearch matches articles whose coordinates live on
**Wikidata**, while `prop=coordinates` only returns locally-stored ones. The
fallback of `0.0` then ranked those distant articles as the *closest*, so the
first facts in the response were the ones furthest away.

Fix: distance now comes from `list=geosearch`, which returns a `dist` for every
hit, requested alongside `generator=geosearch&prop=extracts` in the same
round-trip. Verified: 20/20 distances populated, range 10.7-98.0m, no zeros.
When a distance genuinely cannot be determined the page is dropped and logged
rather than assigned `0.0`.

## 6. Failures degrade to NOTICE, never to HTTP 500

Any `httpx.TimeoutException`, `httpx.HTTPError`, or unusable payload is caught
per-source, logged, and treated as zero facts for that source. AGENTS.md §5
makes NOTICE the deterministic consequence of having no qualifying facts, so an
upstream outage produces a valid mindfulness session rather than a broken walk.
A partial outage still serves whatever the healthy source returned.

**Caveat:** this means a total upstream outage is indistinguishable from a
genuinely fact-free location at the API boundary. Both are logged at WARNING
with the source name, so alerting should key off the logs, not the response.

## 7. Two defects found while writing the tests

Both were pre-existing and both were invisible without a test that read the
value back rather than trusting the code that wrote it.

### `X-Answer-Sources` was percent-encoded twice

The PTT metadata headers encoded each source URL, joined them, and then passed
the joined string through the same encoder again. `%27` became `%2527`, so a
client that decoded once got `...Nelson%2527s_Column` back, not a usable URL.
Fixed by encoding every value **exactly once** in `api/headers.py`.

### `synthesize_incremental` swallowed a source failure

The feeder task published its sentinel from a `finally` block, so an exception
raised by the text source was stored on a task nobody awaited. The effect was
silent truncation: the walker heard fewer sentences than the model produced,
with nothing logged. The consumer now awaits the feeder after the source is
exhausted, which propagates the failure (and still delivers the sentences
dispatched before it).

## 8. Cost guardrails: a limit that cannot be spoofed, a cap that degrades

AGENTS.md §4 asks for three things: a per-IP walk limit, a global daily spend
cap over ElevenLabs and the LLM, and a kill-switch to a labelled recorded demo.
`services/budgets.py` decides, `api/limits.py` turns the decision into headers,
and `services/demo.py` is what gets served when the budget is gone.

### The identity is the hard part

Behind a proxy the socket address is the proxy, so the limit has to key off
`X-Forwarded-For`. That header is client-writable, so trusting it hands out
unlimited walks: send a different address each time and the limiter never trips.
`ECHOES_TRUSTED_PROXY_HOPS` is therefore hop-counted -- the client is that many
entries from the *right*, so a forged prefix lands in the same bucket as the
real address the trusted hop appended. Verified against a live server: five
requests with rotating forged prefixes (`1.1.1.1, <real>`, `2.2.2.2, <real>`, …)
all counted against `<real>`, and the sixth returned **429** with
`Retry-After: 3592`. With `ECHOES_TRUSTED_PROXY_HOPS=0` the header is ignored
entirely and the socket is used, which is the right setting only when nothing
sits in front of the app.

### The window slides

A fixed hourly bucket lets a client spend one allowance at 12:59 and the next at
13:00, which is the exact burst a limit exists to stop. The limiter is a deque of
hits, trimmed from the oldest, so slots free one at a time.

### The cap degrades instead of failing

Crossing the cap does not 500 a request: `/api/walk` streams the recorded demo
and `/api/qa` answers with a recorded line. The assertion that matters is that
**nothing is spent doing it** -- asserted in `tests/test_limits.py` by handing
the container a `StubTTS` and a `ScriptedLLM` and checking both stayed empty.

The unit prices are environment inputs, never quotations, and `/api/health`
reports `spend.prices_configured`. A cap computed from invented numbers is not a
budget, so the flag exists rather than a silent placeholder.

Measured on a live server: a demo walk returned three `sentence` frames, a
`reminder`, then `done` -- all carrying `demo: true`, citing the demo facts
(`a1c4f2`, `7d90be`, `c33a18`) -- with `X-Demo-Mode: True` and
`X-Demo-Reason: demo_forced` on the response, and `spend.spent_usd` still `0.0`.

### What is process-local, and what that costs

The limiter, the ledger and the session store are all per-process. One instance
is correct; two mean two allowances and **two daily budgets against one cap**.
`render.yaml` therefore pins `numInstances: 1`, and `/api/health` reports each
store as `process_local` rather than letting the operator discover it. On a
free Render plan the instance also spins down when idle, which resets the day's
spend -- so the cap is a guard against a runaway loop, not a hard billing
boundary. A Redis-backed `PerIPWalkLimiter`/`SpendLedger` is a new class, not a
rewrite: both are injected into `Guardrails`.

### Three defects found while writing these tests

1. **My demo helper bound the route.** `@router.post("/api/qa")` ended up
decorating `_demo_answer` rather than the handler, so `/api/qa` was undecorated
and every push-to-talk request 422'd on `started` and `Admission` being read as
query and body parameters. Twenty-five pre-existing tests caught it.
2. **`X-RateLimit-Remaining` was off by one.** It was built from the admission
decision taken *before* the walk was charged, so the first walk of five
advertised four remaining instead of three. The headers are now built from the
post-charge decision.
3. **`/api/health` reported configuration, not enforcement.** It echoed the
module constants, so it would have lied about the cap whenever the process was
configured differently -- exactly the case an operator checks it for. It now
reads the live limiter and ledger.

Not covered end to end: the `spend_cap` reason for the switch needs real paid
calls to accrue spend, and no provider keys are configured locally. That path is
covered by a unit test that injects spend directly; the demo path itself
(`demo_forced`) was exercised against a live server.

## 9. `/static` was promised and never served

`api/walk.py` emitted `"audio_url": "/static/reminder.mp3"` for the mandatory
pedestrian reminder, and nothing mounted `/static`, and no such file existed.
The frame therefore handed the client a guaranteed 404 -- which is worse than a
missing reminder, because the client cannot tell a 404 from silence mid-walk.

`server/static` now exists and is mounted in `create_app`, the reminder is
`reminder.wav`, and the demo clips live beside it. The placeholders are
generated rather than checked in blind: `scripts/make_demo_placeholders.py`
writes valid silent WAVs sized to the script they stand in for, and a manifest
that says `"recorded": false`. `/api/health` reports that flag and lists any
clip that is missing, so a demo deployment cannot quietly ship silence believing
it shipped a recording. Verified: all four clips return `200` with
`content-type: audio/wav` and real byte counts (190900 / 136044 / 138328 /
36614), and `tests/test_limits.py` walks every `audio_url` the demo frames
promise and asserts each one serves audio.
