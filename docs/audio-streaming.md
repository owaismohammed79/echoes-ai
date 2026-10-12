# Audio Streaming & PTT Pipeline — Design

**Status:** design only, not implemented. Shapes, protocols and trade-offs; no
production code.

**Governed by:** [AGENTS.md](../AGENTS.md) §3 (iOS audio unlock, minimal screen
time), §5 (fallbacks, kill-switch); [PRD.md](../PRD.md) §2 (screen-locked
playback), §5 (latency, privacy).

**Depends on:** [llm-orchestration.md](llm-orchestration.md) §7 (sentence
assembly and validation) and the `FactObject[]` contract from
[`/api/context`](../server/app/api/context.py).

---

## 1. Scope

In scope: getting _validated_ text into the walker's ears with the lowest
perceivable delay, and answering spoken questions.

Out of scope: validation itself (built), the `/api/context` fetch (built), the
static demo kill-switch (§9 notes where it plugs in).

---

## 2. Where this sits, and the one rule that gates it

```
Gemma 4 token stream
   │
   ▼
SentenceAssembler          (llm-orchestration.md §7.1 — already built)
   │  complete sentences
   ▼
Validator                  (llm-orchestration.md §7.4 / §7.5 — already built)
   │  ValidatedSentence{text, citations, source_urls}
   ▼
TTS dispatcher             ◄── this document starts here
   │
   ├── Channel A: narration metadata frames   →  client text/citations
   └── Channel B: per-sentence audio URLs     →  client audio queue
```

**The gating rule: the dispatcher only ever receives sentences the validator
already cleared.** Priority is "no unvalidated sentence reaches TTS"
(llm-orchestration.md §7), not "fastest possible first byte". This costs
essentially nothing today — validation is a pure in-process check measured in
single-digit milliseconds — but it is the first thing a latency-obsessed
optimisation would try to remove, so it is stated as a constraint rather than a
preference.

Two consequences worth stating up front:

- Audio is synthesised from `ValidatedSentence.text`, which is already
  tag-stripped. A `[cite:...]` tag must never be sent to TTS; it would be
  narrated as literal punctuation.
- Sentence _punctuation is retained_ (`"...rises over the square."`), because
  ElevenLabs uses terminal punctuation for prosody. Only tags are removed.

---

## 3. Latency budget

PRD §5 requires time-to-first-audio under 2.5s. Measured so far:
`/api/context` ≈1.0s (live, Trafalgar Square).

| Stage                         | Budget    | Basis                             |
| ----------------------------- | --------- | --------------------------------- |
| `/api/context`                | ~1.0s     | measured                          |
| LLM time-to-first-token       | ~0.4s     | target                            |
| First sentence completes      | ~0.3s     | 2-3 sentence response             |
| Validation                    | <5ms      | pure, in-process                  |
| ElevenLabs sentence inference | ~0.1-0.3s | ~75ms model inference + network   |
| First audio bytes at client   | ~0.2s     | mobile RTT + progressive download |
| **Total**                     | **~2.2s** | ~0.3s headroom                    |

This budget assumes `eleven_flash_v2_5` and is contingent on §4's playback
model. It is a projection, not a measurement: nothing here has been timed
against the real ElevenLabs service.

---

## 4. The iOS constraint that decides the transport

There is an obvious "lowest latency" architecture: stream raw PCM from
ElevenLabs, decode nothing, and schedule buffers back-to-back on a Web Audio
`AudioContext` for gapless playback. It is wrong for this product, for two
platform reasons.

**1. Web Audio does not survive the lock screen.** WebKit bug 231105 documents
that on iOS `AudioContext` is suspended when the app backgrounds, "whether
playing audio or not". Related platform reporting is blunter: iOS Safari does
not let Web Audio bypass the silent switch or continue on the lock screen.
Echoes' entire premise is that the walker starts a walk and then does not look
at the screen — the audio must keep playing with the phone pocketed and locked.

**2. Media Source Extensions are not available for audio on iOS.** MSE is
unavailable for audio playback in Safari, so the usual "append MP3 chunks to a
`SourceBuffer`" streaming trick is not available either. `ManagedMediaSource`
addresses video, not this.

Together those rule out every chunk-level streaming playback path on iOS. The
only thing that keeps playing with the screen off is a **media element**:
`HTMLAudioElement` (`<audio>`) with Media Session metadata.

### Consequence: audio is delivered per sentence, not per chunk

```
sentence text ──> ElevenLabs ──> complete-ish audio for ONE sentence ──> <audio>
```

Because `<audio>` plays a _resource_, each sentence is served as its own
progressive-download URL (§6, Channel B). The browser begins playback as bytes
arrive and continues in the background — so this is **not** the same as
buffering the whole clip. Progressive download gets most of the streaming
benefit while staying lock-screen-safe.

### The unlock requirement (AGENTS.md §3)

AGENTS.md §3 requires `<audio>` **and** `AudioContext` to be unlocked
_synchronously inside the initial touch handler, before firing any async
request_. The design keeps both:

- `<audio>` — the real playback path (§8). Unlocked by calling `play()` on a
  silent/zero-length source inside the `Start Walk` handler.
- `AudioContext` — not used for narration playback, but a resumed context is
  still needed for any UI feedback (PTT start/stop cue), and unlocking it
  synchronously is the only reliable window.

Ordering matters and is easy to get wrong: if the unlock is awaited on a
network call first, iOS discards the user-gesture context and playback is
blocked for the whole walk. Unlock first, _then_ fetch.

---

## 5. Sentence dispatch workflow

1. **Sentence boundary** — the existing `SentenceAssembler` already splits at
   `.`, `!`, `?` (and `…`), and already handles the two cases that break naive
   splitters: abbreviations/decimals, and a citation tag sitting directly after
   the terminator. This document consumes its output; it does not re-split.
2. **Validate** — `validate_story_sentence` / `validate_notice_sentence`.
3. **Dispatch immediately** — each validated sentence is sent to ElevenLabs as
   soon as it exists. No batching, no waiting for the full narration. This is
   the whole point of the design: sentence 1 is synthesised while Gemma 4 is
   still writing sentence 2.
4. **Bounded concurrency** — at most 2 synthesis requests in flight per walk
   (the sentence playing, plus one prefetching). See §7 for why the cap exists.
5. **Ordered emission** — frames are emitted in sentence order even though
   synthesis may complete out of order, so a fast sentence 2 never overtakes
   sentence 1. A small reorder buffer holds out-of-order completions.

---

## 6. Channel design: text and audio travel separately

Sending audio inline with the text stream is the tempting shortcut, and it is
the wrong one:

| Option                                        | Verdict                                                                                                   |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| SSE with base64 audio                         | Rejected. ~33% inflation plus encode/decode CPU, for bytes that are not text.                             |
| Length-prefixed binary frames in one response | Works, but forces a custom parser and couples audio delivery to the narration request.                    |
| **NDJSON text stream + separate audio URLs**  | **Chosen.** Text stays tiny and readable; audio gets its own cacheable, prefetchable, abortable resource. |

### Channel A — narration metadata (NDJSON)

`POST /api/walk` responds `application/x-ndjson`, one JSON object per line, as
each sentence clears validation:

```json
{"type":"sentence","index":0,"text":"Nelson's Column rises over the square.","citations":["9c1f2ab73e0d5144"],"source_urls":["https://en.wikipedia.org/wiki/Nelson%27s_Column"],"audio_url":"/api/audio/a1b2c3","use_native_tts":false,"mode":"STORY"}
{"type":"sentence","index":1,"text":"Charing Cross sits a short walk away.","citations":["4be7c0a91d2f8836"],"source_urls":["https://en.wikipedia.org/wiki/Charing_Cross"],"audio_url":"/api/audio/d4e5f6","use_native_tts":false,"mode":"STORY"}
{"type":"reminder","text":"Stay on marked pedestrian paths.","audio_url":"/static/reminder.mp3"}
{"type":"done","mode":"STORY","degraded":false}
```

Why NDJSON over SSE: the payload is strictly server→client, one-way, and the
client already reads it with `fetch` + a stream reader. SSE would work but adds
`EventSource` semantics (text-only, auto-reconnect) this endpoint does not want,
since a reconnect mid-narration should restart the walk step, not resume it.

The reminder is a **frame with a static audio URL**, not part of the narration
request — pre-rendered, cacheable, and identical on every response (§7).

### Channel B — audio

`GET /api/audio/{token}` returns the sentence audio, streaming the ElevenLabs
body through as it arrives (`audio/mpeg`). Properties this buys:

- **Progressive playback** — `<audio>` starts on partial bytes.
- **Prefetchable** — the client requests sentence _n+1_ while sentence _n_ plays.
- **Cacheable** — a CDN or the service worker can hold repeated lines (the
  fixed NOTICE copy and the reminder recur constantly).
- **Independently abortable** — a barge-in cancels audio without touching the
  text stream.

### Security: the relay must not be an open TTS proxy

`GET /api/audio/{token}` spends real money per character. If the token encodes
the text — or if the endpoint accepts arbitrary text — anyone can use Echoes as
a free text-to-speech service and drain the quota that AGENTS.md §5 exists to
protect.

Requirements:

- The token is an **opaque server-side key**, never the text.
- Keys are **single-use, short-TTL** (the sentence text is held server-side for
  ~2 minutes, then dropped), and **bound to the session** so one walker cannot
  consume another's synthesised sentence.
- Return `404` for unknown/expired tokens — never re-synthesise on demand from
  a caller-supplied string.
- Cap total synthesis characters per session and per IP, as a second line of
  defence behind the kill-switch.

---

## 7. ElevenLabs integration

### Transfer mode

One endpoint (`POST /v1/text-to-speech/{voice_id}`) in three shapes:

| Mode                                    | Time-to-first-audio        | Complexity                              | Text known up front               | Concurrency accounting            |
| --------------------------------------- | -------------------------- | --------------------------------------- | --------------------------------- | --------------------------------- |
| Batch `.../{voice_id}`                  | Highest (whole clip first) | Lowest                                  | Required                          | Each request counts fully         |
| **HTTP stream `.../{voice_id}/stream`** | **Low**                    | **Low**                                 | **Required**                      | **Each request counts fully**     |
| WebSocket `.../stream-input`            | Lowest                     | Highest (connection lifecycle, framing) | Not required — send incrementally | **Only active generation counts** |

**Chosen: HTTP streaming, one request per validated sentence.** The text of
each sentence is fully known when it is dispatched (the validator produced it),
which is precisely the case HTTP streaming is documented for. The WebSocket
shines when text arrives incrementally and when an idle-but-open connection must
not consume concurrency — neither applies, because our unit is a complete
sentence.

**The concurrency accounting is the reason to revisit later.** Over HTTP, every
in-flight request occupies a concurrency slot for its full duration; over
WebSocket, only active generation counts. With 2 concurrent syntheses per
walker, 10 concurrent walkers consume 20 HTTP slots. If the plan's concurrency
limit becomes the binding constraint, migrating to `stream-input` is the
fix — but it introduces connection lifecycle and framing complexity that buys
nothing at demo scale. Note the limit before scaling, don't pre-optimise for it.

### Model

`eleven_flash_v2_5` — ~75ms model inference, 32 languages, 40,000-character
limit. Documented as the lowest-latency multilingual option and recommended for
real-time paths.

The tension: it is also described as _not_ the narration model.
`eleven_multilingual_v2` is the "highest stable fidelity, narration" choice.
Echoes narrates, but on a 2-3 sentence budget under a 2.5s TTFA requirement, so
**latency wins by default** and the model id is configuration, not a constant.
If walkers report the voice sounds thin, the experiment is `multilingual_v2`
with a measured TTFA delta — not a guess.

### Output format

MP3 (`audio/mpeg`) rather than PCM: it is what `<audio>` plays progressively and
what iOS handles without fuss, at a fraction of PCM's bandwidth. Request a
speech-appropriate bitrate (~64kbps mono is ample for voice).

### Static copy is pre-rendered, never synthesised

The mandatory reminder (`Stay on marked pedestrian paths.`) is appended to
**every** response by AGENTS.md §2, and `STATIC_NOTICE_LINES` are fixed
constants. Both are rendered once at build time and served as static files.

This is worth doing for three independent reasons: it removes latency from
every response, it stops paying repeatedly for identical text, and it means the
last-resort fallback path has _no dependency on ElevenLabs at all_ — which is
exactly what a fallback should be.

### Failure handling

- **429 / 5xx** → exponential backoff with full jitter (the documented
  recommendation), then §9's native-TTS fallback rather than retrying into a
  wall.
- **Timeout** → per-request ceiling (~3s, matching `/api/context`), then
  fallback.
- **Concurrency saturation** → a global semaphore sized just below the plan
  limit; when full, prefer falling back to native TTS over queueing, because a
  queued narration is a narration the walker has already walked past.
- **Circuit breaker** — after N consecutive failures, open the breaker for
  ~60s and serve `use_native_tts: true` proactively (§9), so a struggling
  provider is not hammered by every walker at once.

---

## 8. Client playback queue

```
frame arrives ──> create <audio src="/api/audio/{token}"> (prefetch)
                  │
sentence n ends ──> play element for sentence n+1
```

- **One reused `<audio>` element** per walk, `src` swapped per sentence. Fewer
  elements, one Media Session, one unlock.
- **Prefetch one sentence ahead** so the next clip is buffered before the
  current one ends.
- **Accept a small inter-sentence gap.** Swapping `src` is not gapless; there
  will be a short silence between sentences. This is the honest cost of
  choosing background-safe playback. Mitigation: preload the next clip and
  swap a few hundred milliseconds before the end, and keep sentences short
  (2-3 per response) so the gaps are few. Gapless playback would require Web
  Audio scheduling, which §4 rules out.
- **Media Session metadata** — title from the first sentence's leading fact
  title, artist "Echoes" — so the lock screen shows something meaningful and
  the hardware controls work (PRD §2).
- **Source links panel** is fed from Channel A's `citations`/`source_urls`, not
  from audio. Grounding is visible without any screen time during the walk.

---

## 9. Fallback: `use_native_tts`

Requirement: if ElevenLabs rate-limits or fails, return the raw text with
`use_native_tts: true` so the client uses `window.speechSynthesis`.

### Two trigger paths

**Proactive — the server already knows.** The circuit breaker is open, or the
global quota/kill-switch condition (AGENTS.md §5) is tripped. The narration
frame is emitted with `use_native_tts: true` and **no `audio_url`**, so the
client never issues a doomed request. This is the path that saves the most
latency: it skips a round trip that was going to fail.

**Reactive — the request failed anyway** (429 mid-narration, timeout, 5xx,
network drop). The audio request fails _after_ the client already had the text.

### The flag is mostly informational, and that matters

Because Channel A delivers the sentence text before any audio is requested, the
client **already holds the text** when an audio fetch fails. It does not need
the server to send the text back — `use_native_tts` is a signal, and the
fallback action is entirely client-side:

```
audio request fails, OR frame.use_native_tts is true
   └─> window.speechSynthesis.speak(new SpeechSynthesisUtterance(frame.text))
```

This is why the design does not put text in the audio response: it would be a
second copy of data the client already has, and an inconsistent copy at that.

### Mid-stream flip, and the voice-consistency trade-off

A failure can occur after sentence 1 has already been spoken in the ElevenLabs
voice. Sentences already played cannot be un-played, so the response flips:
sentence 1 in the ElevenLabs voice, sentences 2+ in the system voice. That is
audibly jarring.

Options, exposed as configuration:

- **Seamless-as-possible (default)** — audio already delivered plays as audio;
  remaining sentences go native. Least disruption to the walk.
- **Voice-consistent** — if any sentence in the response fails, stop audio
  playback and re-speak the whole narration natively. Consistent voice, but it
  repeats content the walker just heard. Only sensible if voice switching is
  judged worse than repetition.

Documented rather than silently chosen: this is a product call.

### Response shape

```json
{
  "use_native_tts": true,
  "reason": "elevenlabs_rate_limited",
  "sentences": [
    {
      "index": 0,
      "text": "Nelson's Column rises over the square.",
      "citations": ["9c1f2ab73e0d5144"]
    }
  ]
}
```

`reason` is a closed enum (`elevenlabs_rate_limited`, `elevenlabs_error`,
`circuit_open`, `quota_exhausted`) for logging and for the client to decide
whether to surface anything. Note the text is included here for the
all-native case (nothing was spoken yet), which is the one situation where the
client genuinely needs it.

---

## 10. PTT: `POST /api/qa`

### Request

`multipart/form-data`:

| Field         | Notes                                                                                                                 |
| ------------- | --------------------------------------------------------------------------------------------------------------------- |
| `audio`       | The recorded clip. `m4a`/`mp4` (iOS `MediaRecorder`) or `webm` (Android/desktop) — both are in Groq's supported list. |
| `session_id`  | Opaque walk session, ties the question to cached context and history.                                                 |
| `lat` / `lon` | Optional, 3dp. Used only for the staleness check below.                                                               |

Guard rails before any paid call: reject >10MB, reject clips longer than ~30s,
reject unsupported content types. A PTT question is a few seconds of speech;
anything else is a client bug or abuse.

### Stage pipeline

| #   | Stage                              | Budget        | Notes                                       |
| --- | ---------------------------------- | ------------- | ------------------------------------------- |
| 1   | Upload clip                        | ~0.2s         | 2-4s of m4a ≈ 30-80KB                       |
| 2   | Groq Whisper transcription         | ~0.5-0.8s     | 189-216× realtime; network dominates        |
| 3   | Empty/short-transcript guard       | <5ms          | Don't pay for Gemma on silence              |
| 4   | Cache lookup for `FactObject[]`    | <5ms          | No `/api/context` call                      |
| 5   | Gemma 4 one-sentence answer (TTFT) | ~0.4s         | Cached context + history + question         |
| 6   | Validation                         | <5ms          | Same validator, QA rule set (§10.4)         |
| 7   | TTS (or native fallback)           | ~0.3s         | §7 / §9                                     |
|     | **Total**                          | **~1.7-2.3s** | Release of PTT button to first answer audio |

### 10.1 Groq Whisper

- Endpoint: `POST https://api.groq.com/openai/v1/audio/transcriptions`
  (OpenAI-compatible).
- Model: `whisper-large-v3` by default — 10.3% WER vs turbo's 12%, and PTT
  questions are short so the cost difference is negligible while properly
  spelled place names matter enormously here. `whisper-large-v3-turbo` is
  2.8× cheaper and ~14% faster; make it configurable.
- `language=en` — documented to improve **accuracy and latency**. Always send it.
- `response_format=json`, `temperature=0` (recommended for transcription).

**Seed the transcription with nearby vocabulary.** Whisper accepts a `prompt`
of up to 224 tokens to guide spelling of unfamiliar words. Feed it the titles
from the cached `FactObject[]` ("Nelson's Column, Charing Cross, Trafalgar
Square, …"). This is the cheapest available win for a place-based app: the
walker is asking about the landmark they are standing beside, and its name is
already in hand. A mis-transcribed "Charing Cross" as "charring cross" degrades
every downstream stage.

**Known cost floor:** Groq bills a minimum of 10 seconds per request. A 3-second
PTT question bills as 10. Worth knowing before designing per-second economics.
Limits: 25MB free tier / 100MB dev tier; audio is downsampled to 16kHz mono
server-side, so no client-side conversion is required.

### 10.2 Conversation history

- Append `{role: "user", content: transcript}` per question, and the _validated_
  answer as `{role: "assistant"}`.
- **Cap at a small number of turns** (e.g. 3-4 exchanges). Unbounded history is
  a latency and cost leak, and a walk's useful follow-up window is short
  ("what about that one?").
- **Store the validated answer, never the raw model output.** If the validator
  dropped the model's sentence and substituted a refusal, the history must show
  the refusal — otherwise the model is primed by its own hallucination on the
  next turn.
- Expiry: ~15 minutes, or the walk session's end.

### 10.3 Cached FactObject context

`/api/qa` reuses the facts already fetched for this walk; it never re-queries
`/api/context`. Consequences to design for:

- **Staleness.** The walker moves. A fact set fetched 800m ago is about where
  they _were_. Rule: if the request carries `lat`/`lon` and the walker has moved
  beyond the ~1.5km search radius since the context was built, either refresh
  the context or answer with an explicit "I don't have anything about here"
  rather than answering about a place out of sight. Answering confidently about
  a landmark behind them is a grounding failure even though every citation is
  technically valid.
- **Empty context.** If the walk is in NOTICE mode there are no facts, so no
  question can be grounded. The QA rule set must then always take the refusal
  path (§10.4).

### 10.4 Answer grounding, and the refusal path

Answers are claims, so they get the same treatment as narration: **one
sentence, must cite an in-context `fact_id`, dropping the sentence is the
default**.

The wrinkle is the honest "I don't know". A model told to always cite cannot
answer a question the facts do not cover — and if it answers anyway, the
validator correctly deletes it, leaving silence. So the pipeline needs an
explicit, safe refusal path:

1. The prompt offers a structured escape: if the facts do not answer the
   question, reply with the single sentinel token `INSUFFICIENT`.
2. **The validator is still the guarantee.** If the model's sentence has no
   valid citation and is not the sentinel, it is treated as insufficient
   regardless — the sentinel is a convenience, not the mechanism. This mirrors
   the project-wide rule that prompts advise and code decides.
3. The server then substitutes a **canned refusal** — a fixed constant, never
   model output, exactly like `STATIC_NOTICE_LINES`:
   `"I don't have a source for that one nearby."`
   It is pre-approved, pre-rendered as audio, and cannot hallucinate.

QA rule set (supersedes the narration sets for this path):

| #   | Rule                                                 | Failure →                     |
| --- | ---------------------------------------------------- | ----------------------------- |
| Q1  | Exactly 1 sentence                                   | take the first; drop the rest |
| Q2  | If it cites: every id must be in the cached context  | drop → refusal                |
| Q3  | If it has no citation and is not the sentinel        | drop → refusal                |
| Q4  | Direction, physical-assertion and traffic deny-lists | drop → refusal                |
| Q5  | Sentinel present                                     | replace with canned refusal   |

Note Q2 is the same membership check as R2 — the reason the answer cannot lean
on general knowledge about the area is that an uncited sentence has no path to
being spoken.

### 10.5 Response

```json
{
  "transcript": "who built the column?",
  "answer": "Nelson's Column was completed in 1843.[cite:9c1f2ab73e0d5144]",
  "spoken_text": "Nelson's Column was completed in 1843.",
  "citations": ["9c1f2ab73e0d5144"],
  "source_urls": ["https://en.wikipedia.org/wiki/Nelson%27s_Column"],
  "audio_url": "/api/audio/x9y8z7",
  "use_native_tts": false,
  "refused": false,
  "timings_ms": { "transcribe": 640, "answer": 410, "tts": 280 }
}
```

`timings_ms` is returned deliberately: the PTT path is the one users perceive as
interactive, and per-stage latency is the only way to tell which dependency is
responsible when it feels slow.

**Implemented deviation (as built).** The shipped `/api/qa` streams the audio
body directly (`StreamingResponse`) instead of returning an `audio_url` token,
and carries the text fields in **response headers** rather than a JSON body.

- The body is audio, so a JSON envelope cannot accompany it. Metadata therefore
travels in headers (`X-Transcript`, `X-Answer-Text`, `X-Answer-Citations`,
`X-Answer-Sources`, `X-Timings-Ms`). That is also what makes the answer text
available to the client *before* any audio arrives, which is the precondition
for the native-TTS fallback in section 9.
- Streaming the body removes the "wait for the whole clip" step, so the PTT path
keeps the budget in section 10's table.
- The cost is that a failure *after* the first chunk cannot change the status
code. It does not need to: the client already holds the text. Failures *before*
the first chunk are still detected, because the first chunk is awaited before
the response is committed. `/api/walk` keeps the `audio_url` + token model,
where buffering is required anyway to serve the relay.
- Each header value is percent-encoded **exactly once**. Encoding twice turns
`%27` into `%2527` and the client cannot recover the URL with one decode.

### 10.6 Privacy

The walker's **voice** is new sensitive data that PRD §5 does not currently
mention. Commitments this design adopts:

- Audio is held in memory only for the duration of transcription and is **never
  written to disk or a database**, in keeping with the "zero location history
  stored" stance.
- Audio and transcript are shipped to a third party (Groq) for processing. That
  belongs in PRD §5 explicitly, not only in code comments.
- Transcripts are not persisted; conversation history (§10.2) is session-scoped
  and expires with the walk.

---

## 11. Session state and multi-instance

Both `/api/walk` (fact context, audio tokens) and `/api/qa` (cached facts,
conversation history) need per-walk state. Options:

| Option                                           | Verdict                                                                                                                                  |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Process-local dict with TTL                      | Simplest; **correct only on a single instance.** On Render, a second instance means a PTT request can land where the session is unknown. |
| Shared store (Redis / Render Key Value) with TTL | Recommended once horizontally scaled.                                                                                                    |
| Signed context returned by the client            | Stateless and tamper-evident, but pushes a location-derived blob onto the device, which sits awkwardly with the privacy stance.          |

Recommendation: **process-local TTL store for v1, with a health check that
fails loudly if more than one instance is running without a shared store.**
Failing loudly beats the alternative, which is a walker asking a question and
silently getting a refusal because they hit the wrong box.

Audio tokens (§6) must live in the same store as the narration that issued them.

---

## 12. Cancellation and barge-in

The walker can press PTT while narration is playing. Expected behaviour: audio
stops, the walk listens.

- Client: `AbortController` on the `/api/walk` fetch, pause and clear the
  `<audio>` queue, cancel `speechSynthesis`.
- Server: detect client disconnect and tear down the in-flight ElevenLabs
  request. Without this, cancelled narrations keep consuming concurrency slots
  and quota for audio nobody will hear — the exact resource that §7 says is
  scarce.
- Any sentence already dispatched to ElevenLabs is still billed. Keep the
  prefetch window at one sentence so a barge-in wastes at most one short clip.

---

## 13. Observability

| Metric                                    | Why                                                                                           |
| ----------------------------------------- | --------------------------------------------------------------------------------------------- |
| TTFB per stage (context → LLM → TTS)      | The budget in §3 is only meaningful if measured per stage.                                    |
| `use_native_tts` rate, by reason          | The primary health signal for ElevenLabs. A step change is an incident.                       |
| 429 rate, concurrency saturation          | Precedes the failures that trigger fallback.                                                  |
| Barge-in rate, wasted-clip count          | Measures whether §12's prefetch window is right.                                              |
| `/api/qa` per-stage timings, refusal rate | A rising refusal rate means either the questions are out of scope or retrieval is too narrow. |

p50 and p95, not averages: averages hide exactly the tail that walkers notice.

---

## 14. Test strategy

| Test                | Assertion                                                                      |
| ------------------- | ------------------------------------------------------------------------------ |
| Sentence dispatch   | Sentence 1 is requested from TTS before sentence 2 exists                      |
| Ordering            | Out-of-order synthesis completion still emits frames in order                  |
| No tags to TTS      | The string passed to the TTS layer never contains `[cite:`                     |
| Gating              | Nothing reaches the TTS layer without a `ValidatedSentence`                    |
| Token security      | Unknown/expired/foreign-session token → 404, and **no synthesis is attempted** |
| Token single-use    | Second fetch of the same token is rejected                                     |
| Fallback, proactive | Breaker open → frames carry `use_native_tts: true` and no `audio_url`          |
| Fallback, reactive  | TTS 429 → client path speaks via `speechSynthesis` with the already-held text  |
| Reminder            | The static reminder asset is served without any TTS call                       |
| PTT guards          | Oversize/overlong/unsupported clip rejected before any paid call               |
| PTT refusal         | Uncited answer → canned refusal; `[cite:...]` never spoken                     |
| PTT staleness       | Moved beyond radius → refresh or refuse, never answer about the old location   |
| PTT history         | The stored assistant turn is the validated text, not the raw model output      |
| Barge-in            | Abort cancels upstream TTS and frees the concurrency slot                      |

---

## 15. Verified facts, open decisions, and what is unverified

### Verified against vendor documentation

| Fact                                                                                                                                   | Source                       |
| -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| Three TTS modes; HTTP stream for known text; WS lowest TTFB                                                                            | ElevenLabs integration guide |
| Over HTTP each in-flight request counts fully against concurrency; over WS only active generation                                      | ElevenLabs integration guide |
| `eleven_flash_v2_5` ≈75ms inference, 32 languages, 40k char limit                                                                      | ElevenLabs models docs       |
| Retry 429/5xx with exponential backoff + full jitter                                                                                   | ElevenLabs integration guide |
| Groq transcription endpoint and OpenAI-compatible shape                                                                                | GroqDocs                     |
| Groq limits: 25MB free / 100MB dev, min billed 10s, `language` aids accuracy+latency, `prompt` ≤224 tokens, formats incl. m4a/mp4/webm | GroqDocs                     |
| iOS suspends `AudioContext` in the background regardless of playback                                                                   | WebKit bug 231105            |

### Unverified — no live calls were made

- **No ElevenLabs request has been made.** Voice id, output format string,
  latency-optimisation parameters, and the exact 429 body are assumed. Per the
  lesson from `/api/context`, where `gsrlat`→`gscoord` broke silently with a
  `200`, confirm every parameter against the live API before building on it.
- **No Groq request has been made.** Whether `prompt` seeding measurably
  improves place-name accuracy is an assumption worth measuring, not a fact.
- **The ~2.2s budget is arithmetic, not measurement.** Nothing in §3 has been
  timed end to end.
- **Inter-sentence gap duration is unknown.** §8 accepts a gap without knowing
  its size; measure it before deciding whether §4's trade-off is acceptable.

### Open decisions

1. **Voice-consistency vs seamless fallback** (§9) — repeats content, or
   switches voice mid-narration. A product call.
2. **`flash_v2_5` vs `multilingual_v2`** (§7) — latency vs prose quality.
3. **Session store** (§11) — single-instance assumption vs Redis. Decide when
   the instance count does.
4. **PTT barge-in semantics** — should a question interrupt narration
   immediately, or wait for the current sentence to finish?

### My answers to your Open questions/decisions (can be altered if something isn't optimal)

1. Voice-consistency vs seamless fallback (§9): Go Seamless.
   If ElevenLabs rate-limits mid-response, let the phone switch to the native robotic voice for the remaining sentences. It is audibly jarring, but repeating a 20-second paragraph the user just heard (voice-consistent) is a much worse walking experience.
2. flash_v2_5 vs multilingual_v2 (§7): Use flash_v2_5.
   Your PRD mandates a time-to-first-audio (TTFA) of under 2.5 seconds. Do not sacrifice latency for slightly better prosody. flash_v2_5's ~75ms inference is the only way you hit that budget over mobile networks.
3. Session store (§11): Process-local dict with TTL.
   Do not waste hours setting up and debugging Redis. For this Hacktoberfest MVP, you only need it to survive a 30-minute walk. Keep the context and audio tokens in a simple Python dictionary in memory. Force Render to run on exactly one instance so you don't have routing mismatches.
4. PTT barge-in semantics: Interrupt immediately.
   When the user holds the Push-To-Talk button, instantly abort the /api/walk fetch, clear the <audio> queue, and cancel the in-flight ElevenLabs request. If they ask a question, they want to be heard now, not wait for Gemma to finish describing a monument.
