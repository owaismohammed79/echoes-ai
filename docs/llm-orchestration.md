# LLM Orchestration Pipeline — Design

**Status:** sections 3-7 are implemented and wired to an HTTP route, inside the
`server/app/` package: [`app/services/prompts.py`](../server/app/services/prompts.py),
[`app/validation/`](../server/app/validation),
[`app/services/gemma.py`](../server/app/services/gemma.py),
[`app/services/narration.py`](../server/app/services/narration.py). Covered by
[`tests/test_validator.py`](../server/tests/test_validator.py) and
[`tests/test_narration.py`](../server/tests/test_narration.py), and reachable via
`POST /api/walk` ([`app/api/walk.py`](../server/app/api/walk.py)). The Gemma 4
transport has **not** been exercised against the live service.

**Governed by:** [AGENTS.md](../AGENTS.md) §1 (factual grounding), §2 (safety),
§5 (fallbacks); [PRD.md](../PRD.md) §3 (data pipeline), §5 (latency).

**Depends on:** the `FactObject[]` + `mode` contract produced by
[`/api/context`](../server/app/api/context.py).

---

## 1. Scope and non-goals

In scope: the path from `FactObject[]` to *validated spoken text*.

Out of scope: `/api/context` (already built), TTS synthesis, the iOS audio
unlock, PTT/`/api/qa`, and the static demo kill-switch.

The central claim of this design: **the prompt is advisory, the validator is
authoritative.** AGENTS.md §1 forbids relying on system-prompt instructions to
prevent hallucination. Every sentence reaches TTS only after passing a
deterministic, pure-function check.

---

## 2. Pipeline overview

```
/api/context  ──> mode
                   │
      ┌────────────┴─────────────┐
      │ mode == "STORY"          │ mode == "NOTICE"
      ▼                          ▼
  STORY_PROMPT              NOTICE_PROMPT
  (facts inlined)           (time/daypart [+weather])
      │                          │
      └────────────┬─────────────┘
                   ▼
            LLMProvider.stream()            ← token stream (SSE)
                   ▼
        ┌──────────────────────┐
        │ 1. SentenceAssembler │  buffer → complete sentences
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │ 2. Normaliser        │  unicode/whitespace/tag canonicalisation
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │ 3. Validator         │  mode-specific rules (pure function)
        └──────────┬───────────┘
             pass ─┴─ drop
              │         └──> RejectionReason (logged, never spoken)
              ▼
        ┌──────────────────────┐
        │ 4. Emitter           │  strip tags, attach citations
        └──────────┬───────────┘
                   ▼
        TTS queue  +  client source-links panel
```

Two properties this shape buys:

- **Incremental.** Sentences are validated and enqueued as they complete, so
  time-to-first-audio does not wait for the whole generation.
- **Testable.** Stages 2–4 are pure functions. No network, no LLM, no audio
  needed to test the grounding guarantees.

---

## 3. Provider interface

```python
class LLMProvider(Protocol):
    def stream(self, *, system: str, user: str) -> AsyncIterator[str]: ...
```

### Provider selection

| Provider | Model id | Notes |
| --- | --- | --- |
| Gemini API (default) | `gemma-4-26b-a4b-it` | MoE; lowest latency. `:streamGenerateContent` (SSE) |
| Gemini API (quality) | `gemma-4-31b-it` | Same API surface, slower |
| Groq | Gemma 4 variants | Fallback if AI Studio quota is exhausted |
| Ollama (local) | `gemma-4` | Dev/offline only; not a Render target |

The model id is configuration, not a constant — quota fallback must not require
a code change.

### Two settings that matter for the latency budget

1. **Streaming.** Non-streaming would make TTFA = full generation time. PRD §5
   requires < 2.5s total, and `/api/context` already measured ~1.0s.
2. **`thinking_level: "minimal"`.** Gemma 4 has an internal thinking process
   that can be toggled. The default is not acceptable here: reasoning tokens
   are generated before any spoken text, which directly spends the TTFA budget
   on work this task does not need. Narration from a provided fact list is not
   a "logically demanding domain". Set `minimal` explicitly rather than
   inheriting the default.

> Note: the streaming endpoint name and the `thinking_level` field should be
> confirmed against the provider SDK before implementation; both were skimmed
> from current docs, not exercised.

---

## 4. Context payload

What the model is shown. Design constraints: tokens must be **copyable
verbatim**, ids must be **unambiguous**, and nothing private may leak.

```
NEARBY FACTS (the only permitted source of claims)

FACT 1
  id: 9c1f2ab73e0d5144
  title: Nelson's Column
  detail: Nelson's Column is a monument in Trafalgar Square, London.
  distance: 21.6m

FACT 2
  id: 4be7c0a91d2f8836
  title: Charing Cross
  detail: Charing Cross is a junction in Westminster, London...
  distance: 23.6m
```

Rules:

- **Explicit field mapping from `FactObject`** (the contract returned by
  [`/api/context`](../server/app/api/context.py)) — the prompt is fed four of the five
  fields, renamed for the model's benefit:

  | `FactObject` | prompt field | why |
  | --- | --- | --- |
  | `fact_id` | `id` | the token the model must copy into `[cite:...]` |
  | `title` | `title` | names the place in prose |
  | `snippet` | `detail` | the only permitted claim material |
  | `distance_m` | `distance` | what makes the narration feel local |
  | `source_url` | **withheld** | see below |

- **`source_url` is deliberately withheld from the model.** The model never
  needs it: the validator captures `fact_id` from the tag (§7.7) and the client
  resolves the URL from the validated citation. Passing URLs in would only give
  the model something to mistype or fabricate, and a corrupted link is a
  grounding failure the user can see.
- **No coordinates are sent** — the model has no use for them, and PRD §5 makes
  coordinate handling a privacy matter.
- `detail` is the Wikipedia/park snippet already fetched. Never fetched again,
  never expanded.
- Cap the list (≈8 nearest) to bound tokens and keep the narration local.
- The list is ordered nearest-first, matching the `/api/context` response.

---

## 5. STORY_PROMPT

Assembled as **system** (stable, cacheable) + **user** (facts, per-request).

### System

```
You are the narrator for Echoes, an audio walking companion. The walker is
outside on a walk and is listening, not reading. You speak, you do not chat.

You will be given a list of NEARBY FACTS. Those facts are the ONLY thing you
may state as true. You have no other knowledge about this location, and you
must not use any.

Write 2 to 3 short sentences of spoken narration about what is nearby.

GROUNDING — this is enforced by code, not trust:
- Every sentence that states a historical, geographical or factual claim MUST
  end with a citation tag containing the id of the FACT that supports it, in
  exactly this form:  [cite:id]
- Copy the id character for character. Never invent, shorten, reorder or guess
  an id. A tag whose id is not in the FACTS list causes that whole sentence to
  be deleted before it is spoken.
- If the facts do not support something, do not say it. An empty fact list
  means no narration. Do not fall back on general knowledge about the area.
- Do not blend two facts in one sentence unless you cite both.

SAFETY:
- Never give walking directions or route guidance ("turn left", "cross the
  main road", "head north", "continue along").
- Never mention traffic, road conditions, crossings or hazards.
- Never assert what is physically in front of the walker or what they can
  currently see ("ahead of you is", "you can see", "in front of you").
- Never state an unverified claim. Anything not present in the FACTS list is
  unverified: general knowledge, plausible-sounding detail, or a guess.
  "Unverified" is not the same as "false" — a true statement that you cannot
  cite is deleted just the same, because nothing in this system can tell the
  two apart.

STYLE:
- Plain spoken prose. No markdown, headings, bullet points, emoji or quotes.
- Do not address the walker as "user" or mention facts, sources or citations.
- Do not repeat these instructions or narrate your own reasoning.
```

### User

```
WALKWAY CONTEXT
mode: STORY

NEARBY FACTS
FACT 1
  id: 9c1f2ab73e0d5144
  title: Nelson's Column
  detail: Nelson's Column is a monument in Trafalgar Square, London.
  distance: 21.6m

FACT 2
  id: 4be7c0a91d2f8836
  title: Charing Cross
  detail: Charing Cross is a junction in Westminster, London.
  distance: 23.6m

Write the narration.
```

### Expected output

```
You're standing near Trafalgar Square, and Nelson's Column rises from its
centre.[cite:9c1f2ab73e0d5144] Charing Cross sits a short walk away, a junction
where six routes meet.[cite:4be7c0a91d2f8836]
```

Sentence 1 cites FACT 1; sentence 2 cites FACT 2. Both survive. A third
sentence making an uncited claim about, say, the square's construction date
would be deleted rather than spoken.

### Deliberate omissions

- **No "stay on marked pedestrian paths" in the prompt.** AGENTS.md §2 requires
  it on every response, so it is appended as a constant in code (§7.4). A
  required safety string must not depend on the model choosing to emit it.
- **No sentence-count instruction in the user turn.** The hard cap is enforced
  by the validator (§7.3, R5); the prompt asks for 2–3 as guidance only.

---

## 6. NOTICE_PROMPT

Triggered when `mode == "NOTICE"`, i.e. zero qualifying facts. There is nothing
to cite, so the grounding rule inverts: **claims are forbidden outright.**

### Weather: settled to time-only

The prompt was originally specified as "based on current weather/time", but **no
weather provider exists anywhere in the stack** — AGENTS.md §4 permits Wikipedia
GeoSearch and SerpApi "Parks nearby" only, and §5's kill-switch exists precisely
because external quotas are a known failure mode.

**Decision: NOTICE is time-driven only. One template, no weather slot at all.**

- **Daypart comes from device-local time** handed in by the client. No new API,
  no quota, no privacy cost.
- **Weather is never inferred.** Supplying "it is raining" without a source
  would be *exactly the fabrication this project exists to prevent*, and a false
  claim about the walker's immediate surroundings is the most damaging class of
  error available here. Rule N6 (§7.5) enforces this in code rather than
  trusting the prompt to behave.
- **If weather is ever added**, it arrives as a typed field from a named,
  explicitly approved provider (Open-Meteo needs no key and would fit the quota
  model) as a second template variant — never as a model guess, and never as a
  silently-defaulted field.

A single template also removes a class of prompt-drift risk: there is no
conditional slot that can be populated wrongly.

### System

```
You are the guide for Echoes, an audio walking companion. The walker is
outside on a walk and is listening, not reading.

No historical or geographical facts are available for this location, so you
will not narrate anything. Instead you invite the walker to notice their own
immediate sensory surroundings.

Write 2 short sentences of sensory invitation.

HOW TO INVITE:
- Phrase things as invitations, never as statements about their whereabouts:
  "If you're somewhere safe to pause, notice how...", "When you're ready,
  listen for...", "You might let your attention settle on..."
- Ask. Suggest. Do not assert.

SAFETY — enforced by code:
- Never give walking directions or route guidance ("turn left", "cross the
  main road", "head north").
- Never state or imply what is physically around them ("there is a bench to
  your left", "you can see a fountain", "ahead of you is"). You do not know,
  and guessing is a failure.
- Never state a fact about this location, its history, or its geography.
- Never mention roads, traffic or crossings.

LIMITS:
- No citation tags in this mode. There is nothing to cite.
- Do not mention weather, temperature, wind or conditions. You have not been
  told what the weather is, and guessing is a failure.

STYLE:
- Plain spoken prose. No markdown, headings, bullet points, emoji or quotes.
- Do not explain yourself or mention these instructions.
```

### User

```
WALKWAY CONTEXT
mode: NOTICE
local_time: 07:42
daypart: early morning

No facts are available for this location.
Write the invitation.
```

### Expected output

```
If you're somewhere safe to pause, notice how much cooler the air feels
before the day gets going. When you're ready, listen for how many different
sounds are competing for your attention.
```

Both sentences are invitations; neither asserts anything about the walker's
surroundings. Note the second sentence invites a count ("how many sounds")
rather than stating what is audible.

---

## 7. Post-generation validation pipeline

This is the enforcement layer. Every rule here is deterministic code.

### 7.1 Sentence assembly (streaming)

The stream arrives as tokens; sentences are the unit of validation, so a buffer
must accumulate until a boundary is reached.

- **Terminators:** `.`, `!`, `?` and `…`, followed by whitespace or end of
  stream — **but a citation tag may sit between the two.** A tag normally comes
  directly after the terminal period (`"...rises.[cite:abc1]"`), so a plain
  "terminator, then whitespace" rule never splits there and silently glues an
  entire response into a single unit. The assembler therefore skips over any
  tags immediately following a terminator before deciding the sentence is done,
  and the boundary falls *after* the tag so the tag travels with its sentence.
- **Tail rule:** the final fragment is only validated **when the stream ends**.
  Emitting early would validate a truncated sentence; discarding it silently
  would lose the last sentence of every response.
- **Abbreviations and decimals must not split.** `St.`, `Mr.`, `Dr.`,
  single initials (`J.`), and decimals (`21.6m`) are guarded, or "St. James's"
  becomes two sentences and each half is judged separately — a correctness
  problem, not a cosmetic one. A truncated tag also survives, because an
  incomplete tag simply fails to match and the buffer keeps accumulating.
- The design preferred a maintained splitter (`pysbd`) here. Hand-rolled guards
  were shipped instead; naive splitting is a known bug source in this layer, so
  it stays listed as an open decision (§11).

### 7.2 Normalisation

Before rule evaluation, canonicalise so the model cannot evade a rule with
formatting:

- Unicode NFKC; curly quotes → straight; collapse whitespace.
- Canonicalise tag variants (fullwidth brackets `［cite:id］`, interior
  whitespace `[cite: id]`) toward `[cite:id]`.
- Lower-cased copy retained for deny-list matching; the original is retained
  for speech.

### 7.3 Citation tag

The delimiter is `[cite:fact_id]`:

```python
CITATION_PATTERN = re.compile(r"\[cite:\s*([a-zA-Z0-9_-]+)\]")
```

> **Settled: `[cite:fact_id]`.** AGENTS.md §1 and PRD.md §3 both render the
> citation marker as an empty span (`...end with an inline citation tag: ``.`),
> because the original token was an unescaped HTML-ish fragment the Markdown
> renderer consumed. This document initially proposed `[[id]]`; the delimiter
> was subsequently settled to the `[cite:...]` form above. It lives in exactly
> one place -- `validator.CITATION_PATTERN` -- and `prompts.format_citation_tag`
> renders the prompt from that same source, so the two cannot drift apart. A
> mismatch would silently drop **every** sentence.

The capture class is `[a-zA-Z0-9_-]+`, not the 16-hex shape `make_fact_id`
happens to produce. That is deliberate: the pattern is permissive about *shape*
and the validator is strict about *membership* (R2 below). A well-formed tag
wrapping an id that was never supplied is still rejected, while a future change
to the id scheme does not require editing this regex. The pattern is
case-sensitive, so a model emitting `[Cite:...]` would be rejected -- worth
loosening with `re.IGNORECASE` only after measuring how often models capitalise
it, since loosening it silently widens the accepted surface.

### 7.4 STORY rules

| # | Rule | Failure → |
| --- | --- | --- |
| R1 | Sentence contains ≥1 well-formed `CITATION_PATTERN` match | DROP `NO_CITATION` |
| R2 | Every tag's id ∈ the `fact_id` set of the in-context facts | DROP `UNKNOWN_FACT_ID` |
| R3 | No direction/route language (deny-list, §7.6) | DROP `DIRECTION_LANGUAGE` |
| R4 | No physical-assertion language ("in front of you", "you can see") | DROP `PHYSICAL_ASSERTION` |
| R5 | ≤ 3 spoken sentences total; each ≤ 45 words | DROP overflow |
| R6 | No traffic/hazard vocabulary | DROP `TRAFFIC_LANGUAGE` |

**Why R1 requires a tag on *every* sentence.** The spec says drop sentences that
*make a claim* without a citation. Deciding "does this make a claim?" with
heuristics — named-entity detection, keyword lists, a second LLM call — is
either unreliable or adds latency, and AGENTS.md §1 explicitly rejects
prompt-level enforcement. The deterministic reading is: **every declarative
sentence is presumed to be a claim.** The prompt asks the model to cite every
sentence, so a well-behaved model never trips R1; when R1 does fire, it fires on
exactly the sentence whose provenance we cannot establish. The cost is
occasionally dropping a harmless sentence. That is the correct direction to
err: a missing sentence is a quieter walk, a fabricated one is a broken product.

**R2 is the rule that actually kills hallucination.** Syntactic validity alone
is worthless — a model can produce a perfectly well-formed tag wrapping an id it
invented. Membership in the in-context id set is the real check, and it also
catches citation misattribution (a real id applied to a claim it does not
support).

> Known limitation: R2 verifies the id *exists in context*, not that the tag
> *supports the specific claim* it is attached to. A model could attach a valid
> Nepal-railway id to a sentence about Nelson's Column and pass. Closing this
> fully requires semantic entailment checking (an NLI model or an LLM-judge
> call), which is out of scope here and would spend the latency budget. Worth
> deciding later; today the mitigation is that facts are drawn from a 1.5km
> radius, so cross-attribution is mostly *locally* plausible.

### 7.5 NOTICE rules (the mirror image)

A different rule set, because the failure mode inverts: there are no facts, so a
citation is not evidence of good behaviour — it is evidence the model went
looking for something to claim.

| # | Rule | Failure → |
| --- | --- | --- |
| N1 | ≤ 2 sentences | DROP overflow |
| N2 | **No** citation tags present | DROP `UNEXPECTED_CITATION` |
| N3 | Contains an invitation marker (`notice`, `listen`, `feel`, `if you're somewhere safe`, `you might`) | DROP `NOT_INVITATIONAL` |
| N4 | No direction language | DROP `DIRECTION_LANGUAGE` |
| N5 | No physical assertions | DROP `PHYSICAL_ASSERTION` |
| N6 | No weather predicates at all — weather is never supplied (§6) | DROP `UNGROUNDED_WEATHER` |
| N7 | No historical/factual claim markers (dates, "was built", "dates from") | DROP `FACTUAL_CLAIM` |

N3 and N6 are the load-bearing pair. N3 forces the *grammar* of invitation, so
even if the model is tempted to assert, the sentence has to be phrased as an
invitation to pass. N6 is the guard on §6: since no weather source exists, any
sentence mentioning conditions is a fabrication by construction. It is also the
rule that keeps the prompt change honest — if someone later adds a weather slot
without adding a provider, N6 catches the resulting guess.

### 7.6 Deny-lists

Three independent lists, deliberately kept separate because they have different
owners and different review needs.

```python
DIRECTION = re.compile(r"""
    \b(?:
      turn\s+(?:left|right)|cross\s+the\s+(?:road|street|main\s+road)|
      head\s+(?:north|south|east|west)|continue\s+along|
      go\s+straight|follow\s+the\s+(?:road|path|signs)|walk\s+towards?
    )\b
""", re.IGNORECASE | re.VERBOSE)

PHYSICAL_ASSERTION = re.compile(r"""
    \b(?:
      in\s+front\s+of\s+you|ahead\s+of\s+you|behind\s+you|
      (?:to\s+your|on\s+your)\s+(?:left|right)|
      you\s+can\s+(?:see|hear)\b|you(?:'ll|\s+will)\s+see|
      there\s+is\s+a\s+\w+\s+(?:to\s+your|beside|next\s+to)
    )\b
""", re.IGNORECASE | re.VERBOSE)

TRAFFIC = re.compile(r"""
    \b(?:traffic|crossing|junction\s+ahead|roadworks|hazard|lane\s+closure)\b
""", re.IGNORECASE | re.VERBOSE)
```

**These patterns must not false-positive on NOTICE invitations.**
"If you're somewhere safe to pause" contains "to pause", not "to your left" —
fine. But "notice what's to your right" is both a physical assertion *and* an
invitation, and N5 correctly drops it. This intersection is the main thing to
fuzz with an adversarial corpus.

### 7.7 Tag stripping and citation capture

The tag must never be spoken, but its information must not be discarded.

```python
@dataclass(frozen=True)
class ValidatedSentence:
    text: str                    # tag-free, TTS-ready
    citations: tuple[str, ...]   # fact_ids, in order of appearance
    source_urls: tuple[str, ...] # resolved from the context map
```

- `text` is built by substituting each tag with `""`, then normalising
  whitespace and dropping a stranded space before terminal punctuation.
- `citations`/`source_urls` feed the client's source-links panel (AGENTS.md §3),
  so grounding is visible to the user rather than merely claimed.

This is the point where "the model produced a tag" becomes "the user can audit
the claim", which is the whole premise of the product.

### 7.8 Terminal behaviour

| Condition | Behaviour |
| --- | --- |
| ≥1 sentence survives | Speak survivors, then append the fixed reminder |
| All sentences dropped (STORY) | Degrade once to NOTICE_PROMPT |
| Degradation also yields nothing | Speak static safe copy + reminder |
| Stream errors mid-generation | Emit already-validated sentences; do not retry the LLM inline |
| Empty stream | As "all dropped" |

The **degradation must be capped at one attempt** (a flag, not recursion). A
prompt or model change that makes R1 fire on everything would otherwise become
an unbounded loop on every walk — the failure mode of a strict validator is
infinite retry, and it needs an explicit brake.

The **mandatory reminder** (`Stay on marked pedestrian paths.`) is appended in
code after validation and is exempt from per-sentence rules. It is a constant,
not model output, so validating it would be theatre. Its exemption is exactly
why it must *not* be left to the prompt (§5).

---

## 8. Observability

A strict validator is only safe if you can see it working.

```python
class RejectionReason(StrEnum):
    NO_CITATION = auto()
    UNKNOWN_FACT_ID = auto()
    DIRECTION_LANGUAGE = auto()
    PHYSICAL_ASSERTION = auto()
    TRAFFIC_LANGUAGE = auto()
    OVERFLOW = auto()
    UNEXPECTED_CITATION = auto()
    NOT_INVITATIONAL = auto()
    UNGROUNDED_WEATHER = auto()
    FACTUAL_CLAIM = auto()
```

Emit per response: mode, sentences generated/surviving/dropped, reasons,
prompt template hash, model id.

- **Drop rate is the health metric.** A rising `NO_CITATION` rate means the
  prompt and the model have diverged (usually after an undisclosed model
  update). A rising `UNKNOWN_FACT_ID` rate means the model is inventing ids.
- **The prompt template hash** makes prompt drift visible; a hand-edit to a
  prompt string is otherwise an untracked behavioural change with a safety
  impact.
- **Reason counts are not errors.** Drops are the validator succeeding.

---

## 9. Latency budget

PRD §5 requires TTFA < 2.5s. Measured `/api/context`: ~1.0s (Trafalgar Square,
20 facts, live).

| Stage | Budget |
| --- | --- |
| `/api/context` | ~1.0s (measured) |
| LLM time-to-first-token | ~0.4s target |
| First sentence generation | ~0.3s |
| Validation | < 5ms (pure, in-process) |
| **Headroom** | ~0.8s |

Consequences for the design: `thinking_level: minimal` (§3) is not optional;
validation must be incremental rather than post-hoc; and the prompt asks for
2–3 sentences rather than a paragraph, which shortens the tail as well as the
head.

---

## 10. Test strategy

The validator being pure is what makes the safety property testable without a
model in the loop.

| Test class | Assertion |
| --- | --- |
| Golden STORY | Known facts + known output → both sentences survive, citations resolve |
| Injected hallucination | Sentence with no tag → dropped |
| Fabricated id | Well-formed tag with an off-list id → dropped (R2) |
| Misattribution | Valid id attached to an unrelated claim → *passes*, documented limitation (§7.4) |
| Direction leak | "turn left at the corner" → dropped |
| Physical assertion | "there's a fountain in front of you" → dropped |
| Streaming split | A tag split across two token chunks still validates |
| Abbreviations | `St. James's`, `21.6m` are not split into two sentences |
| Total-drop brake | Every sentence dropped → exactly one NOTICE degradation, no loop |
| NOTICE validity | Any weather reference at all → dropped (N6) |
| NOTICE safety | "there is a bench to your left" → dropped (N5) |

---

## 11. Decisions and remaining questions

### Settled

- **Citation delimiter: `[cite:fact_id]`** — see §7.3.
- **NOTICE weather: time-only.** No weather provider; no weather slot in the
  prompt; N6 enforces the absence in code (§6). This keeps AGENTS.md §4's API
  guardrails intact and adds no dependency.
- **Grounding is enforced in code, not by prompt.** Every sentence is presumed
  a claim and must cite an in-context id (§7.4, R1–R2).
- **The mandatory path reminder is a code constant**, exempt from validation
  (§7.8), not a prompt instruction.

### Still open

1. **Degrade-to-NOTICE vs static fallback** when every STORY sentence is
   dropped (§7.8). Degrading keeps the walk alive; static copy is predictable
   and cheaper.
2. **`26b-a4b-it` vs `31b-it`.** Latency vs prose quality on a 2-sentence
   budget.
3. **Semantic entailment (§7.4).** Accept the cross-attribution limitation, or
   spend latency on an NLI check later.
4. **Sentence splitter.** Adopt `pysbd` or hand-roll the abbreviation rules
   (§7.1) — the latter is a known source of quiet bugs.

## 12. Alternative considered

**Structured output instead of regex over prose.** The Gemini API supports
response schemas, so the model could emit
`{"sentences": [{"text": ..., "fact_ids": [...]}]}` and the validator would
parse JSON rather than regex prose — eliminating the abbreviation/streaming
splitting problems in §7.1 entirely.

Rejected as the primary design because AGENTS.md §1 and PRD §3 mandate an
*inline citation tag* in the text, which implies prose. It remains the stronger
option if the inline-tag requirement is ever relaxed, and would be worth
revisiting.
