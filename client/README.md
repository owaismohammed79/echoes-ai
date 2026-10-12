# Echoes — client

Mobile-first, voice-first walking companion. One screen: the walk starts, the
narrative plays, you can hold to ask, and the sources are always one tap away.

```bash
npm install
npm run dev        # http://127.0.0.1:5311
npm run typecheck
npm run build
```

## How it is composed

The screen is assembled from open-source components rather than hand-written
CSS. Each registry component was pulled with the shadcn CLI, so what is in
`src/components/ui/` is upstream source, not a rewrite:

| Role | Source | Component |
| --- | --- | --- |
| Ambient audio state | magicui | `ripple.tsx` |
| Depth behind the ring | aceternity | `background-beams.tsx` |
| Start Walk CTA | magicui | `shimmer-button.tsx` |
| Push-to-talk light | aceternity | `glowing-effect.tsx` |
| Source panel | shadcn/ui + vaul | `card.tsx` + `Drawer` |
| Surface grain | aceternity | `noise-overlay.tsx` (inlined) |
| Pulsing orb | aceternity | `push-to-talk.tsx` (`PulseRings`) |
| Walk clock, counting up in seconds | magicui | `number-ticker.tsx` |
| Light that circuits the transcript while audio plays | magicui | `border-beam.tsx` |
| State word, shimmering while audio plays | magicui | `animated-shiny-text.tsx` |
| Transcript, revealed word by word with the audio | aceternity | `text-generate-effect.tsx` |

Installed with:

```bash
npx shadcn@latest add card \
  "https://magicui.design/r/ripple.json" \
  "https://magicui.design/r/shimmer-button.json" \
  "https://magicui.design/r/number-ticker.json" \
  "https://magicui.design/r/border-beam.json" \
  "https://magicui.design/r/animated-shiny-text.json" \
  "https://ui.aceternity.com/registry/background-beams.json" \
  "https://ui.aceternity.com/registry/glowing-effect.json" \
  "https://ui.aceternity.com/registry/text-generate-effect.json"
```

Aceternity's `pulsating-button` is not in its public registry (the CLI gets a
401), so the orb pulse is implemented locally in the same idiom. `noise-texture`
is likewise not exposed as a registry item; its technique (an SVG feTurbulence
tile at a few percent opacity) is inlined in `noise-overlay.tsx`.

## Palette: how upstream source lands on two colours

The design caps the system at two colours — base `#09090B`, accent `#10B981`, on
surface `#18181B` — with a 1px radius everywhere and no drop shadows. The
registry components are written against shadcn's semantic tokens rather than raw
hex, so `src/index.css` redefines those tokens (`--foreground`, `--primary`,
`--border`, `--radius-*`) and every component inherits the palette without its
markup being rewritten.

Seven edits to upstream source were unavoidable, because they were not
token-driven:

1. **`background-beams.tsx`** — its gradient stops were `#18CCFC`/`#6344F5`/
   `#AE48FF` (blue/violet) and a `#d4d4d4` vignette. Repointed to the accent and
   the surface. Also dropped a `"use client"` directive and fixed an upstream
   type error (an unused `path` binding, which fails `tsc`).
2. **`glowing-effect.tsx`** — its `--gradient` was a four-colour rainbow
   (`#dd7bbb`, `#d79f1e`, `#5a922c`, `#4c7894`). Repointed to the accent.
3. **`ripple.tsx`** — removed `shadow-xl` and the `bg-foreground/25` fill so the
   rings read as hairlines, and took the border colour from `--primary`. Also
   removed the bottom fade mask, which is a wide-container treatment and wrong
   for a single centred ring.
4. **`shimmer-button.tsx`** — removed the "Highlight" overlay layer (a stack of
   inset shadows) and changed the radius default from `100px` to `1px`.
5. **`border-beam.tsx`** — its `colorFrom`/`colorTo` defaults were `#ffaa40` and
   `#9c40ff` (orange and violet). Repointed to the accent and its transparent
   form, so a beam rendered without props is still on-palette.
6. **`text-generate-effect.tsx`** — upstream reveals on a fixed `stagger(0.2)`
   timer, which cannot be in sync with anything; the transcript would have been
   a slideshow running beside the audio. It now accepts an optional `revealed`
   count that the controller drives from real speech boundaries (see below),
   and behaves exactly as upstream when the prop is omitted. Its hard-coded
   `font-bold text-2xl mt-4` is gone — hierarchy here comes from size and
   tracking, never weight, and the call site owns typography.
7. **`number-ticker.tsx`** and **`animated-shiny-text.tsx`** need no source edit,
   but both ship `dark:` variants (`dark:text-white`,
   `dark:text-neutral-400/70`), and this app puts `.dark` on `<html>`. A plain
   override in the calling `className` loses to a `dark:` variant in the
   cascade, so the palette values are restated as `dark:` at each call site.
   Confirmed against computed styles, not by eye.

`card.tsx` also needed a genuine repair: the CLI emitted
`import { cn } from "cn"`, which does not resolve. Its import now points at
`@/lib/utils`, and the `cn` npm package that had been installed to make that
bogus specifier resolve is gone from `package.json`.

## The transcript follows the voice

Word-by-word reveal is only meaningful if the words track the actual speech, so
`use-walk-controller.ts` listens to `SpeechSynthesisUtterance.onboundary` — the
only true sync signal the Web Speech API offers — and turns each `charIndex`
into a word count (`spoken`). Sentence boundaries are ignored, because rewinding
the reveal to the start of a sentence reads as a glitch.

Safari and several Firefox voices never fire boundaries at all. For them a
120ms timer estimates progress at `REVEAL_FALLBACK_WPS = 2.4` words/sec —
deliberately slower than natural speech so the text lags the voice rather than
printing words the voice has not reached — and it is killed permanently the
moment a real boundary arrives. `onend` snaps the remainder either way.

The transcript renders `spokenText`, not the walk's current segment, because one
code path speaks both walk segments and push-to-talk answers. Following the walk
index would have meant the reveal counting words in a sentence the voice was not
saying.

Every word stays in the DOM; only opacity and blur change. An enclosing
`aria-live="polite"` region therefore announces each sentence **once**, in full,
instead of muttering it out word by word.

## The mobile audio state machine

**Audio unlock.** `primeAudio()` runs synchronously inside the Start Walk tap —
before the geolocation prompt, before `/api/walk`. It does two things: resumes a
single app-lifetime `AudioContext` (constructing a fresh one per tap leaks a
context per walk, after which `resume()` fails silently), and plays a silent
WAV through the transport element.

The element matters. iOS grants playback rights to the element that played
during the gesture, so an element created later — after an `await` — is refused
and the first sentence arrives silent. `lib/audio-queue.ts` therefore keeps ONE
`<audio>` element that plays the unlock WAV, then sentence one, then every
sentence after it, and the Media Session describes that same element.

**The rolling queue.** `AudioQueue` holds at most `QUEUE_DEPTH` (3) sentence
clips in an array and plays them in order through that one element, advancing on
`ended` and revoking each clip's object URL as it goes. The controller owns the
rolling part: `fillQueue` tops the buffer up to three, counting the clip on the
speaker and the ones still in flight, so a slow network cannot push it past its
depth. If the speaker runs dry before the next clip lands, the queue reports
`onDrained` and playback restarts on arrival rather than stalling.

Reveal on the queue path is driven by the element's own `timeupdate`, so the
text is tied to the audio actually playing. Gapless by construction: measured
handoff between clips is 31–32ms.

**Media Session.** Published when the walk goes live — title `Echoes Walking
Guide`, artist `Echoes` — with play/pause/stop bound to the walk. A walk happens
with the phone in a pocket, so the lock screen is part of the interface. The
handlers are registered once and dispatch through a ref, so a button pressed
forty minutes in reaches the walk as it is *now*. Every binding is individually
guarded, because `setActionHandler` throws for unsupported actions rather than
ignoring them, and a missing lock screen must never take the walk down.

**Push-to-talk** is its own four-state machine, separate from the dial's
activity:

```
IDLE -> RECORDING -> UPLOADING -> ANSWERING -> IDLE
```

`RECORDING` is a real recording: `lib/mic.ts` opens a `MediaRecorder` over the
microphone and the clip handed to `transcribe` is the clip the user spoke.
`UPLOADING` is the Groq round trip, `ANSWERING` the grounded reply. Each state
has one signal — pulse rings for capture, because that is the state the user
controls; an orbiting hairline for upload, because it is the one they cannot;
the glow for answering. Presses are serialised (`ptt !== "idle"` is a no-op), so
a second finger cannot restart the machine on top of itself.

A capture that failed is stated in the same breath as the state —
`mic_denied · not recording` — and surfaced again in the transcript row. A
prompt that says "recording" over a dead microphone is the same class of lie as
a citation without a source.

## Structure

```
src/
  EchoesApp.tsx            screen composition
  components/ui/           vendored registry components (upstream)
  components/echoes/       our composition: ring, CTA, orb, drawer, noise
  hooks/use-walk-controller.ts   the state machine
  lib/audio-queue.ts       the transport: one <audio>, 3-clip rolling queue
  lib/media-session.ts     lock-screen title and transport controls
  lib/mic.ts               push-to-talk capture, with visible failure
  lib/demo-engine.ts       the one seam to the backend
  lib/voice-status.ts      remote-fallback vs text-only triage
  index.css                palette, tokens, motion
```

`use-walk-controller.ts` is where the product rules live: the audio unlock runs
synchronously inside the tap, geolocation is requested only afterwards and
rounded to three decimals, `NOTICE` mode is entered deterministically when no
facts qualify, and a failed remote voice re-speaks through `speechSynthesis` and
says so on screen.

## Wiring it to the server

Replace `demoEngine` (passed to `<EchoesApp engine={...} />`) with real callers.
Nothing else in the app knows the network exists:

| Method | Route |
| --- | --- |
| `locate` | `POST /api/context` |
| `narrate` | `POST /api/walk` |
| `transcribe` | `POST /api/qa` (Groq `whisper-large-v3`) |
| `answer` | `POST /api/qa` (Gemma 4) |

The `SimulationStrip` in the source drawer exists only to exercise the two
behaviours that are otherwise unreachable without a backend — the refusal into
Notice mode, and the voice fallback. Delete it with `demoEngine`.

## Two defects found by measuring, not looking

**The Start Walk CTA rendered 40px wide.** It is 56px tall, emerald, 1px radius
and mono — correct in every way except width, because its wrapper was
shrink-to-fit (80px) and the button's `w-[calc(100%-40px)]` was therefore a
percentage of a box that was itself sized *by* that button. The wrapper is now
`w-full px-5` and the button `w-full max-w-[380px]`, which measures 350×56 at
x=20 on a 390px screen.

**The progress dial rewound on pause.** It counted the current segment only
while `speaking`, so pausing — or holding to speak — made the lit ticks retreat.
It now counts segments the walk has *reached* and never moves backwards.

## Verified / not verified

Verified in the browser: idle, live, pause, completion, the full queue through
to the ped-path disclaimer, hold-to-speak, the source sheet, and the
remote-failure fallback (which sets the on-screen `use_native_tts: true`).

All four Component Lego roles were confirmed mounted in the live DOM, not just
imported. While speaking: the magicui `Ripple` at 3 circles, the magicui
`BorderBeam` present, the magicui `AnimatedShinyText` present, and 8 of 32 dial
ticks lit for segment 01/04. On push-to-talk: 3 `PulseRings` and the aceternity
`GlowingEffect` mounted, with the beam correctly gone because the audio stopped.
The source sheet opens as a real `dialog` with all three citations and their
Wikipedia links.

The reveal was measured against real boundary events: `shown` tracked
`boundaries` at 1→3→4→5→6→7→9 over a segment, i.e. ~2.5 words/sec, and the
paragraph's text content held all 29 words while only 9 were visible — the
a11y property above is measured, not assumed. On push-to-talk the reveal
re-targeted the 27-word answer (2 → 21 → 22 → 24 → 26 → 27) and then reset to 0
for the next segment. Pausing showed all 29 words at `blur(0px)` with the beam,
ripple and shimmer all absent.

Also verified against computed styles: the beam's `offsetDistance` steps
(0% → 2.04% → 3.84% → 5.96% → 7.97%) and is mounted only while `speaking`, its
gradient resolves to `rgb(16,185,129) → transparent`; the ticker reads real
elapsed seconds in `white/0.45` JetBrains Mono at 10px; the state word computes
to `white/0.85` over a `white/0.95` band with `background-clip: text` and
`shiny-text 8s`. The de-slop sweep is clean: no `shadow-*` utility anywhere, and
no hex off the palette outside neutral alphas.

The audio transport was verified through its own interface. Driving a real
`AudioQueue` with three generated WAV clips (0.4 / 0.5 / 0.6s) produced strictly
sequential playback — `start:0, end:0, start:1, end:1, start:2, end:2, drained`
— with real timestamps at `+0, +572, +604, +1162, +1193, +1892ms`: each clip
taking roughly its own duration, handoffs of 31–32ms, drain immediately at the
end. One element served all three clips. In the app itself the transport is a
single `<audio data-echoes-transport="queue">`, and the Media Session reported
`title: "Echoes Walking Guide"`, `artist: "Echoes"`, `playbackState: "playing"`.

Push-to-talk was measured through all four states: `aria-pressed` flipping with
RECORDING, 3 pulse rings for capture, 2 orbiting arcs and state word `Thinking`
for UPLOADING, `answering · grounded` for ANSWERING. `lib/mic.ts` was driven
directly and produced a real 12,766-byte `audio/webm;codecs=opus` clip. Orb
visuals were confirmed per state with the transition neutralised: idle
`rgba(250,250,250,0.18)` → recording `rgb(16,185,129)` with glow opacity `1` →
uploading `rgba(16,185,129,0.55)`.

Not verified: the real ElevenLabs/Groq/Gemma calls (no backend attached), the
geolocation permission flow — this browser's location provider returns 403, so
the app took its fallback path instead — and the `text only` branch of the
fallback triage, which is unreachable here because local voices exist.

**A limitation of the measuring environment, not of the app.** This preview
webview never advances CSS *transitions*: `document.getAnimations()` reports the
`CSSTransition` entries as `playState: "running"` with `currentTime` stuck at 0
indefinitely, while CSS *animations* advance normally (211s of page time). So
any property carrying a `transition-*` utility reads as its pre-change value
here — the orb's border, the glow's opacity. That is also why screenshots fail
with "produced no frames". The indicators were therefore verified by
neutralising the transition and re-reading: computed then matched the inline
value exactly in every state. What could **not** be observed is the 150–200ms
interpolation itself.

One real gap: the walk queue speaks ElevenLabs audio, but a push-to-talk
*answer* still goes through `say()` and the browser voice. With a real backend
the answer would be the one sentence in the walk that is not ElevenLabs, and it
should be routed through the same queue.
