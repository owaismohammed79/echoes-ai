import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, MutableRefObject, SetStateAction } from "react";

import {
  type Activity,
  type Fact,
  type Fix,
  type Mode,
  type Phase,
  type Segment,
  type VoicePath,
  type WalkEngine,
  type WalkPrefs,
} from "@/lib/demo-engine";
import {
  AudioQueue,
  getAudioQueue,
  QUEUE_DEPTH,
  type QueueHandlers,
} from "@/lib/audio-queue";
import {
  clearWalkSession,
  publishWalkSession,
  setWalkSessionState,
} from "@/lib/media-session";
import { LlmTemporaryOutageError } from "@/lib/real-engine";
import { useContinuousVad } from "@/hooks/use-continuous-vad";
import { getAudioContext } from "@/lib/audio-context";
import {
  AMBIENT_PAUSE_MS,
  QA_GRACE_MS,
  makeAmbientBed,
  shouldResumeOnVisible,
} from "@/lib/audio-lifecycle";
import { ClipSequencer } from "@/lib/clip-sequencer";

export interface WalkController {
  phase: Phase;
  activity: Activity;
  /** True while the walker is audibly talking — the hands-free barge-in state. */
  listening: boolean;
  /** Live microphone amplitude, 0–1, for the orb. A ref so it animates without re-rendering. */
  levelRef: MutableRefObject<number>;
  /** The walker switched the microphone off. */
  muted: boolean;
  /** Why the last question could not be answered, when it could not. */
  notice: string | null;
  /**
   * The narration cascade is down (`LLM_TEMPORARY_OUTAGE`). Holds the server's
   * own message; null when nothing is wrong. Drives the retry modal.
   */
  outage: string | null;
  /** Try the walk again, with the preferences it failed on. */
  retryOutage: () => void;
  /** Close the outage modal without retrying. */
  dismissOutage: () => void;
  mode: Mode;
  facts: Fact[];
  segments: Segment[];
  index: number;
  timings: Record<string, number> | null;
  fix: Fix | null;
  /** Set when a voice call failed, holding the reason (e.g. tts_rate_limited). */
  fallback: string | null;
  /** Why the microphone could not be used, when it could not. Null when fine. */
  mic: string | null;
  refused: boolean;
  /** True once the queue has been spoken through to the disclaimer. */
  completed: boolean;
  /** Whole seconds since the walk went live. Drives the header ticker. */
  seconds: number;
  /** Whole seconds the screen was on during the walk, for the summary. */
  screenOnSeconds: number;
  setScreenOnSeconds: Dispatch<SetStateAction<number>>;
  /** The sentence the voice is on: a walk segment, or a push-to-talk answer. */
  spokenText: string;
  /** How many words of `spokenText` the voice has actually reached. */
  spoken: number;
  drawer: boolean;
  voicePath: VoicePath;
  setDrawer: Dispatch<SetStateAction<boolean>>;
  setVoicePath: Dispatch<SetStateAction<VoicePath>>;
  setMode: Dispatch<SetStateAction<Mode>>;
  start: (prefs?: WalkPrefs) => void;
  togglePause: () => void;
  /** Switch the microphone off, or back on. */
  toggleMute: () => void;
  endWalk: () => void;
}

/**
 * Reveal rate, in words per second, for engines that fire no word-boundary
 * events (Safari, and several Firefox voices). Deliberately slower than natural
 * speech so the text lags the voice rather than printing words the voice has
 * not reached yet — `onend` then snaps the remainder. It is corrected upwards
 * the moment a real boundary arrives, so a browser that does report boundaries
 * is unaffected.
 */
const REVEAL_FALLBACK_WPS = 2.4;

/**
 * Queue ids are namespaced because the transport carries two kinds of clip. A
 * walk segment moves the walk on; an answer does not, and confusing the two
 * would let a push-to-talk reply advance the walk's index.
 */
const SEGMENT_PREFIX = "segment:";
const ANSWER_ID = "answer";
const TRANSITION_ID = "transition";
const PAUSE_ID = "pause";
const GRACE_ID = "grace";
/** Silence that makes a final cycle that came in short end on the chosen duration. */
const PAD_ID = "pad";
/** A failed continuation is retried after a short gap, this many times, then the walk ends. */
const MAX_CONTINUE_RETRIES = 2;
const RETRY_PAUSE_MS = 20_000;

type AmbientKind = "pause" | "grace" | "pad";

/** Words in `text`: the denominator for the reveal. */
function countWords(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

/** Words of `text` that start at or before `charIndex`, clamped to `total`. */
function wordsUpTo(text: string, charIndex: number, total: number): number {
  if (charIndex <= 0) return 0;
  const before = text.slice(0, charIndex).split(/\s+/).filter(Boolean).length;
  return Math.min(total, before + 1);
}

/**
 * Sound must be unlocked inside the tap handler itself, before any await — a
 * gesture token does not survive an await on iOS. Called synchronously from the
 * Start Walk click, and again from start() ahead of the first network call.
 *
 * This opens the *shared transport element*, never a throwaway: on iOS the right
 * to play belongs to the element that played during the gesture, so an element
 * created later — after the geolocation prompt, after /api/walk returns — would
 * be refused, and the first sentence would arrive silent. The shared
 * AudioContext (lib/audio-context) is resumed here for the same reason, and is
 * the very context the VAD meters the microphone through.
 */
export function primeAudio(): void {
  getAudioContext();
  getAudioQueue().unlock();
}

export function useWalkController(engine: WalkEngine): WalkController {
  const [phase, setPhase] = useState<Phase>("idle");
  const [activity, setActivity] = useState<Activity>("ready");
  const [muted, setMuted] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [outage, setOutage] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("STORY");
  const [facts, setFacts] = useState<Fact[]>([]);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [index, setIndex] = useState(0);
  const [timings, setTimings] = useState<Record<string, number> | null>(null);
  const [fix, setFix] = useState<Fix | null>(null);
  const [fallback, setFallback] = useState<string | null>(null);
  const [refused, setRefused] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [seconds, setSeconds] = useState(0);
  // Screen-on time, accumulated by the ScreenLockGuard while the walk is live.
  // It is the one number on the summary the walk's audio does not produce.
  const [screenOnSeconds, setScreenOnSeconds] = useState(0);
  const [spokenText, setSpokenText] = useState("");
  const [spoken, setSpoken] = useState(0);
  const [drawer, setDrawer] = useState(false);
  // An engine that can serve sentence audio (the real backend) is spoken over
  // the queue by default; an engine without it — the demo — has nothing to
  // queue, so it starts on the browser voice rather than faking a remote path.
  const [voicePath, setVoicePath] = useState<VoicePath>(() =>
    engine.audio ? "remote" : "native"
  );

  const cancelRef = useRef<(() => void) | null>(null);
  /**
   * Bumped every time we stop audio on purpose. Chrome fires an `interrupted`
   * error event for the utterance it cancels, so without a generation counter
   * every pause, every PTT press, and every hand-off between segments looks
   * like a voice failure.
   */
  const genRef = useRef(0);
  const indexRef = useRef(0);
  const segmentsRef = useRef<Segment[]>([]);
  const pausedRef = useRef(false);
  /** The preferences the last walk attempt used, so a Retry repeats the walk. */
  const lastPrefsRef = useRef<WalkPrefs | undefined>(undefined);
  // StrictMode mounts twice; the flag must be reset on mount, not only cleared.
  const mounted = useRef(true);

  /** The one transport. Shared process-wide; this instance owns its handlers. */
  const queueRef = useRef<AudioQueue | null>(null);
  /** The next sentence index to load into the queue. */
  const queueNextRef = useRef(0);
  /** Clips in flight, so the queue is never pumped past QUEUE_DEPTH. */
  const pendingRef = useRef(0);
  /**
   * Holds a fetched clip back until every clip before it has arrived, so the
   * voice speaks sentences in the order the transcript shows them.
   */
  const sequencerRef = useRef(new ClipSequencer<Blob>());
  /**
   * True once this walk is spoken by the browser voice (the server said
   * `use_native_tts`, or the remote voice failed). Sticky for the whole walk:
   * the remote queue must never be driven again, or it would restart speech
   * from a stale index on every cycle.
   */
  const nativeModeRef = useRef(false);
  /** The queue emptied before the next clip arrived; restart it on arrival. */
  const starvedRef = useRef(false);
  /** The bridge line currently on the transport, for the reveal. */
  const transitionTextRef = useRef("");
  /** The answer currently on the transport, for the reveal and the fallback. */
  const answerTextRef = useRef("");
  /**
   * Bumped whenever the queue is emptied on purpose — a barge-in, a resume, an
   * answer. A clip fetched before the bump belongs to a walk state that no
   * longer exists and must not be enqueued behind the new one.
   */
  const fillGenRef = useRef(0);
  /** Mirrors `completed` for the speech callbacks, which outlive any one render. */
  const completedRef = useRef(false);
  /** True while the app's own voice is playing, so the VAD can raise its floor. */
  const playingRef = useRef(false);
  /** Runs when the answer clip ends: bridge, then resume. Refreshed per render. */
  const answerDoneRef = useRef<() => void>(() => {});
  /** What the screen showed before a barge-in, so a noise blip can restore it. */
  const preBargeActivityRef = useRef<Activity>("ready");
  /** Whether the barge-in actually silenced something, so a blip only un-hushes it. */
  const hushedRef = useRef(false);
  /** Ends an answer and puts the walk back on the road. Refreshed per render. */
  const resumeRef = useRef<() => void>(() => {});
  /**
   * Bumped when a walk ends. An answer is fetched over the network, so it can
   * resolve after the walk that asked for it is over; the id is how the late
   * arrival knows to stay quiet.
   */
  const runIdRef = useRef(0);
  /**
   * Whether the whole script has arrived. Until it has, the end of the buffered
   * sentences means "more is coming", not "the walk is over".
   */
  const scriptCompleteRef = useRef(false);
  /** Word count of the sentence on the speaker, for the time-driven reveal. */
  const spokenWordsRef = useRef(0);
  /** Mirrors `activity` for callbacks that must not close over one render. */
  const activityRef = useRef<Activity>("ready");
  /** Mirrors the media-session callbacks; refreshed every render. */
  const sessionRef = useRef({
    play: () => {},
    pause: () => {},
    stop: () => {},
  });
  /** Escape hatch back to the browser voice, kept fresh for the queue's handlers. */
  const nativeFallbackRef = useRef<(reason: string, at?: number) => void>(
    () => {}
  );
  /** The server closed this cycle; play ambient then fetch the next. */
  const cyclePauseRef = useRef(false);
  /** The server said the walk is over. */
  const walkCompleteRef = useRef(false);
  const pauseMsRef = useRef(AMBIENT_PAUSE_MS);
  const graceMsRef = useRef(QA_GRACE_MS);
  const fetchingCycleRef = useRef(false);
  const continueCycleRef = useRef<() => void>(() => {});
  const playAmbientRef = useRef<(kind: AmbientKind, ms?: number) => void>(
    () => {}
  );
  /** The ambient bed on the transport, so a question asked over it can restore it. */
  const ambientRef = useRef<{
    kind: AmbientKind;
    startedAt: number;
    ms: number;
  } | null>(null);
  /** What follows the last buffered sentence: the gap, the pad, the Q&A window, or the end. */
  const finishCycleRef = useRef<() => void>(() => {});
  /** The script is complete: start whatever should follow, if nothing is playing. */
  const settleCycleRef = useRef<(from: number, started: boolean) => void>(
    () => {}
  );
  const padMsRef = useRef(0);
  const retriesRef = useRef(0);
  const stopMicRef = useRef<() => void>(() => {});
  const lastFixRef = useRef<Fix | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      cancelRef.current?.();
      window.speechSynthesis?.cancel();
    };
  }, []);

  /** A live walk is timed on screen. Restarted by phase, not by activity, so
      pausing or holding to speak does not reset the walk's elapsed time. */
  useEffect(() => {
    if (phase !== "live") return;
    const started = Date.now();
    setSeconds(0);
    const id = window.setInterval(() => {
      setSeconds(Math.floor((Date.now() - started) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, [phase]);

  /**
   * Speak one segment. `remote` simulates the ElevenLabs path and fails the way
   * a 429 does; the caller catches that and re-speaks natively, which is the
   * on-screen equivalent of the use_native_tts header.
   */
  const say = useCallback(
    (
      text: string,
      path: VoicePath,
      handlers: { onEnd: () => void; onError: (reason: string) => void }
    ): (() => void) => {
      const synth = window.speechSynthesis;
      const gen = genRef.current;

      /** Ignore anything from an utterance we already replaced or cancelled. */
      const live = () => gen === genRef.current && mounted.current;

      // The transcript follows the voice rather than the walk index, because
      // this one path speaks both walk segments and push-to-talk answers.
      if (mounted.current) {
        setSpokenText(text);
        setSpoken(0);
      }

      const utter = new SpeechSynthesisUtterance(text);
      const voices = synth?.getVoices?.() ?? [];
      utter.voice =
        voices.find((v) => /en-GB/i.test(v.lang)) ??
        voices.find((v) => /^en/i.test(v.lang)) ??
        null;
      utter.rate = 0.98;
      utter.pitch = 1;

      if (path === "native") {
        if (!synth) {
          handlers.onError("speech_synthesis_unavailable");
          return () => {};
        }

        const total = text.split(/\s+/).filter(Boolean).length;
        let estimate: number | null = null;
        /** Once a real boundary lands, the guess is dead weight. */
        let heardBoundary = false;
        const stopEstimate = () => {
          if (estimate !== null) {
            window.clearInterval(estimate);
            estimate = null;
          }
        };

        utter.onstart = () => {
          if (!live()) return;
          const began = Date.now();
          estimate = window.setInterval(() => {
            if (!live() || heardBoundary) {
              stopEstimate();
              return;
            }
            const elapsed = (Date.now() - began) / 1000;
            setSpoken(Math.min(total, Math.floor(elapsed * REVEAL_FALLBACK_WPS)));
          }, 120);
        };

        // The only true sync signal the Web Speech API offers. Chrome reports
        // one per word; sentence boundaries are ignored because rewinding the
        // reveal to the start of a sentence would read as a glitch.
        utter.onboundary = (event) => {
          if (!live()) return;
          const name = (event as SpeechSynthesisEvent & { name?: string }).name;
          if (name && name !== "word") return;
          heardBoundary = true;
          stopEstimate();
          setSpoken(wordsUpTo(text, event.charIndex, total));
        };

        utter.onend = () => {
          stopEstimate();
          if (!live()) return;
          // Whatever the estimator drifted by, the sentence is over.
          setSpoken(total);
          handlers.onEnd();
        };
        utter.onerror = (event) => {
          stopEstimate();
          if (!live()) return;
          const kind = (event as SpeechSynthesisErrorEvent).error;
          // A cancelled utterance is not a broken one.
          if (kind === "interrupted" || kind === "canceled") return;
          handlers.onError(
            kind === "not-allowed" ? "speech_blocked" : `speech_${kind}`
          );
        };
        // No cancel() here: the caller cancels deliberately through
        // invalidate(), and cancelling on every hand-off is what produced the
        // spurious `interrupted` errors.
        synth.speak(utter);
        return () => synth.cancel();
      }

      const timer = setTimeout(() => {
        if (live()) handlers.onError("tts_rate_limited");
      }, 650);
      return () => clearTimeout(timer);
    },
    []
  );

  /** Stop audio on purpose: bump the generation first, then cancel. */
  const invalidate = useCallback(() => {
    genRef.current += 1;
    cancelRef.current?.();
    cancelRef.current = null;
    window.speechSynthesis?.cancel();
  }, []);

  /**
   * Load sentence clips until the queue holds QUEUE_DEPTH — counting the clip on
   * the speaker and the ones still in flight, so a slow network cannot make the
   * transport overshoot its depth. This is the rolling half of the queue.
   */
  const fillQueue = useCallback(
    (queue: AudioQueue) => {
      // A walk on the browser voice has no clips to fetch.
      if (nativeModeRef.current) return;
      const list = segmentsRef.current;
      const fetchAudio = engine.audio;
      if (!fetchAudio) {
        // No audio source at all: this is `use_native_tts: true` on screen.
        nativeFallbackRef.current("tts_unconfigured");
        return;
      }
      const gen = fillGenRef.current;
      const sequencer = sequencerRef.current;
      const buffered = () =>
        queue.depth +
        (queue.busy ? 1 : 0) +
        pendingRef.current +
        sequencer.waiting;
      const enqueueReleased = (released: { index: number; clip: Blob }[]) => {
        for (const { index, clip } of released) {
          queue.enqueue({ id: `${SEGMENT_PREFIX}${index}`, blob: clip });
        }
      };

      while (buffered() < QUEUE_DEPTH && queueNextRef.current < list.length) {
        const at = queueNextRef.current++;
        if (!list[at]?.text) {
          // An empty slot still holds its place in the order.
          enqueueReleased(sequencer.put(at, null));
          continue;
        }
        pendingRef.current += 1;
        void fetchAudio(list[at].text, at)
          .then((blob) => {
            // A barge-in or a restart bumped the generation: this clip belongs
            // to a walk state that no longer exists.
            if (!mounted.current || gen !== fillGenRef.current) return;
            if (!blob) {
              nativeFallbackRef.current("tts_unconfigured", at);
              return;
            }
            // Released in sentence order, not arrival order.
            enqueueReleased(sequencer.put(at, blob));
            // The speaker ran dry while this was in flight: start it again.
            if (starvedRef.current && queue.depth > 0) {
              starvedRef.current = false;
              queue.play();
            }
          })
          .catch(() => {
            if (mounted.current && gen === fillGenRef.current) {
              nativeFallbackRef.current("tts_provider_error", at);
            }
          })
          .finally(() => {
            if (gen === fillGenRef.current) pendingRef.current -= 1;
          });
      }
    },
    [engine]
  );

  /**
   * Bind the shared transport to this walk. The handlers are attached to the
   * queue, not to a render, so everything they touch is a ref.
   */
  const ensureQueue = useCallback((): AudioQueue => {
    const queue = queueRef.current ?? getAudioQueue();
    queueRef.current = queue;

    const handlers: QueueHandlers = {
      onStart: (id) => {
        if (!mounted.current) return;

        if (id === PAUSE_ID || id === GRACE_ID || id === PAD_ID) {
          setActivity("ready");
          return;
        }

        if (id === ANSWER_ID || id === TRANSITION_ID) {
          // An answer and the bridge that follows it are not steps of the walk:
          // they must not touch the index or the completion flag, only what is
          // being said.
          const text =
            id === ANSWER_ID ? answerTextRef.current : transitionTextRef.current;
          setActivity("answering");
          spokenWordsRef.current = countWords(text);
          setSpokenText(text);
          setSpoken(0);
          return;
        }

        const at = Number(id.slice(SEGMENT_PREFIX.length));
        indexRef.current = at;
        setIndex(at);
        setActivity("speaking");
        setCompleted(false);
        const text = segmentsRef.current[at]?.text ?? "";
        spokenWordsRef.current = countWords(text);
        setSpokenText(text);
        setSpoken(0);
      },
      onEnd: (id) => {
        if (!mounted.current) return;
        setSpoken(spokenWordsRef.current);

        if (id === ANSWER_ID) {
          // The answer is over: bridge back into the walk.
          answerDoneRef.current();
          return;
        }
        if (id === TRANSITION_ID) {
          // The bridge is over: the narration it interrupted resumes.
          resumeRef.current();
          return;
        }
        if (id === PAUSE_ID) {
          ambientRef.current = null;
          continueCycleRef.current();
          return;
        }
        if (id === PAD_ID) {
          ambientRef.current = null;
          playAmbientRef.current("grace");
          return;
        }
        if (id === GRACE_ID) {
          ambientRef.current = null;
          stopMicRef.current();
          setActivity("ready");
          setCompleted(true);
          return;
        }

        // The last sentence of a cycle is either "play the ambient bed, then
        // ask the server for the next chunk" or "the walk is over, start the
        // Q&A grace window". The server owns that distinction.
        const at = Number(id.slice(SEGMENT_PREFIX.length));
        const last =
          scriptCompleteRef.current && at >= segmentsRef.current.length - 1;
        if (last) {
          finishCycleRef.current();
          return;
        }
        fillQueue(queue);
      },
      onProgress: (_id, fraction) => {
        if (!mounted.current) return;
        // The element's own clock, so the reveal is tied to the audio that is
        // actually playing rather than to a timer running beside it.
        setSpoken(Math.round(fraction * spokenWordsRef.current));
      },
      onDrained: () => {
        starvedRef.current = true;
      },
      onError: () => {
        if (mounted.current) {
          nativeFallbackRef.current("tts_provider_error", indexRef.current);
        }
      },
    };

    queue.attach(handlers);
    return queue;
  }, [fillQueue]);

  const playFrom = useCallback(
    (startIndex: number, path: VoicePath, attempt = 0) => {
      const list = segmentsRef.current;
      if (!mounted.current) return;
      // Once the walk is on the browser voice it stays there.
      if (nativeModeRef.current && path === "remote") path = "native";
      // Whatever was in flight for the previous playback is void now.
      fillGenRef.current += 1;
      if (startIndex >= list.length) {
        // Hold the transcript on the last thing actually said. The server, not
        // this branch, decides whether that is a pause or the end of the walk.
        indexRef.current = Math.max(0, list.length - 1);
        setIndex(indexRef.current);
        if (!scriptCompleteRef.current) {
          setActivity("ready");
          return;
        }
        finishCycleRef.current();
        return;
      }
      if (path === "remote" && attempt === 0 && engine.audio) {
        // The rolling queue: 2–3 sentence clips buffered ahead, played one at a
        // time through the single reused element.
        const queue = ensureQueue();
        queue.clear();
        queueNextRef.current = startIndex;
        sequencerRef.current.reset(startIndex);
        pendingRef.current = 0;
        starvedRef.current = false;
        indexRef.current = startIndex;
        setIndex(startIndex);
        setActivity("speaking");
        setCompleted(false);
        fillQueue(queue);
        // Nothing is buffered yet; `onStart` fires when the first clip sounds.
        queue.play();
        return;
      }

      indexRef.current = startIndex;
      setIndex(startIndex);
      setActivity("speaking");
      setCompleted(false);

      cancelRef.current = say(list[startIndex].text, path, {
        onEnd: () => {
          if (!mounted.current || pausedRef.current) return;
          playFrom(startIndex + 1, path, 0);
        },
        onError: (reason) => {
          if (!mounted.current) return;
          setFallback(reason);
          if (attempt === 0 && path === "remote") {
            playFrom(startIndex, "native", 1);
            return;
          }
          setActivity("ready");
        },
      });
    },
    [engine.audio, ensureQueue, fillQueue, say]
  );

  /**
   * Speak the bridge between an answer and the narration it interrupted, then
   * resume. The line is the server's, not the client's, so this cannot become a
   * free text-to-speech call; when the remote voice has no audio for it, the
   * browser speaks it and the walk resumes either way.
   */
  const bridgeAndResume = useCallback(async () => {
    if (!mounted.current) return;
    const transition = engine.transition;
    if (!transition) {
      resumeRef.current();
      return;
    }
    let line: { text: string; audio: Blob | null };
    try {
      line = await transition();
    } catch {
      if (mounted.current) resumeRef.current();
      return;
    }
    if (!mounted.current) return;

    transitionTextRef.current = line.text;
    if (!line.audio) {
      // No remote voice for the bridge: the browser speaks it, then the walk.
      cancelRef.current = say(line.text, "native", {
        onEnd: () => resumeRef.current(),
        onError: () => resumeRef.current(),
      });
      return;
    }
    const queue = ensureQueue();
    queue.enqueue({ id: TRANSITION_ID, blob: line.audio });
    queue.play();
  }, [engine, ensureQueue, say]);

  // Refreshed every render: the queue's handlers outlive any single one of them,
  // and they must always reach the walk's *current* position and callbacks.
  useEffect(() => {
    nativeFallbackRef.current = (reason: string, at?: number) => {
      nativeModeRef.current = true;
      queueRef.current?.clear();
      if (!mounted.current) return;
      setFallback(reason);
      // Re-speak the sentence that failed, and everything after it, through the
      // browser voice: the on-screen form of `use_native_tts: true`. ``at`` is
      // the sentence that failed. The last one *heard* (``indexRef``) is only
      // right when that is the same sentence, and on a continuation cycle it is
      // the previous cycle's last line -- which would be spoken twice.
      playFrom(at ?? indexRef.current, "native", 1);
    };

    // The transition has played (or there was none): put the narration back.
    resumeRef.current = () => {
      if (!mounted.current) return;
      pausedRef.current = false;
      // A walk that finished before the question was asked has nothing to
      // resume into — replaying its last sentence would be a loop, not a resume.
      if (completedRef.current) {
        setActivity("ready");
        return;
      }
      // A question asked over an ambient gap resumes the gap, with what was
      // left of it: replaying the previous sentence would be a repeat.
      const ambient = ambientRef.current;
      if (ambient) {
        const left = ambient.ms - (Date.now() - ambient.startedAt);
        playAmbientRef.current(ambient.kind, Math.max(5_000, left));
        return;
      }
      if (walkCompleteRef.current) {
        playAmbientRef.current("grace");
        return;
      }
      // Back on the road, in whatever voice is currently working.
      playFrom(indexRef.current, fallback ? "native" : voicePath);
    };

    // The answer has finished: bridge back into the walk.
    answerDoneRef.current = () => {
      if (!mounted.current) return;
      if (walkCompleteRef.current) {
        playAmbientRef.current("grace");
        return;
      }
      void bridgeAndResume();
    };

    playAmbientRef.current = (kind, msOverride) => {
      if (!mounted.current) return;
      const queue = ensureQueue();
      const ms =
        msOverride ??
        (kind === "grace"
          ? graceMsRef.current
          : kind === "pad"
            ? padMsRef.current
            : pauseMsRef.current);
      ambientRef.current = { kind, startedAt: Date.now(), ms };
      queue.enqueue({
        id: kind === "grace" ? GRACE_ID : kind === "pad" ? PAD_ID : PAUSE_ID,
        blob: makeAmbientBed(ms),
      });
      setActivity("ready");
      queue.play();
    };

    // What follows the last sentence of a cycle. The server decided which: the
    // walk is over (pad if it ended short, then the Q&A window), or there is a
    // gap and then another cycle.
    finishCycleRef.current = () => {
      if (!mounted.current) return;
      if (walkCompleteRef.current) {
        playAmbientRef.current(padMsRef.current > 0 ? "pad" : "grace");
        return;
      }
      if (cyclePauseRef.current) {
        playAmbientRef.current("pause");
        return;
      }
      setActivity("ready");
      setCompleted(true);
    };

    // The whole script for this cycle has arrived. If everything buffered has
    // already been spoken the end-of-clip handlers will never fire again, so
    // whatever comes next has to be started from here.
    settleCycleRef.current = (from, started) => {
      if (!mounted.current || pausedRef.current) return;
      if (nativeModeRef.current) {
        if (activityRef.current === "ready") {
          playFrom(started ? indexRef.current + 1 : from, "native");
        }
        return;
      }
      const queue = ensureQueue();
      fillQueue(queue);
      const idle =
        !queue.busy &&
        queue.depth === 0 &&
        pendingRef.current === 0 &&
        sequencerRef.current.waiting === 0 &&
        queueNextRef.current >= segmentsRef.current.length;
      if (idle) finishCycleRef.current();
    };

    continueCycleRef.current = () => {
      if (!mounted.current || fetchingCycleRef.current || walkCompleteRef.current) {
        return;
      }
      fetchingCycleRef.current = true;
      const run = runIdRef.current;
      void (async () => {
        const located = (await requestFix()) ?? lastFixRef.current;
        if (!mounted.current || run !== runIdRef.current) {
          fetchingCycleRef.current = false;
          return;
        }
        if (!located) {
          fetchingCycleRef.current = false;
          walkCompleteRef.current = true;
          playAmbientRef.current("grace");
          return;
        }
        lastFixRef.current = located;
        setFix(located);

        // A new cycle starts here. Its script is still arriving, so the end of
        // what is buffered means "more is coming", not "the cycle is over" --
        // the flags left over from the previous cycle would say otherwise.
        const from = segmentsRef.current.length;
        scriptCompleteRef.current = false;
        cyclePauseRef.current = false;
        fillGenRef.current += 1;
        pendingRef.current = 0;
        queueNextRef.current = from;
        sequencerRef.current.reset(from);
        starvedRef.current = true;
        let started = false;

        try {
          const narration = await engine.narrate({
            mode,
            fix: located,
            prefs: lastPrefsRef.current,
            continuation: true,
            onPartial: (partial) => {
              if (!mounted.current || run !== runIdRef.current) return;
              segmentsRef.current = partial.segments;
              setSegments(partial.segments);
              setFacts(partial.facts);
              if (partial.remoteAudio === false) nativeModeRef.current = true;
              if (partial.segments.length <= from) return;
              if (nativeModeRef.current) {
                // Speak the cycle's first sentence now; the rest follow as the
                // end of each one asks for the next.
                if (!started) {
                  started = true;
                  playFrom(from, "native");
                }
                return;
              }
              started = true;
              fillQueue(ensureQueue());
            },
          });
          if (!mounted.current || run !== runIdRef.current) return;
          segmentsRef.current = narration.segments;
          setSegments(narration.segments);
          setFacts(narration.facts);
          walkCompleteRef.current = narration.complete === true;
          cyclePauseRef.current = narration.complete !== true;
          if (narration.pauseMs) pauseMsRef.current = narration.pauseMs;
          if (narration.graceMs) graceMsRef.current = narration.graceMs;
          padMsRef.current = narration.padMs ?? 0;
          retriesRef.current = 0;
          scriptCompleteRef.current = true;
        } catch {
          if (!mounted.current || run !== runIdRef.current) return;
          scriptCompleteRef.current = true;
          retriesRef.current += 1;
          if (retriesRef.current > MAX_CONTINUE_RETRIES) {
            // The server is not coming back: end the walk properly rather than
            // leaving the walker in silence.
            walkCompleteRef.current = true;
            padMsRef.current = 0;
          } else {
            // A hiccup, not the end: another short gap, then ask again.
            cyclePauseRef.current = true;
            pauseMsRef.current = RETRY_PAUSE_MS;
          }
        } finally {
          fetchingCycleRef.current = false;
        }
        if (!mounted.current || run !== runIdRef.current) return;
        settleCycleRef.current(from, started);
      })();
    };

  });

  /** Ask for a fix, once, after the tap. Rounded to 3dp and never stored. */
  const requestFix = useCallback(async (): Promise<Fix | null> => {
    if (!navigator.geolocation) return null;
    return new Promise<Fix | null>((resolve) => {
      const timer = window.setTimeout(() => resolve(null), 4000);
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          window.clearTimeout(timer);
          resolve({
            lat: Number(pos.coords.latitude.toFixed(3)),
            lon: Number(pos.coords.longitude.toFixed(3)),
          });
        },
        () => {
          window.clearTimeout(timer);
          resolve(null);
        },
        { timeout: 3500, maximumAge: 60000, enableHighAccuracy: false }
      );
    });
  }, []);

  /**
   * The answer in the browser's voice, then straight back to the walk. Shares
   * :data:`resumeRef` with the queued path so both endings are identical.
   */
  const speakAnswerNatively = useCallback(
    (text: string) => {
      if (!mounted.current) return;
      setActivity("answering");
      cancelRef.current = say(text, "native", {
        onEnd: () => answerDoneRef.current(),
        onError: (reason) => {
          if (!mounted.current) return;
          setFallback(reason);
          answerDoneRef.current();
        },
      });
    },
    [say]
  );

  /**
   * The answer over the walk's own transport.
   *
   * Push-to-talk used to bypass the queue and speak through the browser, which
   * made the reply the one utterance in the walk that was not the remote voice,
   * and gave iOS a second audio path with its own unlock requirements. Queued,
   * it inherits the element that was unlocked in the tap handler, the reveal
   * clock, and the same failure handling as every other sentence.
   */
  const playAnswer = useCallback(
    (text: string) => {
      answerTextRef.current = text;
      const audio = engine.audio;
      const queue = ensureQueue();
      if (!audio) {
        // No remote voice configured at all: nothing to route through.
        speakAnswerNatively(text);
        return;
      }

      // Whatever the walk was saying is dropped rather than left under the
      // answer: one transport, one voice. The walk restarts that segment when
      // the answer ends.
      queue.clear();
      pendingRef.current = 0;
      starvedRef.current = false;
      const run = runIdRef.current;

      const native = (reason: string | null) => {
        if (!mounted.current || run !== runIdRef.current) return;
        if (reason) setFallback(reason);
        speakAnswerNatively(text);
      };

      // -1: an answer is not a step of the walk, so it has no segment index.
      void audio(text, -1)
        .then((blob) => {
          if (!mounted.current || run !== runIdRef.current) return;
          // A missing clip is the provider being unconfigured, not a failure —
          // the same distinction the walk's prefetch makes.
          if (!blob) return native("tts_unconfigured");
          queue.enqueue({ id: ANSWER_ID, blob });
          queue.play();
        })
        .catch(() => native("tts_provider_error"));
    },
    [engine.audio, ensureQueue, speakAnswerNatively]
  );

  /** Silence playback now, but hold its position: a false trigger is undoable. */
  const hushPlayback = useCallback(() => {
    queueRef.current?.pause();
    window.speechSynthesis?.pause();
  }, []);

  /** Undo a hush: the narration continues exactly where it stopped. */
  const unhushPlayback = useCallback(() => {
    if (queueRef.current?.busy) queueRef.current.play();
    window.speechSynthesis?.resume();
  }, []);

  /**
   * A finished utterance becomes a turn: transcribe, answer, speak. The walk is
   * held underneath it; the answer's own end runs the bridge and the resume.
   */
  const submitUtterance = useCallback(
    (clip: Blob) => {
      if (!mounted.current) return;
      // The utterance is real: cut the hushed narration for good, and void every
      // clip still being fetched for it.
      invalidate();
      queueRef.current?.interrupt();
      fillGenRef.current += 1;
      pendingRef.current = 0;
      starvedRef.current = false;

      const run = runIdRef.current;
      setNotice(null);
      setActivity("thinking");
      void (async () => {
        try {
          const question = await engine.transcribe(clip);
          const answer = await engine.answer({ question, facts, mode });
          // A walk that ended while the answer was in flight must not be spoken
          // into: the run id is how the late arrival knows to stay quiet.
          if (!mounted.current || run !== runIdRef.current) return;

          setRefused(answer.refused);
          setTimings((prev) => ({ ...(prev ?? {}), ...answer.timings }));
          setActivity("answering");

          // The same transport as the walk, so the answer is spoken by the walk's
          // voice rather than by a second, browser-owned one.
          if (voicePath === "remote" && !fallback) {
            playAnswer(answer.text);
          } else {
            speakAnswerNatively(answer.text);
          }
        } catch {
          // The turn failed — no network, a 503, a provider. Say so and put the
          // walk back, rather than leaving it stuck on "thinking" forever.
          if (!mounted.current || run !== runIdRef.current) return;
          setNotice("question_failed");
          resumeRef.current();
        }
      })();
    },
    [
      engine,
      facts,
      fallback,
      invalidate,
      mode,
      playAnswer,
      speakAnswerNatively,
      voicePath,
    ]
  );

  /**
   * The walker started talking. Silence whatever is playing *now* — the
   * narration or a previous answer — but keep its position, so a noise blip can
   * be undone without replaying the sentence it interrupted. This is the
   * barge-in edge the VAD fires on the first syllable.
   */
  const handleSpeechStart = useCallback(() => {
    if (phase !== "live") return;
    const current = activityRef.current;
    // A turn is already in flight, or we are already listening: nothing to cut.
    if (current === "thinking" || current === "listening") return;

    // Only silence something that is actually playing. A walk the user paused
    // has nothing to hush — and un-hushing it later would start it again.
    const wasPlaying = current === "speaking" || current === "answering";
    if (wasPlaying) hushPlayback();
    hushedRef.current = wasPlaying;
    preBargeActivityRef.current = current;
    pausedRef.current = true;
    setActivity("listening");
  }, [hushPlayback, phase]);

  const handleSpeechEnd = useCallback(
    (clip: Blob | null) => {
      if (!mounted.current || phase !== "live") return;
      // Only an utterance that began with a barge-in — which put us in
      // "listening" — is handled. Speech that arrived while a turn was already
      // in flight is ignored, so a second question cannot race the first.
      if (activityRef.current !== "listening") return;

      if (!clip) {
        // Noise, not a question: resume exactly where the narration paused, so
        // nothing is repeated. If nothing was playing, leave it alone.
        if (hushedRef.current) {
          pausedRef.current = false;
          unhushPlayback();
        }
        hushedRef.current = false;
        setActivity(preBargeActivityRef.current);
        return;
      }
      hushedRef.current = false;
      submitUtterance(clip);
    },
    [phase, submitUtterance, unhushPlayback]
  );

  // The hands-free loop. Opened by the Start tap (and by unmuting), closed by
  // endWalk or muting; the callbacks are read fresh on every tick.
  const vad = useContinuousVad(
    { onSpeechStart: handleSpeechStart, onSpeechEnd: handleSpeechEnd },
    playingRef
  );

  useEffect(() => {
    stopMicRef.current = () => vad.stop();
  });

  const start = useCallback((prefs?: WalkPrefs) => {
    // Synchronous, before the geolocation prompt and before /api/walk: this is
    // the gesture that buys the right to play on iOS.
    primeAudio();
    // The microphone opens here too, inside the same tap: a stream acquired
    // after the gesture is refused on iOS, and this walk is hands-free from the
    // first second.
    void vad.start();
    // Every walk gets its own run id, so a clip fetched for the last one cannot
    // speak into this one.
    runIdRef.current += 1;
    lastPrefsRef.current = prefs;
    // A fresh walk starts with a clean playback state. It speaks through the
    // browser voice from the outset only when the engine has no audio at all.
    nativeModeRef.current = !engine.audio;
    sequencerRef.current.reset(0);
    ambientRef.current = null;
    padMsRef.current = 0;
    retriesRef.current = 0;
    setPhase("arming");
    setDrawer(false);
    setScreenOnSeconds(0);
    setRefused(false);
    setCompleted(false);
    setFallback(null);
    setNotice(null);
    setOutage(null);

    void (async () => {
      // The device fix and the server-side locate are alternatives, not a race.
      // Running them concurrently looks like a saving and is not: the server's
      // fix is an approximation, so the facts it would buy describe somewhere
      // the walker may never have been — and a wrong-place walk is the one
      // failure the validator cannot catch, because the citations are real. The
      // device is therefore asked first and the server only answers for it.
      // A walk with no fix has no place to describe, so a locate that fails
      // returns to the landing screen rather than arming forever.
      let located: Fix | null = null;
      try {
        located = (await requestFix()) ?? (await engine.locate(prefs));
      } catch {
        located = null;
      }
      if (!mounted.current) return;
      if (!located) {
        setPhase("idle");
        return;
      }
      setFix(located);
      lastFixRef.current = located;
      scriptCompleteRef.current = false;
      walkCompleteRef.current = false;
      cyclePauseRef.current = false;
      fetchingCycleRef.current = false;

      // Set the first time there are sentences to speak, which for a streaming
      // backend is well before the script is finished.
      let live = false;
      const goLive = (
        narrationFacts: Fact[],
        narrationSegments: Segment[],
        narrationMode: Mode,
        path: VoicePath,
        reminderText?: string
      ) => {
        // The reminder is the server's line, appended after validation. It is
        // only added once the whole script is in: seating it mid-script would
        // shift every sentence that arrives after it, and the queue's
        // forward-only cursor would then skip them. The text comes from the
        // backend, never a hardcoded client copy.
        const alreadyHasReminder = narrationSegments.some(
          (segment) =>
            segment.role === "reminder" ||
            (reminderText ? segment.text === reminderText : false)
        );
        const list =
          scriptCompleteRef.current && reminderText && !alreadyHasReminder
            ? [
                ...narrationSegments,
                { text: reminderText, citation: null, role: "reminder" as const },
              ]
            : [...narrationSegments];
        live = true;
        setMode(narrationMode);
        setFacts(narrationFacts);
        setSegments(list);
        segmentsRef.current = list;
        setPhase("live");
        // The lock screen is part of the interface once a walk is running.
        // Registered once, dispatching through a ref so it always reaches the
        // current callbacks rather than the ones from this render.
        publishWalkSession({
          onPlay: () => sessionRef.current.play(),
          onPause: () => sessionRef.current.pause(),
          onStop: () => sessionRef.current.stop(),
        });
        // A refusal with no facts opens the drawer: the honesty is the product.
        setDrawer(narrationFacts.length === 0);
        // A clip is synthesized from its sentence, so the first sentence is the
        // earliest thing there is to fetch. The first two are requested together
        // and playback starts on whichever lands first, rather than waiting for
        // the script to end.
        playFrom(0, path);
      };

      const narration = await engine
        .narrate({
          mode,
          fix: located,
          prefs,
          onPartial: (partial) => {
            if (!mounted.current || live) return;
            // A walk the backend cannot speak remotely starts on the browser voice
            // rather than fetching audio that will not come and calling it a
            // failure. This is the normal local mode (ECHOES_FORCE_NATIVE_TTS).
            const path: VoicePath =
              partial.remoteAudio === false ? "native" : voicePath;
            if (path !== voicePath) setVoicePath(path);
            if (path === "native") nativeModeRef.current = true;
            goLive(partial.facts, partial.segments, partial.mode, path);
          },
        })
        .catch((err: unknown) => {
          if (!mounted.current) return null;
          if (err instanceof LlmTemporaryOutageError) {
            // The cascade spent its budget before streaming a frame: nothing was
            // said, so this is a pause, not a failure. The modal offers Retry.
            setOutage(err.message);
            setPhase("idle");
            return null;
          }
          throw err;
        });
      if (!mounted.current || narration === null) return;

      setTimings(narration.timings);
      setMode(narration.mode);
      walkCompleteRef.current = narration.complete === true;
      cyclePauseRef.current = narration.complete !== true;
      if (narration.pauseMs) pauseMsRef.current = narration.pauseMs;
      if (narration.graceMs) graceMsRef.current = narration.graceMs;
      padMsRef.current = narration.padMs ?? 0;
      if (live) {
        // The rest of the script arrived while the first sentences were already
        // being spoken. The transcript gains the sentences still to come, and
        // the queue is refilled because it may have run dry waiting for them.
        const alreadyHasReminder = narration.segments.some(
          (segment) =>
            segment.role === "reminder" ||
            (narration.reminder ? segment.text === narration.reminder : false)
        );
        const list =
          narration.reminder && !alreadyHasReminder
            ? [
                ...narration.segments,
                {
                  text: narration.reminder,
                  citation: null,
                  role: "reminder" as const,
                },
              ]
            : [...narration.segments];
        segmentsRef.current = list;
        setSegments(list);
        setFacts(narration.facts);
        scriptCompleteRef.current = true;
        // The rest of the script is in. If everything buffered has already been
        // spoken, nothing else will fire, so start what follows from here.
        settleCycleRef.current(0, true);
        return;
      }

      scriptCompleteRef.current = true;
      goLive(
        narration.facts,
        narration.segments,
        narration.mode,
        voicePath,
        narration.reminder
      );
    })();
  }, [engine, fallback, fillQueue, mode, playFrom, requestFix, voicePath]);

  /** Close the outage modal; the walker chose not to retry. */
  const dismissOutage = useCallback(() => setOutage(null), []);

  /** Retry the walk that the outage interrupted, with the same preferences. */
  const retryOutage = useCallback(() => {
    setOutage(null);
    start(lastPrefsRef.current);
  }, [start]);

  /**
   * Registered once with the Media Session, dispatched every time. A lock-screen
   * button pressed an hour into a walk must reach the walk as it is now.
   */
  useEffect(() => {
    sessionRef.current = {
      play: () => {
        if (activityRef.current === "paused") togglePause();
      },
      pause: () => {
        if (activityRef.current === "speaking") togglePause();
      },
      stop: () => endWalk(),
    };
  });

  useEffect(() => {
    activityRef.current = activity;
    // The VAD reads this to decide whether to raise its onset floor: while the
    // app is talking, the microphone is hearing it too.
    playingRef.current = activity === "speaking" || activity === "answering";
    if (phase === "live") {
      setWalkSessionState(activity === "paused" ? "paused" : "playing");
    }
  }, [activity, phase]);

  useEffect(() => {
    completedRef.current = completed;
  }, [completed]);

  useEffect(() => {
    if (phase !== "live") return;
    const onVisible = () => {
      const el = queueRef.current?.element_ ?? null;
      if (
        !shouldResumeOnVisible({
          visibilityState: document.visibilityState,
          walkActive: true,
          userPaused: pausedRef.current,
          elementPaused: el?.paused ?? false,
        })
      ) {
        return;
      }
      queueRef.current?.resumeIfPaused();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, [phase]);

  const togglePause = useCallback(() => {
    if (phase !== "live") return;
    if (activity === "speaking") {
      pausedRef.current = true;
      invalidate();
      // A queue pauses in place, so resuming picks the sentence up mid-word
      // rather than repeating it — which the browser voice cannot do.
      queueRef.current?.pause();
      setActivity("paused");
      return;
    }
    if (activity === "paused") {
      pausedRef.current = false;
      if (queueRef.current?.busy) {
        queueRef.current.play();
        setActivity("speaking");
        return;
      }
      playFrom(indexRef.current, fallback ? "native" : voicePath);
    }
  }, [activity, fallback, phase, playFrom, voicePath]);

  /** Leave the walk and return to the resting state. */
  const endWalk = useCallback(() => {
    invalidate();
    runIdRef.current += 1;
    scriptCompleteRef.current = false;
    completedRef.current = false;
    walkCompleteRef.current = false;
    cyclePauseRef.current = false;
    fetchingCycleRef.current = false;
    ambientRef.current = null;
    nativeModeRef.current = false;
    padMsRef.current = 0;
    retriesRef.current = 0;
    sequencerRef.current.reset(0);
    queueRef.current?.clear();
    // The microphone goes with the walk: a hands-free loop that keeps listening
    // after the walk ends is a hot mic.
    vad.stop();
    clearWalkSession();
    pausedRef.current = false;
    segmentsRef.current = [];
    indexRef.current = 0;
    setSegments([]);
    setIndex(0);
    setFacts([]);
    setTimings(null);
    setFallback(null);
    setRefused(false);
    setCompleted(false);
    setDrawer(false);
    setNotice(null);
    setActivity("ready");
    setPhase("idle");
    setSpokenText("");
    setSpoken(0);
    setScreenOnSeconds(0);
  }, [invalidate, vad]);

  /**
   * The privacy switch. Muting releases the microphone outright — the hardware
   * light goes out — rather than merely ignoring it. Unmuting is a tap, so the
   * stream is re-acquired inside that gesture, the only moment iOS grants it.
   */
  const toggleMute = useCallback(() => {
    if (muted) {
      void vad.start();
      setMuted(false);
    } else {
      vad.stop();
      setMuted(true);
    }
  }, [muted, vad]);

  return useMemo(
    () => ({
      phase,
      activity,
      mode,
      facts,
      segments,
      index,
      timings,
      fix,
      fallback,
      refused,
      completed,
      seconds,
      screenOnSeconds,
      spokenText,
      spoken,
      mic: vad.error,
      listening: vad.speaking,
      levelRef: vad.levelRef,
      muted,
      notice,
      outage,
      drawer,
      voicePath,
      setDrawer,
      setVoicePath,
      setMode,
      setScreenOnSeconds,
      start,
      retryOutage,
      dismissOutage,
      togglePause,
      toggleMute,
      endWalk,
    }),
    [
      activity,
      completed,
      drawer,
      seconds,
      screenOnSeconds,
      spokenText,
      spoken,
      endWalk,
      facts,
      fallback,
      fix,
      index,
      mode,
      muted,
      notice,
      outage,
      phase,
      refused,
      segments,
      start,
      retryOutage,
      dismissOutage,
      timings,
      toggleMute,
      togglePause,
      vad.error,
      vad.levelRef,
      vad.speaking,
      voicePath,
    ]
  );
}
