/**
 * Continuous voice activity detection.
 *
 * The hands-free replacement for hold-to-talk: one microphone stream, held open
 * for the length of the walk, sampled through a Web Audio `AnalyserNode` so the
 * app can tell the walker's voice from the street *without a library*. hark and
 * friends are a few hundred lines of exactly this loop; the loop is small enough
 * to own, and owning it means the threshold, the silence window and the barge-in
 * latency are ours to tune rather than a dependency's.
 *
 * The decision itself is not here — it is in `lib/vad.ts`, a pure state machine
 * over RMS samples, so it can be tested against a scripted signal. This hook is
 * the browser half: acquire the microphone, meter it, record it, and hand the
 * boundaries to the caller.
 *
 * Why a rolling `MediaRecorder` rather than one started at the speech edge: a
 * recorder takes tens of milliseconds to spin up, so starting it *after* the
 * voice is detected clips the first syllable — the "W" in "Wait, what's around
 * here?". Keeping it running and retaining a few hundred milliseconds of chunks
 * ahead of the edge means the submitted clip contains the whole word. Each turn
 * resets the buffer back to its WebM header, so a long walk does not accumulate
 * audio.
 *
 * The analyser is a level meter, not a tap: capture is a separate
 * `MediaRecorder` over the same stream, which is why both exist.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { MutableRefObject } from "react";

import { VoiceActivityDetector, type VadConfig } from "@/lib/vad";
import { getAudioContext } from "@/lib/audio-context";
import { stopAllTracks } from "@/lib/audio-lifecycle";

/** RMS above which the microphone is considered to be carrying a voice. */
export const SILENCE_RMS = 0.015;

/** How long the voice must be absent before the utterance is submitted. */
export const SILENCE_MS = 1200;

/** A blip shorter than this is a cough, a door, or the wind — not a question. */
export const MIN_SPEECH_MS = 250;

/** Consecutive above-threshold ticks before speech is declared (~150ms). */
const ONSET_FRAMES = 3;

/** The analyser poll cadence. Barge-in lands within about one tick. */
const TICK_MS = 50;

/** One `MediaRecorder` timeslice: the unit of pre-roll and of utterance audio. */
const CHUNK_MS = 200;

/** Chunks retained ahead of the speech edge, so the first syllable survives. */
const PREROLL_CHUNKS = 4;

/** RMS that fills the orb. A raised voice at arm's length is about 0.15. */
const LEVEL_FULL_SCALE = 0.15;

/**
 * While the app's own voice is playing, the microphone hears it too. Echo
 * cancellation removes most of it, but not all on every device, so the onset
 * threshold is raised for the duration — a barge-in still lands, but the app
 * talking to itself does not.
 */
const PLAYBACK_RMS_FACTOR = 1.8;

/**
 * How far above the measured ambient level a voice must be. A quiet room stays
 * governed by `SILENCE_RMS`; a noisy street lifts the bar, so traffic and wind
 * do not barge the walk in.
 */
const NOISE_MULTIPLIER = 2;

/** Ticks spent learning the room before detection starts (~400ms). */
const CALIBRATION_TICKS = 8;

/** The detector's tuning, in one place so the tests and the hook agree. */
export const VAD_CONFIG: VadConfig = {
  silenceRms: SILENCE_RMS,
  noiseMultiplier: NOISE_MULTIPLIER,
  silenceMs: SILENCE_MS,
  minSpeechMs: MIN_SPEECH_MS,
  onsetFrames: ONSET_FRAMES,
  calibrationTicks: CALIBRATION_TICKS,
};

/** WebM/Opus first; the fallbacks keep a Safari that lacks it working. */
const MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
];

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  return MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported?.(type));
}

export interface VadHandlers {
  /** The walker began speaking. The barge-in edge. */
  onSpeechStart: () => void;
  /** The walker stopped. `clip` is the trimmed utterance, or null if too short. */
  onSpeechEnd: (clip: Blob | null) => void;
}

export interface ContinuousVad {
  /**
   * Live microphone amplitude, 0–1. A ref, not state, on purpose: the orb
   * animates at frame rate from this without re-rendering the whole app 20
   * times a second.
   */
  levelRef: MutableRefObject<number>;
  /** True while the walker is audibly talking. Changes rarely, so it is state. */
  speaking: boolean;
  /** Why the microphone could not be used, when it could not. */
  error: string | null;
  /** Acquire the microphone and start listening. Call inside a user gesture. */
  start: () => Promise<void>;
  /** Release the microphone. The hardware light goes out. */
  stop: () => void;
}

export function useContinuousVad(
  handlers: VadHandlers,
  /** Read each tick: true while the app's own voice is playing. */
  playingRef: MutableRefObject<boolean>
): ContinuousVad {
  const [speaking, setSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const levelRef = useRef(0);
  /** The latest handlers, so the running loop never closes over a stale render. */
  const handlersRef = useRef(handlers);
  useEffect(() => {
    handlersRef.current = handlers;
  });

  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const bufferRef = useRef<Float32Array<ArrayBuffer> | null>(null);
  const timerRef = useRef<number | null>(null);

  /** The decision state machine, reset at the start of every listen. */
  const detectorRef = useRef<VoiceActivityDetector | null>(null);

  /** Audio chunks since the last turn, [0] always the WebM header. */
  const rollingRef = useRef<Blob[]>([]);
  /** Index into `rolling` where this utterance's pre-roll begins. */
  const captureFromRef = useRef(0);

  const stop = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    try {
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
    } catch {
      /* already stopped */
    }
    recorderRef.current = null;

    sourceRef.current?.disconnect();
    sourceRef.current = null;
    analyserRef.current = null;
    bufferRef.current = null;

    // The light must go out on every path: a muted or ended walk that leaves the
    // microphone open is a privacy bug, not a state. Never close the shared
    // AudioContext — that would throw away the iOS unlock for the next walk.
    stopAllTracks(streamRef.current);
    streamRef.current = null;
    ctxRef.current = null;

    rollingRef.current = [];
    detectorRef.current = null;
    levelRef.current = 0;
    setSpeaking(false);
  }, []);

  /** The utterance recorded so far, trimmed of leading and trailing silence. */
  const assemble = useCallback((): Blob | null => {
    const rolling = rollingRef.current;
    const header = rolling[0];
    // [0] is the WebM header and is required for any later cluster to decode;
    // everything before the pre-roll mark is dropped as silence.
    const parts = header
      ? [header, ...rolling.slice(captureFromRef.current)]
      : [...rolling];
    // Reset to the header alone so a long walk never accumulates audio.
    rollingRef.current = header ? [header] : [];
    if (!parts.length) return null;
    return new Blob(parts, { type: header?.type || "audio/webm" });
  }, []);

  const start = useCallback(async () => {
    if (streamRef.current) return; // already listening

    const media = navigator.mediaDevices;
    if (!media?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("mic_unsupported");
      return;
    }

    let stream: MediaStream;
    try {
      stream = await media.getUserMedia({
        audio: {
          // The browser's own cancellation is the first line of defence against
          // hearing the narration; the raised threshold above is the second.
          echoCancellation: true,
          noiseSuppression: true,
          // AGC would fight the RMS threshold by lifting quiet rooms to it.
          autoGainControl: false,
        },
      });
    } catch (err) {
      const name = (err as { name?: string }).name;
      setError(
        name === "NotAllowedError" || name === "SecurityError"
          ? "mic_denied"
          : "mic_unavailable"
      );
      return;
    }

    // The app's one context, already resumed by the Start tap's primeAudio().
    // A context created here — after getUserMedia's await, outside the gesture —
    // would stay suspended on iOS and meter silence forever.
    const ctx = getAudioContext();
    if (!ctx) {
      stream.getTracks().forEach((track) => track.stop());
      setError("mic_unsupported");
      return;
    }

    streamRef.current = stream;
    setError(null);
    detectorRef.current = new VoiceActivityDetector(VAD_CONFIG);

    ctxRef.current = ctx;
    const source = ctx.createMediaStreamSource(stream);
    sourceRef.current = source;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.4;
    source.connect(analyser);
    analyserRef.current = analyser;
    bufferRef.current = new Float32Array(analyser.fftSize);

    rollingRef.current = [];
    const mimeType = pickMimeType();
    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    } catch {
      stop();
      setError("mic_unavailable");
      return;
    }
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) rollingRef.current.push(event.data);
    };
    try {
      recorder.start(CHUNK_MS);
    } catch {
      stop();
      setError("mic_unavailable");
      return;
    }
    recorderRef.current = recorder;

    const tick = () => {
      const meter = analyserRef.current;
      const buffer = bufferRef.current;
      const detector = detectorRef.current;
      if (!meter || !buffer || !detector) return;

      meter.getFloatTimeDomainData(buffer);
      let sum = 0;
      for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
      const rms = Math.sqrt(sum / buffer.length);
      levelRef.current = Math.min(1, rms / LEVEL_FULL_SCALE);

      const event = detector.push(
        rms,
        performance.now(),
        playingRef.current ? PLAYBACK_RMS_FACTOR : 1
      );
      if (event?.type === "start") {
        // Keep a little audio from before the edge so the first word is whole.
        captureFromRef.current = Math.max(
          0,
          rollingRef.current.length - PREROLL_CHUNKS
        );
        setSpeaking(true);
        handlersRef.current.onSpeechStart();
        return;
      }
      if (event?.type === "end") {
        setSpeaking(false);
        // Too short to be speech: no clip, and the caller resumes the walk.
        handlersRef.current.onSpeechEnd(event.isSpeech ? assemble() : null);
      }
    };
    timerRef.current = window.setInterval(tick, TICK_MS);
  }, [assemble, playingRef, stop]);

  // Release the microphone if the hook unmounts mid-walk.
  useEffect(() => stop, [stop]);

  return { levelRef, speaking, error, start, stop };
}
