/**
 * The voice-activity decision, separated from the browser.
 *
 * The Web Audio plumbing — the analyser, the recorder, the stream — lives in
 * `use-continuous-vad`; this is the part that *decides*, and it decides from
 * nothing but a stream of RMS samples and timestamps. Keeping it pure is what
 * lets the thresholds be tested against a scripted signal rather than a
 * microphone, and what keeps the tuning honest instead of hopeful.
 *
 * Two layers keep a noisy street from being mistaken for a question:
 *
 *  - a **calibration window**: the first few ticks only measure the room, so the
 *    ambient level is known before any detection happens. Without it, noise that
 *    is simply above the absolute floor fires a barge-in before the floor has
 *    had a chance to rise past it;
 *  - a **noise floor**: after calibration, voice must clear the measured ambient
 *    by a factor, so steady background sound lifts the bar above itself.
 *
 * Neither replaces the caller's own defence: a false trigger is *undone* by the
 * controller (it hushes playback rather than clearing it), so the worst a blip
 * can do is a brief pause, never a repeated sentence.
 */

export interface VadConfig {
  /** RMS above which a sample counts as voice, however quiet the room is. */
  silenceRms: number;
  /** Voice must also clear the measured ambient level by this factor. */
  noiseMultiplier: number;
  /** Silence this long ends an utterance. */
  silenceMs: number;
  /** Speech shorter than this is noise, not a question. */
  minSpeechMs: number;
  /** Consecutive loud samples required before speech is declared. */
  onsetFrames: number;
  /** Ticks spent learning the room before detection starts. */
  calibrationTicks: number;
}

export type VadEvent =
  | { type: "start" }
  | { type: "end"; spokeMs: number; isSpeech: boolean };

/** Calibration adapts fast: the room is learned within a few hundred ms. */
const CALIBRATION_ALPHA = 0.25;
/** Afterwards the floor rises slowly, so a single loud moment is not the norm. */
const FLOOR_UP_ALPHA = 0.08;
/** But it falls at once: a quiet room is believed immediately. */
const FLOOR_DOWN_ALPHA = 0.4;

export class VoiceActivityDetector {
  private speaking = false;
  private above = 0;
  private lastVoiceAt = 0;
  private startedAt = 0;
  /** A running estimate of the room, so a noisy street does not read as speech. */
  private noiseFloor = 0;
  private calibration: number;

  constructor(private readonly config: VadConfig) {
    this.calibration = config.calibrationTicks;
  }

  get isSpeaking(): boolean {
    return this.speaking;
  }

  /**
   * Feed one RMS sample. Returns the boundary it crossed, or null.
   *
   * `gate` scales the threshold: while the app's own voice is playing, the
   * microphone is hearing it too, so the bar to barge in is raised rather than
   * left where the narration can trip it.
   */
  push(rms: number, now: number, gate = 1): VadEvent | null {
    if (this.calibration > 0) {
      this.calibration -= 1;
      this.noiseFloor += (rms - this.noiseFloor) * CALIBRATION_ALPHA;
      return null;
    }

    // Track the ambient level only while nobody is talking, so a voice never
    // raises the floor that is meant to sit under it.
    if (!this.speaking) {
      const alpha = rms < this.noiseFloor ? FLOOR_DOWN_ALPHA : FLOOR_UP_ALPHA;
      this.noiseFloor += (rms - this.noiseFloor) * alpha;
    }

    const threshold =
      Math.max(
        this.config.silenceRms,
        this.noiseFloor * this.config.noiseMultiplier
      ) * gate;

    if (rms >= threshold) {
      this.lastVoiceAt = now;
      this.above += 1;
      if (!this.speaking && this.above >= this.config.onsetFrames) {
        this.speaking = true;
        this.startedAt = now;
        return { type: "start" };
      }
      return null;
    }

    this.above = 0;
    if (this.speaking && now - this.lastVoiceAt >= this.config.silenceMs) {
      this.speaking = false;
      const spokeMs = this.lastVoiceAt - this.startedAt;
      return {
        type: "end",
        spokeMs,
        isSpeech: spokeMs >= this.config.minSpeechMs,
      };
    }
    return null;
  }
}
