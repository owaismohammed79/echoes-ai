/**
 * The walk's audio transport.
 *
 * One `<audio>` element, created once and reused for every sentence, fed by an
 * array-backed rolling queue that never holds more than three clips
 * (AGENTS.md 3: "prefetch a rolling queue of 2–3 short segments"). A clip is the
 * encoded audio for one validated sentence — in production the ElevenLabs bytes.
 *
 * Why the unlock and the queue are the *same* element: iOS grants playback
 * rights to an element that played during a user gesture, and refuses one
 * created after the first await. So the element that plays the silent WAV inside
 * the Start Walk tap is the very element that later plays sentence one, and it
 * is the element the Media Session describes.
 *
 * The queue is deliberately dumb: it plays what it is handed, in order, and
 * reports what happened. Deciding *which* sentences to load is the controller's
 * business, because only the controller knows the walk.
 */

export interface QueuedClip {
  /** The walk segment this clip belongs to, as a string. */
  id: string;
  blob: Blob;
}

export interface QueueHandlers {
  /** A clip has actually begun producing sound. */
  onStart: (id: string) => void;
  /** A clip finished on its own. */
  onEnd: (id: string) => void;
  /** Playback position inside the current clip, 0–1. Real time, not a guess. */
  onProgress: (id: string, fraction: number) => void;
  /** The queue ran dry while something expected audio. */
  onDrained: () => void;
  /** A clip could not be played at all. */
  onError: (reason: string) => void;
}

/** How many sentence clips may be buffered ahead of the speaker. */
export const QUEUE_DEPTH = 3;

/**
 * A silent but *valid* WAV. Validity matters: Safari treats a play() that fails
 * to decode as a failed gesture and revokes the token, so the unlock has to be
 * real audio.
 */
const SILENT_WAV =
  "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=";

const NO_HANDLERS: QueueHandlers = {
  onStart: () => {},
  onEnd: () => {},
  onProgress: () => {},
  onDrained: () => {},
  onError: () => {},
};

export class AudioQueue {
  private el: HTMLAudioElement | null = null;
  private handlers: QueueHandlers;
  private items: QueuedClip[] = [];
  private current: QueuedClip | null = null;
  /** Object URLs are revoked as soon as their clip is done; they leak otherwise. */
  private objectUrl: string | null = null;
  /** The user asked for silence, so do not auto-advance past it. */
  private held = false;

  constructor(handlers: QueueHandlers = NO_HANDLERS) {
    this.handlers = handlers;
  }

  /** Rebind the callbacks. The transport outlives any one React render. */
  attach(handlers: QueueHandlers): void {
    this.handlers = handlers;
  }

  /**
   * Must be called synchronously inside a user gesture, and never awaited.
   * Safe to call again mid-walk: it refuses to interrupt a clip already
   * playing, so a second Start tap cannot cut off the sentence under it.
   */
  unlock(): void {
    const el = this.element();
    if (this.current) return;
    try {
      el.src = SILENT_WAV;
      void el.play().catch(() => {
        /* the policy refused; nothing else to try from here */
      });
    } catch {
      /* no element available */
    }
  }

  private element(): HTMLAudioElement {
    if (this.el) return this.el;
    const el = new Audio();
    el.preload = "auto";
    // Hidden but present in the document, so the transport is inspectable in
    // devtools and the Media Session has a real element behind it.
    el.hidden = true;
    el.setAttribute("aria-hidden", "true");
    el.setAttribute("data-echoes-transport", "queue");
    el.addEventListener("ended", this.handleEnded);
    el.addEventListener("error", this.handleError);
    el.addEventListener("timeupdate", this.handleTime);
    document.body.appendChild(el);
    this.el = el;
    return el;
  }

  /** The one element, for anything that needs to describe the transport. */
  get element_(): HTMLAudioElement | null {
    return this.el;
  }

  get depth(): number {
    return this.items.length;
  }

  /** True while a clip holds the speaker. */
  get busy(): boolean {
    return this.current !== null;
  }

  get nowPlaying(): string | null {
    return this.current?.id ?? null;
  }

  enqueue(clip: QueuedClip): void {
    this.items.push(clip);
  }

  play(): void {
    this.held = false;
    const el = this.el;
    if (this.current && el) {
      // Resuming a clip that was paused mid-sentence.
      void el.play().catch(() => {});
      return;
    }
    this.advance();
  }

  /**
   * Resume if the OS paused us (screen lock, tab hide) but the walker did not.
   * ``held`` is the user-pause flag; it must stay paused until they tap.
   */
  resumeIfPaused(): boolean {
    const el = this.el;
    if (this.held || !el || !el.paused) return false;
    void el.play().catch(() => {});
    return true;
  }

  /** Held: the current clip keeps its position until play() again. */
  pause(): void {
    this.held = true;
    this.el?.pause();
  }

  /** Drop everything, including the clip on the speaker. */
  clear(): void {
    this.halt();
  }

  /**
   * Barge-in: stop the clip on the speaker *now*, drop every clip behind it, and
   * leave the transport silent and held until the caller decides what plays
   * next.
   *
   * The difference from :meth:`clear` is intent, not effect. `clear` tears the
   * transport down at the end of a walk; `interrupt` is the walk still running,
   * the walker talking over it. It reports whether a clip was actually cut off,
   * so the caller can tell a barge-in over narration from a question asked into
   * silence — the two resume differently.
   */
  interrupt(): boolean {
    const wasPlaying = this.current !== null;
    this.halt();
    return wasPlaying;
  }

  /** The one teardown both :meth:`clear` and :meth:`interrupt` share. */
  private halt(): void {
    this.items = [];
    this.held = true;
    const el = this.el;
    if (el) {
      el.pause();
      el.removeAttribute("src");
    }
    this.current = null;
    this.releaseUrl();
  }

  private advance(): void {
    this.releaseUrl();
    const next = this.items.shift() ?? null;
    this.current = next;
    if (!next) {
      this.handlers.onDrained();
      return;
    }
    const el = this.element();
    this.objectUrl = URL.createObjectURL(next.blob);
    el.src = this.objectUrl;
    void el.play().then(
      () => this.handlers.onStart(next.id),
      () => {
        // Autoplay refused, or the bytes are undecodable. Either way this clip
        // will not play and the caller has to decide what happens instead.
        this.current = null;
        this.releaseUrl();
        this.handlers.onError("queue_play_rejected");
        if (!this.held) this.advance();
      }
    );
  }

  private handleEnded = (): void => {
    const done = this.current;
    this.current = null;
    this.releaseUrl();
    if (done) this.handlers.onEnd(done.id);
    if (this.held) return;
    // ``onEnd`` may itself have started the next clip (the last sentence of a
    // cycle starts the ambient bed). Advancing again here would orphan that
    // clip: its own `ended` would then find no current clip, never report
    // ``onEnd``, and the walk would stall behind the silence it just started.
    if (this.current) return;
    // The silent unlock clip finishing is not the queue draining.
    if (!done && this.items.length === 0) return;
    this.advance();
  };

  private handleError = (): void => {
    // Swapping `src` ourselves reports a spurious error; only a failure while a
    // clip is loaded is real.
    if (!this.current) return;
    this.current = null;
    this.releaseUrl();
    this.handlers.onError("queue_clip_error");
    if (!this.held) this.advance();
  };

  private handleTime = (): void => {
    const el = this.el;
    const clip = this.current;
    if (!el || !clip) return;
    const duration = el.duration;
    if (!Number.isFinite(duration) || duration <= 0) return;
    this.handlers.onProgress(clip.id, Math.min(1, el.currentTime / duration));
  };

  private releaseUrl(): void {
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }
  }
}

/**
 * The app's single transport. Module-level so that the synchronous unlock in the
 * Start tap and the asynchronous playback later are guaranteed to be the same
 * element — which is the entire point on iOS.
 */
let shared: AudioQueue | null = null;

export function getAudioQueue(handlers?: QueueHandlers): AudioQueue {
  if (!shared) {
    shared = new AudioQueue(handlers);
  } else if (handlers) {
    shared.attach(handlers);
  }
  return shared;
}
