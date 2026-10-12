/**
 * The walk engine: the one seam between this UI and the backend.
 *
 * Every method maps to a route on the FastAPI service in ../server:
 *   locate     -> POST /api/context (geolocation happens client-side)
 *   narrate    -> POST /api/walk   (FactObject[] -> validated, cited segments)
 *   transcribe -> POST /api/qa     (Groq whisper-large-v3)
 *   answer     -> POST /api/qa     (Gemma 4, grounded in the cached facts)
 *
 * Nothing else in the app knows the network exists, so swapping this object for
 * real callers is the whole integration.
 */

export type Mode = "STORY" | "NOTICE";

export type Phase = "idle" | "arming" | "live";

/** What the ring is doing. Drives every visual state on the screen. */
export type Activity =
  | "ready"
  | "speaking"
  | "paused"
  | "listening"
  | "thinking"
  | "answering";

/** Which voice produced the audio. Mirrors the use_native_tts response header. */
export type VoicePath = "native" | "remote";

export interface Fact {
  id: string;
  title: string;
  detail: string;
  distanceM: number;
  source: string;
  url: string;
  /** Source thumbnail, when the provider supplied one (Wikipedia pageimages). */
  imageUrl?: string;
}

export interface Segment {
  text: string;
  /** The [[fact_id]] this sentence is grounded in, when there is one. */
  citation: string | null;
}

export interface Fix {
  lat: number;
  lon: number;
}

/**
 * What the walker chose on the landing page before the walk began. Both are
 * optional: a walk with neither is exactly the pre-existing behaviour.
 */
export interface WalkPrefs {
  /** Requested walk length in minutes (5-45). */
  duration?: number;
  /** Selected mood key: energetic | calm | sad | reflective. */
  mood?: string;
}

export interface Narration {
  mode: Mode;
  facts: Fact[];
  segments: Segment[];
  timings: Record<string, number>;
  /**
   * The final safety reminder, as the backend supplied it. The frontend never
   * hardcodes this line -- it is the server's, appended in code after validation
   * (AGENTS.md section 2) -- so a walk without one simply carries no reminder.
   */
  reminder?: string;
}

/**
 * The walk so far, handed to `onPartial` as a streaming backend emits it. It is
 * a `Narration` without the timings, because those are only known once the walk
 * is over — and the whole point of a partial is that the walk is not.
 */
export interface PartialNarration {
  mode: Mode;
  facts: Fact[];
  segments: Segment[];
  /**
   * False when the backend has said it has no remote voice for this walk
   * (``use_native_tts``). The controller then starts on the browser voice
   * rather than fetching audio that will not come and calling it a failure.
   */
  remoteAudio?: boolean;
}

export interface Answer {
  refused: boolean;
  reason?: string;
  text: string;
  citations: string[];
  timings: Record<string, number>;
}

export interface WalkEngine {
  locate(prefs?: WalkPrefs): Promise<Fix>;
  /**
   * `onPartial` is called as the walk is written, not after it: a streaming
   * backend hands each sentence over the moment it clears validation, so the
   * controller can go live and start speaking sentence one while sentence three
   * is still being generated. An engine that only knows the whole script calls
   * it once, at the end, and is still correct.
   *
   * `prefs` carries the walker's chosen duration and mood through to the
   * `/api/walk` and `/api/context` payloads.
   */
  narrate(input: {
    mode: Mode;
    fix: Fix;
    prefs?: WalkPrefs;
    onPartial?: (partial: PartialNarration) => void;
  }): Promise<Narration>;
  /** The recorded clip, or null when the microphone gave us nothing. */
  transcribe(clip: Blob | null): Promise<string>;
  answer(input: { question: string; facts: Fact[]; mode: Mode }): Promise<Answer>;
  /**
   * The encoded audio for one validated sentence — in production the ElevenLabs
   * bytes — or null when the remote voice is unavailable.
   *
   * Optional on purpose. An engine without it makes the walk speak through the
   * browser voice, which is `use_native_tts: true` on screen; the demo engine
   * has no audio to give, so that whole path stays honest rather than silent.
   */
  audio?(text: string, index: number): Promise<Blob | null>;
  /**
   * The bridge spoken between an answer and the narration it interrupted. A
   * short, fixed line — never user text — that keeps a resumed walk from
   * restarting mid-air. `audio: null` means the browser voice speaks it.
   * Optional: an engine without it resumes silently.
   */
  transition?(): Promise<{ text: string; audio: Blob | null }>;
}

/**
 * The bridge between a push-to-talk answer and the narration it interrupted.
 * Fixed text, so it is a constant here and on the server rather than a knob the
 * client could turn into a free TTS proxy.
 */
export const RESUME_TRANSITION = "Now, continuing our walk.";

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const DEMO_FACTS: Fact[] = [
  {
    id: "a1c4f2",
    title: "Nelson's Column",
    detail:
      "A 52-metre granite column raised in 1843 to honour Admiral Horatio Nelson, watching over the square from a Corinthian capital.",
    distanceM: 21.6,
    source: "en.wikipedia.org · Nelson's Column",
    url: "https://en.wikipedia.org/wiki/Nelson%27s_Column",
  },
  {
    id: "7d90be",
    title: "St Martin-in-the-Fields",
    detail:
      "A Greek Revival church on the north-east corner, its portico raised in the 1720s over an earlier Tudor building.",
    distanceM: 180.4,
    source: "en.wikipedia.org · St Martin-in-the-Fields",
    url: "https://en.wikipedia.org/wiki/St_Martin-in-the-Fields",
  },
  {
    id: "c33a18",
    title: "The Trafalgar Fountains",
    detail:
      "Two fountain basins added in 1845 were rebuilt in 1939 with pumps that can lift 46 litres a second.",
    distanceM: 62.1,
    source: "en.wikipedia.org · Trafalgar Square",
    url: "https://en.wikipedia.org/wiki/Trafalgar_Square",
  },
];

/** Notice mode: sensory invitations only. No assertions, no directions. */
const DEMO_NOTICE: Segment[] = [
  {
    text: "If you are somewhere safe to pause, notice how the light falls on the nearest surface.",
    citation: null,
  },
  {
    text: "If you are somewhere safe to pause, notice the sound underneath the loudest one.",
    citation: null,
  },
];

const DEMO_STORY: Segment[] = [
  {
    text: "You are a few steps from a granite column raised in 1843 for Admiral Nelson — fifty-two metres of it, capped with a Corinthian capital cut from Devon granite.",
    citation: "a1c4f2",
  },
  {
    text: "Further along the north-east corner stands St Martin-in-the-Fields, its portico added in the 1720s over a Tudor church that had stood there for two centuries.",
    citation: "7d90be",
  },
  {
    text: "The two fountain basins beside you were rebuilt in 1939, and their pumps can lift forty-six litres of water every second.",
    citation: "c33a18",
  },
];

export const demoEngine: WalkEngine = {
  async locate() {
    await wait(700);
    return { lat: 51.5074, lon: -0.1278 };
  },

  async narrate({ mode }) {
    await wait(800);
    if (mode === "NOTICE") {
      return {
        mode: "NOTICE",
        facts: [],
        timings: { context: 214, walk: 0 },
        segments: DEMO_NOTICE,
      };
    }
    return {
      mode: "STORY",
      facts: DEMO_FACTS,
      timings: { context: 612, walk: 431 },
      segments: DEMO_STORY,
    };
  },

  async transcribe(clip: Blob | null) {
    // Deliberately ignores the clip: with no Groq key there is nothing to send
    // it to. The controller already tells the screen when no clip was captured,
    // so this stand-in never has to pretend one arrived.
    void clip;
    await wait(1100);
    return "what am I looking at";
  },

  async answer({ question, facts }) {
    await wait(850);
    if (facts.length === 0) {
      return {
        refused: true,
        reason: "no_qualifying_facts",
        text: "I have nothing verified for this spot, so I will stay quiet rather than guess.",
        citations: [],
        timings: { stt: 1180, llm: 0, tts: 0 },
      };
    }
    return {
      refused: false,
      text: `The column behind you commemorates Nelson, and the church portico to the north-east predates it by more than a century. You asked: ${question}.`,
      citations: [facts[0].id],
      timings: { stt: 1180, llm: 402, tts: 356 },
    };
  },

  async transition() {
    // No remote voice to give: the browser speaks the bridge, exactly as it
    // speaks the rest of this walk.
    await wait(120);
    return { text: RESUME_TRANSITION, audio: null };
  },
};
