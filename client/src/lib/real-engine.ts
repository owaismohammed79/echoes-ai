/**
 * The real walk engine: the client's direct line to the FastAPI service.
 *
 * `demoEngine` is a stand-in that never touches the network. This is its
 * replacement — every method maps to a route in `../server`, and nothing else in
 * the app knows the difference:
 *
 *   locate     -> POST /api/context        (the device fix, then the server's facts)
 *   narrate    -> POST /api/walk           (NDJSON metadata frames, streamed)
 *   transcribe -> POST /api/qa             (Groq whisper-large-v3)
 *   answer     -> POST /api/qa             (Gemma 4, grounded in the walk's facts)
 *   audio      -> GET  /api/audio/{token}  (Channel B: one sentence's bytes)
 *
 * Three shapes drive the design, and each one is why a method looks the way it
 * does:
 *
 * - `narrate` streams. Each `sentence` frame is handed to `onPartial` the moment
 *   it arrives, so the AudioQueue can start sentence one while the model is
 *   still writing sentence three (docs/audio-streaming.md 5). The facts do not
 *   travel in that stream — it carries text and citations only — so they are
 *   fetched from `/api/context` and cached.
 * - `transcribe` and `answer` are one round trip, not two. `/api/qa` runs STT
 *   and the grounded answer together, so `transcribe` performs the single
 *   multipart request and parks the result; `answer` reads it back.
 * - Channel B is single-use and session-bound (audio_tokens.py), so fetched
 *   bytes are cached by sentence: a resume or a restart must be served from
 *   memory, never by redeeming the token a second time.
 */

import {
  RESUME_TRANSITION,
  type Fact,
  type Fix,
  type Mode,
  type Segment,
  type WalkEngine,
  type WalkPrefs,
} from "@/lib/demo-engine";

/**
 * Where the API lives. Empty by default, so relative URLs hit the same origin —
 * the Vite dev proxy (vite.config.ts) forwards `/api` and `/static` to the
 * local server. A deployed client on its own origin sets `VITE_API_BASE_URL` at
 * build time.
 */
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

function api(path: string): string {
  return `${API_BASE}${path}`;
}

/** A session id the client owns, so audio tokens and PTT answers bind to this walk. */
function newSessionId(): string {
  const c = globalThis.crypto;
  if (c?.randomUUID) return c.randomUUID();
  return `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/** 3dp (~100m), the same quantisation the server enforces in ContextRequest. */
function round3(v: number): number {
  return Number(v.toFixed(3));
}

/**
 * The device's own fix. `locate` is only reached when the controller's first
 * attempt failed, so this can fail too — the rejection is the honest outcome,
 * and the caller decides what a walk with no location means.
 */
function deviceFix(): Promise<{ lat: number; lon: number }> {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error("geolocation_unavailable"));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      () => reject(new Error("geolocation_denied")),
      { timeout: 3500, maximumAge: 60000, enableHighAccuracy: false }
    );
  });
}

// --- Frame and header decoding ---------------------------------------------

/** A wire frame. Loose on purpose: the backend owns the shape, the client narrows. */
type WalkFrame = { type?: string; [key: string]: unknown };

function asString(v: unknown): string | undefined {
  return typeof v === "string" ? v : undefined;
}

function asStringArray(v: unknown): string[] {
  return Array.isArray(v)
    ? v.filter((x): x is string => typeof x === "string")
    : [];
}

function asMode(v: unknown): Mode | undefined {
  if (v === "STORY" || v === "NOTICE") return v;
  return undefined;
}

/** Header values are percent-encoded once (headers.py); decode exactly once. */
function safeDecode(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function decodeHeader(value: string | null): string {
  return value ? safeDecode(value) : "";
}

/** A comma-joined header (citations, sources), trimmed to items. */
function listHeader(value: string | null): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

/** `X-Timings-Ms`: `transcribe=640;answer=410;tts=280`. */
function parseTimings(value: string | null): Record<string, number> {
  const out: Record<string, number> = {};
  if (!value) return out;
  for (const part of value.split(";")) {
    const [key, raw] = part.split("=");
    const ms = Number(raw);
    if (key && Number.isFinite(ms)) out[key.trim()] = ms;
  }
  return out;
}

/** A filename for the multipart part, from the recorder's own MIME type. */
function clipName(clip: Blob): string {
  const type = clip.type || "audio/webm";
  if (type.includes("mp4") || type.includes("m4a")) return "clip.m4a";
  if (type.includes("ogg")) return "clip.ogg";
  if (type.includes("wav")) return "clip.wav";
  return "clip.webm";
}

/** What `/api/context` returns, narrowed to the fields a Fact needs. */
interface RawFact {
  fact_id?: unknown;
  title?: unknown;
  snippet?: unknown;
  source_url?: unknown;
  distance_m?: unknown;
  image_url?: unknown;
}

function toFact(raw: RawFact): Fact {
  const url = asString(raw.source_url) ?? "";
  const title = asString(raw.title) ?? "Untitled";
  return {
    id: asString(raw.fact_id) ?? "",
    title,
    detail: asString(raw.snippet) ?? "",
    distanceM: typeof raw.distance_m === "number" ? raw.distance_m : 0,
    source: sourceLabel(url, title),
    url,
    imageUrl: asString(raw.image_url),
  };
}

/** `en.wikipedia.org · Nelson's Column`, the same shape the demo engine uses. */
function sourceLabel(url: string, title: string): string {
  try {
    return `${new URL(url).hostname} · ${title}`;
  } catch {
    return title;
  }
}

/** `.../wiki/Nelson%27s_Column` -> `Nelson's Column`, a readable title from a URL. */
function titleFromUrl(url: string): string {
  try {
    const slug = new URL(url).pathname.split("/").filter(Boolean).pop() ?? "";
    return safeDecode(slug).replace(/_/g, " ") || "Source";
  } catch {
    return "Source";
  }
}

/**
 * Rebuild facts from the walk's own frames.
 *
 * The stream carries each sentence's citations and their source URLs but not
 * the fact bodies, so this is the fallback for when /api/context could not be
 * reached: a coarser fact (the title from the URL, the citing sentence as the
 * detail) beats an empty source drawer on a walk that plainly cited something.
 */
function factsFromFrames(
  cited: { citations: string[]; sourceUrls: string[]; text: string }[]
): Fact[] {
  const byId = new Map<string, Fact>();
  for (const frame of cited) {
    frame.citations.forEach((id, i) => {
      if (byId.has(id)) return;
      const url = frame.sourceUrls[i] ?? frame.sourceUrls[0] ?? "";
      const title = titleFromUrl(url);
      byId.set(id, {
        id,
        title,
        detail: frame.text,
        distanceM: 0,
        source: sourceLabel(url, title),
        url,
      });
    });
  }
  return [...byId.values()];
}

interface CachedContext {
  mode: Mode;
  facts: Fact[];
  ms: number;
}

/** One sentence's audio source, as the walk stream described it. */
interface SentenceAudio {
  /** Channel B URL, when the provider produced audio for this sentence. */
  url?: string;
  /** The server already decided this sentence falls back to the browser voice. */
  native: boolean;
}

/** The parked result of the one `/api/qa` round trip. */
interface QAState {
  transcript: string;
  answerText: string;
  citations: string[];
  sources: string[];
  refused: boolean;
  native: boolean;
  reason: string | null;
  timings: Record<string, number>;
}

function createRealEngine(): WalkEngine {
  // --- Per-walk state -----------------------------------------------------
  let sessionId: string | null = null;
  let lastFix: Fix | null = null;
  /** index -> audio source, populated as `sentence` frames arrive. */
  const sentenceAudio = new Map<number, SentenceAudio>();
  /** The pre-rendered reminder the backend appends, which the controller speaks as the disclaimer. */
  let reminder: { url: string; text: string } | null = null;
  /** Read the reminder text through a call, so control-flow narrowing on the
   * straight-line return cannot collapse `reminder` to `null`. */
  const reminderText = (): string | undefined => reminder?.text;
  /** Fetched sentence bytes, keyed `seg:<index>` / `reminder`. Single-use tokens live here. */
  const fetchedAudio = new Map<string, Blob>();
  /** Decoded push-to-talk answer bytes, keyed by the answer text it was spoken from. */
  const answerAudio = new Map<string, Blob>();
  /** The parked `/api/qa` result. */
  let qa: QAState | null = null;
  /** `/api/context`, keyed by the quantised fix, so locate and narrate share one call. */
  let contextCache: { key: string; value: CachedContext } | null = null;

  async function fetchContext(fix: Fix, prefs?: WalkPrefs): Promise<CachedContext> {
    const key = `${fix.lat.toFixed(3)},${fix.lon.toFixed(3)}`;
    if (contextCache && contextCache.key === key) return contextCache.value;

    const started = performance.now();
    const res = await fetch(api("/api/context"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat: fix.lat,
        lon: fix.lon,
        duration: prefs?.duration,
        mood: prefs?.mood,
      }),
    });
    if (!res.ok) throw new Error(`context_failed_${res.status}`);

    const data = (await res.json()) as { mode?: unknown; facts?: RawFact[] };
    const value: CachedContext = {
      mode: asMode(data.mode) ?? "STORY",
      facts: (data.facts ?? []).map(toFact),
      ms: Math.round(performance.now() - started),
    };
    contextCache = { key, value };
    return value;
  }

  async function fetchAudio(url: string, key: string): Promise<Blob | null> {
    const cached = fetchedAudio.get(key);
    if (cached) return cached;

    const headers: Record<string, string> = {};
    // Channel B is bound to the session that minted the token. Sending it means
    // a token that is not ours is refused rather than served.
    if (sessionId) headers["x-session-id"] = sessionId;

    const res = await fetch(api(url), { headers });
    if (!res.ok) return null;
    const blob = await res.blob();
    if (blob.size === 0) return null;
    // A token is single-use server-side, so this cache is not an optimisation:
    // it is what lets the same sentence be replayed after a pause or a resume.
    fetchedAudio.set(key, blob);
    return blob;
  }

  return {
    async locate(prefs) {
      const coords = await deviceFix();
      const fix: Fix = { lat: round3(coords.lat), lon: round3(coords.lon) };
      lastFix = fix;
      // The device fix is what the walk uses; /api/context is called for its
      // facts. narrate reuses the cached result, so this is one call, not two,
      // and a failure here must not stop the walk — narrate retries it.
      await fetchContext(fix, prefs).catch(() => null);
      return fix;
    },

    async narrate({ mode, fix, onPartial, prefs }) {
      const walkStarted = performance.now();
      sessionId = newSessionId();
      lastFix = fix;
      // Every walk starts clean: a clip fetched for the last one must not speak
      // into this one, and a stale token must not be reused.
      sentenceAudio.clear();
      fetchedAudio.clear();
      answerAudio.clear();
      reminder = null;
      qa = null;

      // The facts are not in the walk stream, so they come from /api/context.
      // Started now and awaited only when the first sentence needs them, so it
      // overlaps the walk's own aggregation rather than adding a round trip in
      // front of it.
      const contextPromise = fetchContext(fix, prefs).catch(() => null);

      const res = await fetch(api("/api/walk"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // session_id is the client's own, so the audio tokens minted during the
        // stream and the PTT answers after it are all bound to this walk.
        body: JSON.stringify({
          lat: fix.lat,
          lon: fix.lon,
          session_id: sessionId,
          duration: prefs?.duration,
          mood: prefs?.mood,
        }),
      });
      if (!res.ok || !res.body) throw new Error(`walk_failed_${res.status}`);

      const context = await contextPromise;
      const contextFacts = context?.facts ?? [];
      const segments: Segment[] = [];
      let resolvedMode: Mode = context?.mode ?? mode;
      // Every cited sentence frame, kept so the walk still has sources when
      // /api/context fails: the frames carry each citation and its source URL,
      // so facts can be rebuilt from them rather than lost.
      const cited: { citations: string[]; sourceUrls: string[]; text: string }[] =
        [];
      const factsNow = () =>
        contextFacts.length ? contextFacts : factsFromFrames(cited);

      const handle = (frame: WalkFrame): void => {
        if (frame.type === "sentence") {
          const index =
            typeof frame.index === "number" ? frame.index : segments.length;
          const text = asString(frame.text) ?? "";
          const citations = asStringArray(frame.citations);
          const sourceUrls = asStringArray(frame.source_urls);
          sentenceAudio.set(index, {
            url: asString(frame.audio_url),
            native: frame.use_native_tts === true,
          });
          segments.push({ text, citation: citations[0] ?? null });
          cited.push({ citations, sourceUrls, text });
          resolvedMode = asMode(frame.mode) ?? resolvedMode;
          // Handed over the instant it clears validation, so the queue can
          // start sentence one before the script is finished. The controller
          // ignores every call after the first; passing the running list keeps
          // the engine honest about what it knows so far.
          onPartial?.({
            mode: resolvedMode,
            facts: factsNow(),
            segments: segments.slice(),
            // Decisive on the first sentence: a server that cannot speak this
            // walk says so once, and the client starts on the browser voice.
            remoteAudio: asString(frame.audio_url) !== undefined,
          });
          return;
        }

        if (frame.type === "reminder") {
          const url = asString(frame.audio_url);
          const text = asString(frame.text);
          if (url && text) {
            reminder = { url, text };
          }
          return;
        }

        if (frame.type === "done") {
          resolvedMode = asMode(frame.mode) ?? resolvedMode;
        }
      };

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl = buffer.indexOf("\n");
        while (nl >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (line) dispatch(line, handle);
          nl = buffer.indexOf("\n");
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) dispatch(buffer.trim(), handle);

      return {
        mode: resolvedMode,
        facts: factsNow(),
        segments,
        timings: {
          context: context?.ms ?? 0,
          walk: Math.round(performance.now() - walkStarted),
        },
        // The server's reminder line, handed back so the controller can append
        // it as the final spoken segment. The client holds no copy of the text.
        reminder: reminderText(),
      };
    },

    async transcribe(clip) {
      qa = null;
      // The previous answer's bytes are stale the moment a new question starts.
      answerAudio.clear();
      // No clip means the microphone gave us nothing to send. Answering anyway
      // would be inventing a question, so the request is skipped and answer()
      // refuses instead.
      if (!clip) return "";

      const form = new FormData();
      form.append("audio", clip, clipName(clip));
      if (sessionId) form.append("session_id", sessionId);
      if (lastFix) {
        form.append("lat", String(lastFix.lat));
        form.append("lon", String(lastFix.lon));
      }

      const headers: Record<string, string> = {};
      if (sessionId) headers["x-session-id"] = sessionId;

      const res = await fetch(api("/api/qa"), {
        method: "POST",
        body: form,
        headers,
      });

      // The metadata rides in headers because the body may be audio (headers.py).
      const transcript = decodeHeader(res.headers.get("X-Transcript"));
      const answerText = decodeHeader(res.headers.get("X-Answer-Text"));
      const citations = listHeader(res.headers.get("X-Answer-Citations"));
      const sources = listHeader(res.headers.get("X-Answer-Sources")).map(safeDecode);
      const refused = res.headers.get("X-Answer-Refused") === "true";
      const native = res.headers.get("X-Use-Native-Tts") === "true";
      const reason = res.headers.get("X-Refusal-Reason");
      const timings = parseTimings(res.headers.get("X-Timings-Ms"));

      if (!res.ok) throw new Error(`qa_failed_${res.status}`);

      // Success with audio: the body is the answer's bytes, streamed. Read them
      // now, while the response is open, and park them under the answer text so
      // audio(text, -1) can hand them to the queue.
      if (!native && res.body) {
        const blob = await res.blob();
        if (blob.size > 0) answerAudio.set(answerText, blob);
      }

      qa = {
        transcript,
        answerText,
        citations,
        sources,
        refused,
        native,
        reason,
        timings,
      };
      return transcript;
    },

    async answer({ question }) {
      const result = qa;
      if (!result) {
        // The question never left the device (no clip), so there is nothing
        // grounded to say. This is a statement about the system, not about what
        // is in front of the walker (AGENTS.md 2).
        return {
          refused: true,
          reason: "no_audio",
          text: "I did not catch a question, so I have nothing to answer yet.",
          citations: [],
          timings: {},
        };
      }
      // The server answered from its own transcript; the client's copy is not
      // sent again.
      void question;
      return {
        refused: result.refused,
        reason: result.refused ? result.reason ?? "no_source" : undefined,
        text: result.answerText,
        citations: result.citations,
        timings: result.timings,
      };
    },

    async audio(text, index) {
      if (index < 0) {
        // A push-to-talk answer: /api/qa streams the body rather than issuing a
        // token, so its bytes were decoded by transcribe() and parked by text.
        return answerAudio.get(text) ?? null;
      }

      const entry = sentenceAudio.get(index);
      // The token path: fetch the sentence's bytes from Channel B.
      if (entry?.url) return fetchAudio(entry.url, `seg:${index}`);
      // The proactive fallback: the server flagged use_native_tts, so there is
      // no remote voice for this sentence and the browser speaks it instead.
      if (entry?.native) return null;

      // The controller appends the backend's reminder as a final, un-tokened
      // segment; the backend serves it as the pre-rendered asset and supplies
      // the exact text, so this maps that text back to its audio.
      if (reminder && text === reminder.text) {
        return fetchAudio(reminder.url, "reminder");
      }
      return null;
    },

    async transition() {
      // The bridge back into the walk. The server owns the phrase — a fixed,
      // allow-listed line, never client text — so this endpoint cannot be turned
      // into a free text-to-speech service.
      const headers: Record<string, string> = {};
      if (sessionId) headers["x-session-id"] = sessionId;
      const res = await fetch(api("/api/transition"), { method: "POST", headers });
      if (!res.ok) throw new Error(`transition_failed_${res.status}`);

      const text =
        decodeHeader(res.headers.get("X-Transition-Text")) || RESUME_TRANSITION;
      const native = res.headers.get("X-Use-Native-Tts") === "true";
      // No remote voice (or the provider is down): the browser speaks the line.
      if (native || !res.body) return { text, audio: null };
      const blob = await res.blob();
      return { text, audio: blob.size > 0 ? blob : null };
    },
  };
}

/** One NDJSON line -> a frame. A malformed line is skipped, not fatal. */
function dispatch(line: string, handle: (frame: WalkFrame) => void): void {
  try {
    const parsed = JSON.parse(line) as unknown;
    if (parsed && typeof parsed === "object") handle(parsed as WalkFrame);
  } catch {
    /* a partial or corrupt frame must not take the walk down */
  }
}

/**
 * The app's real engine. Module-level like `demoEngine`, so a walk's session id
 * and cached clips outlive any one React render.
 */
export const realEngine: WalkEngine = createRealEngine();
