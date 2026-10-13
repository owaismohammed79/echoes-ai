import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AudioQueue } from "@/lib/audio-queue";
import { realEngine } from "@/lib/real-engine";

// ---------------------------------------------------------------------------
// AudioQueue: the ambient-bed hand-off
// ---------------------------------------------------------------------------

class FakeAudio {
  src = "";
  paused = true;
  hidden = false;
  preload = "";
  currentTime = 0;
  duration = 1;
  listeners: Record<string, Array<() => void>> = {};

  setAttribute() {}
  addEventListener(type: string, fn: () => void) {
    (this.listeners[type] ??= []).push(fn);
  }
  play() {
    this.paused = false;
    return Promise.resolve();
  }
  pause() {
    this.paused = true;
  }
  removeAttribute() {
    this.src = "";
  }
  dispatch(type: string) {
    for (const fn of this.listeners[type] ?? []) fn();
  }
}

describe("AudioQueue: a clip started from onEnd is not orphaned", () => {
  let audio: FakeAudio;

  beforeEach(() => {
    audio = new FakeAudio();
    vi.stubGlobal(
      "Audio",
      class {
        constructor() {
          return audio;
        }
      }
    );
    vi.spyOn(document.body, "appendChild").mockImplementation(
      (node) => node as unknown as Node
    );
    URL.createObjectURL = vi.fn(() => "blob:fake");
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("reports the bed's own end after the last sentence started it", async () => {
    const ends: string[] = [];
    const starts: string[] = [];
    let drained = 0;

    const queue: AudioQueue = new AudioQueue({
      onStart: (id) => starts.push(id),
      onEnd: (id) => {
        ends.push(id);
        // The controller starts the ambient bed from the last sentence's end.
        if (id === "segment:1") {
          queue.enqueue({ id: "ambient:pause", blob: new Blob(["bed"]) });
          queue.play();
        }
      },
      onProgress: () => {},
      onDrained: () => {
        drained += 1;
      },
      onError: () => {},
    });
    queue.unlock();

    queue.enqueue({ id: "segment:0", blob: new Blob(["a"]) });
    queue.enqueue({ id: "segment:1", blob: new Blob(["b"]) });
    queue.play();
    await Promise.resolve();

    audio.dispatch("ended"); // segment:0
    await Promise.resolve();
    audio.dispatch("ended"); // segment:1 -> onEnd starts the bed
    await Promise.resolve();

    expect(queue.nowPlaying).toBe("ambient:pause");

    audio.dispatch("ended"); // the bed finishes
    await Promise.resolve();

    // Before the fix the bed was advanced past, so this never fired and the
    // next cycle never started.
    expect(ends).toEqual(["segment:0", "segment:1", "ambient:pause"]);
    expect(starts).toContain("ambient:pause");
    expect(drained).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// real-engine: replaying two cycles of /api/walk frames
// ---------------------------------------------------------------------------

const enc = new TextEncoder();

function ndjson(frames: object[], chunk = 37): Response {
  const bytes = enc.encode(frames.map((f) => JSON.stringify(f)).join("\n") + "\n");
  // Split mid-line on purpose: the parser must reassemble frames across chunks.
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (let i = 0; i < bytes.length; i += chunk) {
        controller.enqueue(bytes.slice(i, i + chunk));
      }
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "application/x-ndjson" },
  });
}

const FIX = { lat: 12.97, lon: 77.59 };

describe("realEngine.narrate across cycles", () => {
  const walkBodies: Array<Record<string, unknown>> = [];
  let walkResponses: Response[] = [];

  beforeEach(() => {
    walkBodies.length = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/api/context")) {
          return new Response(JSON.stringify({ mode: "STORY", facts: [] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (url.includes("/api/walk")) {
          walkBodies.push(JSON.parse(String(init?.body ?? "{}")));
          const next = walkResponses.shift();
          if (!next) throw new Error("no scripted /api/walk response");
          return next;
        }
        throw new Error(`unexpected fetch ${url}`);
      })
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps indices contiguous, never re-sends text, and surfaces pacing", async () => {
    walkResponses = [
      ndjson([
        { type: "data", mode: "STORY" },
        { type: "text", index: 0, content: "First sentence.", audio_url: "/api/audio/t0" },
        { type: "text", index: 1, content: "Second sentence.", audio_url: "/api/audio/t1" },
        { type: "indicator", status: "cycle_complete", pause_ms: 61_000 },
      ]),
      ndjson([
        { type: "data", mode: "STORY" },
        { type: "text", index: 2, content: "Third sentence.", audio_url: "/api/audio/t2" },
        { type: "indicator", status: "walk_complete", grace_ms: 45_000, pad_ms: 8_000 },
      ]),
    ];

    const partials: number[] = [];
    const first = await realEngine.narrate({
      mode: "STORY",
      fix: FIX,
      prefs: { duration: 5, mood: "calm" } as never,
      onPartial: (p) => partials.push(p.segments.length),
    });

    expect(first.complete).toBe(false);
    expect(first.pauseMs).toBe(61_000);
    expect(first.padMs).toBeUndefined();
    expect(first.segments.map((s) => s.text)).toEqual([
      "First sentence.",
      "Second sentence.",
    ]);

    const second = await realEngine.narrate({
      mode: "STORY",
      fix: FIX,
      continuation: true,
      prefs: { duration: 5, mood: "calm" } as never,
    });

    // The transcript is cumulative and in order: what is on screen is exactly
    // what the sequencer will speak, with no repeat of cycle one.
    expect(second.segments.map((s) => s.text)).toEqual([
      "First sentence.",
      "Second sentence.",
      "Third sentence.",
    ]);
    expect(second.complete).toBe(true);
    expect(second.graceMs).toBe(45_000);
    expect(second.padMs).toBe(8_000);

    // Both cycles belong to one session.
    expect(walkBodies).toHaveLength(2);
    expect(walkBodies[1].session_id).toBe(walkBodies[0].session_id);

    // Audio for a later cycle resolves by absolute index.
    expect(partials.at(-1)).toBe(2);
  });

  it("skips a corrupt line instead of failing the cycle", async () => {
    walkResponses = [
      new Response(
        [
          JSON.stringify({ type: "text", index: 0, content: "Kept.", audio_url: "/a" }),
          "{not json",
          JSON.stringify({ type: "indicator", status: "cycle_complete", pause_ms: 40_000 }),
        ].join("\n") + "\n",
        { status: 200 }
      ),
    ];

    const result = await realEngine.narrate({
      mode: "STORY",
      fix: FIX,
      prefs: { duration: 5, mood: "calm" } as never,
    });
    expect(result.segments.map((s) => s.text)).toEqual(["Kept."]);
    expect(result.pauseMs).toBe(40_000);
  });
});
