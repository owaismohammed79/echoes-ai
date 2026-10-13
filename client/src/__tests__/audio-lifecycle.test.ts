import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import {
  AMBIENT_PAUSE_MS,
  QA_GRACE_MS,
  makeAmbientBed,
  shouldResumeOnVisible,
  stopAllTracks,
} from "@/lib/audio-lifecycle";
import { AudioQueue } from "@/lib/audio-queue";

class FakeTrack {
  stopped = false;
  stop() {
    this.stopped = true;
  }
}

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

describe("shouldResumeOnVisible", () => {
  it("resumes a live walk the OS paused, not a user pause", () => {
    expect(
      shouldResumeOnVisible({
        visibilityState: "visible",
        walkActive: true,
        userPaused: false,
        elementPaused: true,
      })
    ).toBe(true);

    expect(
      shouldResumeOnVisible({
        visibilityState: "visible",
        walkActive: true,
        userPaused: true,
        elementPaused: true,
      })
    ).toBe(false);

    expect(
      shouldResumeOnVisible({
        visibilityState: "hidden",
        walkActive: true,
        userPaused: false,
        elementPaused: true,
      })
    ).toBe(false);
  });
});

describe("stopAllTracks", () => {
  it("stops every MediaStreamTrack so the hardware mic lock releases", () => {
    const a = new FakeTrack();
    const b = new FakeTrack();
    const stream = {
      getTracks: () => [a, b],
    } as unknown as MediaStream;

    stopAllTracks(stream);
    expect(a.stopped).toBe(true);
    expect(b.stopped).toBe(true);
  });
});

describe("makeAmbientBed", () => {
  it("builds a WAV long enough to keep the media session alive", async () => {
    const blob = makeAmbientBed(200, 8000);
    expect(blob.type).toBe("audio/wav");
    expect(blob.size).toBeGreaterThan(44);
  });

  it("uses the server-dictated 75s pause and 45s Q&A grace", () => {
    expect(AMBIENT_PAUSE_MS).toBe(75_000);
    expect(QA_GRACE_MS).toBe(45_000);
  });
});

describe("AudioQueue", () => {
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
    document.body.innerHTML = "";
    vi.spyOn(document.body, "appendChild").mockImplementation(
      (node) => node as unknown as Node
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("unlocks a single element and routes later clips through it", async () => {
    const ends: string[] = [];
    const queue = new AudioQueue({
      onStart: () => {},
      onEnd: (id) => ends.push(id),
      onProgress: () => {},
      onDrained: () => {},
      onError: () => {},
    });
    queue.unlock();
    expect(audio.src).toContain("data:audio/wav");

    queue.enqueue({ id: "segment:0", blob: new Blob(["aa"], { type: "audio/mpeg" }) });
    queue.play();
    expect(audio.paused).toBe(false);

    audio.dispatch("ended");
    expect(ends).toEqual(["segment:0"]);
  });

  it("resumeIfPaused plays when the OS paused an active walk", () => {
    const queue = new AudioQueue();
    queue.unlock();
    queue.enqueue({ id: "segment:0", blob: new Blob(["aa"]) });
    queue.play();
    audio.paused = true;
    expect(queue.resumeIfPaused()).toBe(true);
    expect(audio.paused).toBe(false);
  });

  it("resumeIfPaused does not override a user pause", () => {
    const queue = new AudioQueue();
    queue.unlock();
    queue.enqueue({ id: "segment:0", blob: new Blob(["aa"]) });
    queue.play();
    queue.pause();
    expect(queue.resumeIfPaused()).toBe(false);
    expect(audio.paused).toBe(true);
  });
});
