import { describe, expect, it } from "vitest";

import { ClipSequencer } from "@/lib/clip-sequencer";

const order = (released: { index: number }[]) => released.map((r) => r.index);

describe("ClipSequencer", () => {
  it("releases clips in sentence order however they arrive", () => {
    const seq = new ClipSequencer<string>();
    seq.reset(0);

    expect(seq.put(2, "c")).toEqual([]);
    expect(seq.put(1, "b")).toEqual([]);
    expect(seq.waiting).toBe(2);

    // Sentence 0 lands last: everything behind it is released at once.
    expect(order(seq.put(0, "a"))).toEqual([0, 1, 2]);
    expect(seq.waiting).toBe(0);
    expect(seq.nextIndex).toBe(3);
  });

  it("lets an empty sentence keep its slot without holding the rest", () => {
    const seq = new ClipSequencer<string>();
    seq.reset(0);

    expect(seq.put(1, "b")).toEqual([]);
    // Sentence 0 has no audio (native / empty): its slot is skipped, not awaited.
    expect(order(seq.put(0, null))).toEqual([1]);
    expect(seq.nextIndex).toBe(2);
  });

  it("starts a continuation cycle at its own first index", () => {
    const seq = new ClipSequencer<string>();
    seq.reset(7);

    expect(seq.put(8, "b")).toEqual([]);
    expect(order(seq.put(7, "a"))).toEqual([7, 8]);
  });

  it("drops stale clips from a cycle that was already passed", () => {
    const seq = new ClipSequencer<string>();
    seq.reset(0);
    seq.put(0, "a");

    expect(seq.put(0, "again")).toEqual([]);
    expect(seq.waiting).toBe(0);
  });

  it("forgets held clips on reset", () => {
    const seq = new ClipSequencer<string>();
    seq.reset(0);
    seq.put(3, "d");
    seq.reset(10);

    expect(seq.waiting).toBe(0);
    expect(order(seq.put(10, "x"))).toEqual([10]);
  });
});
