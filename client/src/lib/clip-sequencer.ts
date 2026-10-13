/**
 * Puts sentence clips back in sentence order.
 *
 * The controller fetches up to `QUEUE_DEPTH` clips at once, and the network
 * does not return them in the order they were asked for: a short sentence
 * routinely lands before the long one ahead of it. Enqueueing in arrival order
 * therefore plays sentences out of order -- the voice says sentence 3 while the
 * transcript is still on sentence 2.
 *
 * This holds a clip back until every clip before it has either arrived or been
 * declared empty, and releases them in index order. It knows nothing about audio
 * or React, which is what keeps it cheap to test.
 */
export interface Released<T> {
  index: number;
  clip: T;
}

export class ClipSequencer<T> {
  private readonly held = new Map<number, T | null>();
  private next = 0;

  /** Start (or restart) delivery at ``start``, dropping anything held. */
  reset(start = 0): void {
    this.held.clear();
    this.next = start;
  }

  /** The index that must arrive (or be skipped) before anything else is released. */
  get nextIndex(): number {
    return this.next;
  }

  /** Clips that have arrived but are waiting on an earlier one. */
  get waiting(): number {
    return this.held.size;
  }

  /**
   * Record the clip for ``index`` and return every clip now free to play, in
   * order. ``null`` means this index has nothing to play (an empty sentence):
   * it still occupies its slot so the clips behind it are not held forever.
   * A clip for an index that has already been passed is stale and is dropped.
   */
  put(index: number, clip: T | null): Released<T>[] {
    if (index < this.next) return [];
    this.held.set(index, clip);

    const released: Released<T>[] = [];
    while (this.held.has(this.next)) {
      const item = this.held.get(this.next) as T | null;
      this.held.delete(this.next);
      if (item !== null) released.push({ index: this.next, clip: item });
      this.next += 1;
    }
    return released;
  }
}
