/**
 * Screen Wake Lock.
 *
 * A walk is listened to with the phone away (AGENTS.md section 3), so the one
 * thing that must not happen is the OS sleeping the page mid-sentence: a
 * backgrounded tab is throttled, and its audio and timers go with it. The Wake
 * Lock API holds the screen — and the tab's place in the foreground — awake for
 * the length of the walk.
 *
 * The browser releases the lock itself whenever the page is hidden, so it is
 * re-acquired on `visibilitychange`. The whole thing is feature-detected and
 * swallowed on failure: Safari before 16.4 has no Wake Lock at all, and a
 * missing lock must never take the walk down (the same contract as
 * `media-session`).
 */

import { useEffect } from "react";

/** Just the slice of `WakeLockSentinel` this module touches. */
interface WakeLockSentinelLike {
  released: boolean;
  release: () => Promise<void>;
}

interface WakeLockLike {
  request: (type: "screen") => Promise<WakeLockSentinelLike>;
}

function wakeLock(): WakeLockLike | undefined {
  if (typeof navigator === "undefined") return undefined;
  return (navigator as Navigator & { wakeLock?: WakeLockLike }).wakeLock;
}

/** Hold a screen wake lock for as long as `active` is true. */
export function useScreenWakeLock(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    const api = wakeLock();
    if (!api) return;

    let sentinel: WakeLockSentinelLike | null = null;
    let cancelled = false;

    const acquire = async () => {
      // Hidden means there is nothing to keep awake, and requesting while
      // hidden is refused anyway.
      if (cancelled || document.visibilityState !== "visible") return;
      if (sentinel && !sentinel.released) return;
      try {
        sentinel = await api.request("screen");
      } catch {
        /* denied, low battery, or unsupported: the walk carries on */
      }
    };

    const onVisibility = () => {
      if (document.visibilityState === "visible") void acquire();
    };

    void acquire();
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVisibility);
      void sentinel?.release().catch(() => {});
      sentinel = null;
    };
  }, [active]);
}
