import { useCallback, useEffect, useRef, useState } from "react";

import { MouseMascot } from "@/components/echoes/mouse-mascot";
import { cn } from "@/lib/utils";

/**
 * The phone-in-pocket guard.
 *
 * The walk is meant to be listened to with the phone away (AGENTS.md section 3),
 * but a walker who wakes the screen has every right to look at it. So the
 * reminder is a slim, non-blocking banner (with the mascot) that appears for a
 * couple of seconds on a genuine return and then leaves on its own. It never
 * covers the walk UI and never intercepts a tap; the time spent looking is
 * simply counted as screen-on time.
 *
 * Two things run the whole time the walk is live:
 *  - a screen-on clock, which counts only while the page is actually visible, so
 *    a locked/pocketed phone stops it; and
 *  - a pair of listeners — `visibilitychange` and `blur`/`focus` — because iOS
 *    and Android disagree about which of the two fires when a phone is unlocked,
 *    and a return has to be caught either way.
 */

/** How long the banner stays up after the walker returns to the screen. */
const BANNER_MS = 2000;

/**
 * MM:SS, zero-padded. Exported because the summary prints the same figure it was
 * handed at the end of the walk.
 */
export function formatMMSS(totalSeconds: number): string {
  const safe = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(safe / 60);
  const seconds = safe % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

export interface ScreenLockGuardProps {
  /** The walk is running. The guard is inert — and renders nothing — otherwise. */
  active: boolean;
  /**
   * The accumulated screen-on time, pushed to the walk store every second so the
   * summary can report it once the walk ends.
   */
  onScreenOnChange: (seconds: number) => void;
}

export function ScreenLockGuard({
  active,
  onScreenOnChange,
}: ScreenLockGuardProps) {
  const [screenOnSeconds, setScreenOnSeconds] = useState(0);
  const [nudge, setNudge] = useState(false);
  // Set when the walker takes the app off-screen (blur/hide) and read when it
  // comes back, so the banner fires on a genuine return — never on first paint.
  const awayRef = useRef(false);
  // The auto-dismiss timer for the banner.
  const hideRef = useRef<number | null>(null);

  const clearHide = useCallback(() => {
    if (hideRef.current !== null) {
      window.clearTimeout(hideRef.current);
      hideRef.current = null;
    }
  }, []);

  // --- the screen-on clock ------------------------------------------------
  // Counts a second at a time, but only while the page is visible: a locked
  // phone fires visibilitychange, so the clock stops the instant it is pocketed.
  useEffect(() => {
    if (!active) return;
    const interval = window.setInterval(() => {
      if (document.visibilityState !== "visible") return;
      setScreenOnSeconds((prev) => prev + 1);
    }, 1000);
    return () => window.clearInterval(interval);
  }, [active]);

  // Mirror the clock into the walk store, which is where the summary reads it.
  useEffect(() => {
    onScreenOnChange(screenOnSeconds);
  }, [screenOnSeconds, onScreenOnChange]);

  // A fresh walk starts the clock, the banner and the away-flag from zero.
  useEffect(() => {
    if (!active) return;
    setScreenOnSeconds(0);
    setNudge(false);
    awayRef.current = false;
    clearHide();
  }, [active, clearHide]);

  // --- leaving and returning ---------------------------------------------
  useEffect(() => {
    if (!active) return;

    const markAway = () => {
      awayRef.current = true;
    };
    const markPresent = () => {
      // Only a return — an away followed by a present — raises the banner.
      if (!awayRef.current) return;
      awayRef.current = false;
      setNudge(true);
      // Restarts on every return, so a quick lock/unlock never cuts a banner short.
      clearHide();
      hideRef.current = window.setTimeout(() => {
        hideRef.current = null;
        setNudge(false);
      }, BANNER_MS);
    };
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        markPresent();
      } else {
        markAway();
        setNudge(false);
        clearHide();
      }
    };
    const onBlur = () => markAway();
    const onFocus = () => markPresent();

    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
      clearHide();
    };
  }, [active, clearHide]);

  if (!active || !nudge) return null;

  return (
    // The wrapper is pointer-transparent: the banner informs, it never blocks a
    // tap on the walk underneath. Safe-area aware so it clears the iOS notch.
    <div
      className="pointer-events-none absolute inset-x-0 top-0 z-[60] flex justify-center px-4"
      style={{ paddingTop: "calc(env(safe-area-inset-top, 0px) + 8px)" }}
    >
      <div
        role="status"
        aria-live="polite"
        className={cn(
          "animate-in fade-in slide-in-from-top-2 duration-300",
          "flex w-full max-w-[440px] items-center gap-[10px] border border-white/10 bg-[#101010]/95 py-[6px] pl-[6px] pr-[12px]",
          "shadow-[0_0_40px_rgba(16,185,129,0.10)] backdrop-blur-sm"
        )}
      >
        {/* Small, and still watching the pointer on desktop. */}
        <MouseMascot className="h-10 w-10" />
        <p className="min-w-0 flex-1 text-[12px] leading-[1.35] text-white/85">
          Phone back in your pocket? 🦔 This time counts as screen time.
        </p>
        <span className="shrink-0 font-mono text-[11px] tabular-nums text-[#10B981]">
          {formatMMSS(screenOnSeconds)}
        </span>
      </div>
    </div>
  );
}
