import { useCallback, useEffect, useRef, useState } from "react";

import { MouseMascot } from "@/components/echoes/mouse-mascot";
import { Hairline, TechLabel } from "@/components/echoes/tech-label";
import { cn } from "@/lib/utils";

/**
 * The phone-in-pocket guard.
 *
 * The walk is meant to be listened to with the phone away (AGENTS.md section 3),
 * so the moment the walker brings the app back into view mid-walk the screen
 * gets a friendly intervention instead of the walk: a modal that names what they
 * are doing and asks them to pocket it again.
 *
 * Two things run the whole time the walk is live:
 *  - a screen-on clock, which counts only while the page is actually visible, so
 *    a locked/pocketed phone stops it; and
 *  - a pair of listeners — `visibilitychange` and `blur`/`focus` — because iOS
 *    and Android disagree about which of the two fires when a phone is unlocked,
 *    and a return has to be caught either way.
 */

/** How long the walker has to pocket the phone before the reminder returns. */
const RECHECK_MS = 3000;

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
  // comes back, so the reminder fires on a genuine return — never on first paint.
  const awayRef = useRef(false);
  // The snooze timer behind the "I'm putting it away now" button.
  const recheckRef = useRef<number | null>(null);

  const clearRecheck = useCallback(() => {
    if (recheckRef.current !== null) {
      window.clearTimeout(recheckRef.current);
      recheckRef.current = null;
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

  // A fresh walk starts the clock, the reminder and the away-flag from zero.
  useEffect(() => {
    if (!active) return;
    setScreenOnSeconds(0);
    setNudge(false);
    awayRef.current = false;
    clearRecheck();
  }, [active, clearRecheck]);

  // --- leaving and returning ---------------------------------------------
  useEffect(() => {
    if (!active) return;

    const markAway = () => {
      awayRef.current = true;
    };
    const markPresent = () => {
      // Only a return — an away followed by a present — raises the reminder.
      if (!awayRef.current) return;
      awayRef.current = false;
      setNudge(true);
    };
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        markPresent();
      } else {
        markAway();
        setNudge(false);
        clearRecheck();
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
      clearRecheck();
    };
  }, [active, clearRecheck]);

  // --- dismissal ----------------------------------------------------------
  const dismiss = useCallback(() => {
    setNudge(false);
    clearRecheck();
    // Not a dismissal, a snooze: pocket the phone and it stays away; keep
    // looking at the screen and the reminder comes back.
    recheckRef.current = window.setTimeout(() => {
      recheckRef.current = null;
      if (document.visibilityState === "visible") setNudge(true);
    }, RECHECK_MS);
  }, [clearRecheck]);

  if (!active || !nudge) return null;

  return (
    <div
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="screen-lock-message"
      className="absolute inset-0 z-40 flex items-center justify-center bg-[#09090B]/85 px-5 backdrop-blur-sm"
    >
      <div className="w-full max-w-[440px] border border-white/10 bg-[#101010] p-5 shadow-[0_0_60px_rgba(16,185,129,0.08)]">
        <div className="flex items-center justify-between gap-3">
          <TechLabel tone="accent">phone-in-pocket</TechLabel>
          <span className="flex items-baseline gap-[8px] border border-white/10 bg-white/[0.03] px-[10px] py-[5px]">
            <span className="font-mono text-[10px] tracking-[0.064em] text-white/40">
              Screen-on time:
            </span>
            <span className="font-mono text-[11px] tabular-nums text-white/85">
              {formatMMSS(screenOnSeconds)}
            </span>
          </span>
        </div>

        <Hairline className="my-[14px]" />

        <div className="flex items-center gap-[18px]">
          {/* The mascot watches the pointer, so it turns to face the hand that
              just brought the phone to eye level. */}
          <MouseMascot className="h-20 w-20" />
          <p
            id="screen-lock-message"
            className="text-[15px] leading-[1.45] text-white/85"
          >
            Are you trying to look at your phone? 🦔 Put it back in your pocket
            and let your ears do the walking!
          </p>
        </div>

        <button
          type="button"
          onClick={dismiss}
          className={cn(
            "mt-[18px] h-[48px] w-full border border-[#10B981] font-mono text-[12px] uppercase tracking-[0.053em] text-[#10B981]",
            "transition-colors hover:bg-[#10B981]/10"
          )}
        >
          I&rsquo;m putting it away now
        </button>
      </div>
    </div>
  );
}
