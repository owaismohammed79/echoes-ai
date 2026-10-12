import { useEffect, useRef } from "react";
import type { CSSProperties, MutableRefObject } from "react";

import type { Activity } from "@/lib/demo-engine";
import { cn } from "@/lib/utils";

import { TechLabel } from "./tech-label";

/**
 * The hands-free orb.
 *
 * Where the old push-to-talk button was held to record, this is a visualizer:
 * the walker never touches it. It reflects what the loop is doing — narrating,
 * hearing you, thinking, answering — and carries the one control that still
 * matters, the microphone switch.
 *
 * The listening ring is driven from `levelRef` by a `requestAnimationFrame` loop
 * writing a CSS custom property, not from React state. That is deliberate: the
 * microphone level changes ~20 times a second, and re-rendering the whole screen
 * at that rate to move one ring would be the wrong trade. The orb animates at
 * frame rate while the app renders only when a *state* changes.
 */

/** The mic glyph: monoline, drawn rather than pulled from an icon set. */
function MicGlyph({ muted }: { muted: boolean }) {
  const stroke = muted ? "rgba(250,250,250,0.45)" : "#10B981";
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 22 22"
      fill="none"
      stroke={stroke}
      strokeWidth="1.25"
      strokeLinecap="square"
      aria-hidden="true"
    >
      <rect x="8.25" y="2.5" width="5.5" height="10" rx="1" />
      <path d="M5 10.5a6 6 0 0 0 12 0" />
      <path d="M11 16.5V20" />
      {muted && <path d="M3.5 3.5l15 15" />}
    </svg>
  );
}

/** What the orb is doing, in the system's mono voice. */
const LABELS: Record<Activity, string> = {
  ready: "hands-free · speak any time",
  speaking: "narrating · talk to interrupt",
  paused: "paused · tap the ring to resume",
  listening: "listening",
  thinking: "thinking · whisper large v3",
  answering: "answering · grounded",
};

export function ConversationalOrb({
  activity,
  listening,
  levelRef,
  muted,
  mic,
  onToggleMute,
}: {
  activity: Activity;
  /** The walker is audibly talking — the barge-in state. */
  listening: boolean;
  /** Live microphone amplitude, 0–1. Read by rAF, never rendered. */
  levelRef: MutableRefObject<number>;
  muted: boolean;
  /** Why the microphone is unavailable, when it is. */
  mic: string | null;
  onToggleMute: () => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const el = wrapRef.current;
      if (el) el.style.setProperty("--level", levelRef.current.toFixed(3));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [levelRef]);

  const narrating = activity === "speaking";
  const answering = activity === "answering";
  const thinking = activity === "thinking";
  const active = listening || narrating || answering;

  return (
    <div className="mb-[16px] flex flex-col items-center gap-[10px]">
      <div
        ref={wrapRef}
        className="relative h-[72px] w-[72px]"
        style={{ "--level": 0 } as CSSProperties}
      >
        {/* Listening: a ring that expands and brightens with the walker's voice.
            The barge-in has already happened by the time this is on screen. */}
        {listening && (
          <>
            <span
              className="pointer-events-none absolute inset-0 rounded-full border border-[#10B981] motion-reduce:hidden"
              style={{
                transform: "scale(calc(1 + var(--level) * 0.55))",
                opacity: "calc(0.3 + var(--level) * 0.7)",
              }}
            />
            <span
              className="pointer-events-none absolute inset-[10px] rounded-full bg-[#10B981]"
              style={{
                transform: "scale(calc(0.5 + var(--level) * 0.5))",
                opacity: "calc(var(--level) * 0.3)",
              }}
            />
          </>
        )}

        {/* Narrating: a subdued green breath — present, not demanding. */}
        {narrating && (
          <span className="animate-breathe pointer-events-none absolute inset-0 rounded-full border border-[#10B981]/60 motion-reduce:animate-none" />
        )}

        {/* Thinking: a fast pulse — the one state the walker cannot control. */}
        {thinking && (
          <span
            className="animate-pulse-ring pointer-events-none absolute inset-0 rounded-full border border-[#10B981] motion-reduce:animate-none"
            style={{ animationDuration: "1.1s" }}
          />
        )}

        {/* Answering: a bright, steady ring. */}
        {answering && (
          <span className="pointer-events-none absolute inset-0 rounded-full border border-[#10B981]" />
        )}

        <div
          className={cn(
            "absolute inset-[10px] flex items-center justify-center rounded-full border bg-[#18181B]",
            active ? "border-[#10B981]/60" : "border-white/15"
          )}
        >
          <MicGlyph muted={muted} />
        </div>
      </div>

      <div className="flex items-baseline gap-[12px]">
        <TechLabel tone={active ? "accent" : "faint"}>
          {muted ? "microphone off" : LABELS[activity]}
        </TechLabel>
        <button
          type="button"
          onClick={onToggleMute}
          aria-pressed={muted}
          className="font-mono text-[10px] tracking-[0.064em] text-white/45 uppercase transition-colors hover:text-white/80"
        >
          {muted ? "unmute microphone" : "mute microphone"}
        </button>
      </div>

      {/* A microphone that failed is stated, never swallowed: a hands-free walk
          with a dead mic is a walk that cannot be asked anything. */}
      {mic && !muted && <TechLabel tone="accent">{mic}</TechLabel>}
    </div>
  );
}
