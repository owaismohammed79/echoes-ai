import { AnimatedShinyText } from "@/components/ui/animated-shiny-text";
import { BackgroundBeams } from "@/components/ui/background-beams";
import { Ripple } from "@/components/ui/ripple";
import type { Activity } from "@/lib/demo-engine";
import { cn } from "@/lib/utils";

import { TechLabel } from "./tech-label";

/** States in which sound is actually moving, so the screen should move too. */
const MOVING = new Set<Activity>([
  "speaking",
  "listening",
  "thinking",
  "answering",
]);

const TICKS = 32;
const BAR_DELAYS = [0, 0.16, 0.32, 0.1, 0.24];

/**
 * The screen's only object. Three things are stacked here and each one is load
 * bearing: the beams give the dark field some depth while audio plays, the
 * ripple rings are the pulse, and the 32 ticks are a gauge that doubles as
 * progress through the walk — the walk literally lights up as it is spoken.
 */
export function AmbientState({
  activity,
  progress,
  title,
  sub,
}: {
  activity: Activity;
  progress: number;
  title: string;
  sub: string;
}) {
  const moving = MOVING.has(activity);
  const lit = Math.round(Math.min(1, Math.max(0, progress)) * TICKS);

  return (
    <div className="relative flex h-[260px] w-[260px] items-center justify-center">
      {/* aceternity Background Beams, damped to a whisper: accent paths that
          drift behind the ring. At 12% they read as depth, not decoration. */}
      <div className="pointer-events-none absolute inset-[-60px] overflow-hidden opacity-[0.12]">
        <BackgroundBeams className="h-full w-full" />
      </div>

      {/* magicui Ripple: the pulse. Present only while something is moving. */}
      {moving && (
        <Ripple
          mainCircleSize={132}
          numCircles={3}
          mainCircleOpacity={0.16}
          className="opacity-80"
        />
      )}

      {/* the breath: a hairline that expands and fades, like a status LED */}
      {moving && (
        <span
          className="animate-breathe pointer-events-none absolute h-[188px] w-[188px] rounded-full border border-[#10B981] motion-reduce:animate-none"
          aria-hidden="true"
        />
      )}

      <svg
        viewBox="0 0 260 260"
        className="absolute inset-0 h-full w-full"
        aria-hidden="true"
      >
        <circle
          cx="130"
          cy="130"
          r="108"
          fill="none"
          stroke="rgba(250,250,250,0.10)"
        />

        {Array.from({ length: TICKS }, (_, i) => {
          const angle = (i / TICKS) * Math.PI * 2 - Math.PI / 2;
          const on = i < lit;
          return (
            <line
              key={i}
              x1={130 + Math.cos(angle) * 112}
              y1={130 + Math.sin(angle) * 112}
              x2={130 + Math.cos(angle) * 120}
              y2={130 + Math.sin(angle) * 120}
              stroke={on ? "#10B981" : "rgba(250,250,250,0.14)"}
              strokeWidth="1.5"
            />
          );
        })}

        {/* a network edge is open: sweep the dial while we wait */}
        {activity === "thinking" && (
          <circle
            cx="130"
            cy="130"
            r="94"
            fill="none"
            stroke="#10B981"
            strokeWidth="1"
            strokeDasharray="16 574"
            className="animate-sweep origin-center motion-reduce:animate-none"
          />
        )}
      </svg>

      <div className="relative flex flex-col items-center">
        {/* magicui Animated Shiny Text: while sound is actually moving, a
            glare pans across the state word. Upstream it defaults to a black
            band on grey text, which would vanish on this base, so the band and
            the ink are both restated here in the system's own values. */}
        {moving ? (
          <AnimatedShinyText
            shimmerWidth={72}
            className="text-[15px] leading-none tracking-tight text-white/85 via-white/95 dark:text-white/85 dark:via-white/95 motion-reduce:animate-none"
          >
            {title}
          </AnimatedShinyText>
        ) : (
          <span
            className={cn(
              "text-[15px] leading-none tracking-tight",
              activity === "ready" ? "text-white/45" : "text-white/90"
            )}
          >
            {title}
          </span>
        )}

        <TechLabel tone={moving ? "accent" : "muted"} className="mt-[10px]">
          {sub}
        </TechLabel>

        {/* amplitude. Rendered always, opacity-gated, so it never reflows. */}
        <div
          className={cn(
            "mt-[18px] flex h-[22px] items-end gap-[5px] transition-opacity duration-300",
            moving ? "opacity-100" : "opacity-0"
          )}
          aria-hidden="true"
        >
          {BAR_DELAYS.map((delay, i) => (
            <span
              key={i}
              className="animate-amp w-[2px] origin-bottom bg-[#10B981] motion-reduce:animate-none"
              style={{
                height: 22,
                animationDelay: `${delay}s`,
                transform: "scaleY(0.4)",
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
