import type { ComponentPropsWithoutRef } from "react";

import { cn } from "@/lib/utils";

type Tone = "muted" | "ink" | "accent" | "faint";

const TONES: Record<Tone, string> = {
  faint: "text-white/25",
  muted: "text-white/45",
  ink: "text-white/70",
  accent: "text-[#10B981]",
};

/**
 * The system voice: anything a terminal would print — state, mode, coordinates,
 * citations, stage timings — renders in mono at 10px with wide tracking.
 * Anything a human reads uses the sans face at 14-16px.
 */
export function TechLabel({
  children,
  tone = "muted",
  className,
  ...rest
}: ComponentPropsWithoutRef<"span"> & { tone?: Tone }) {
  return (
    <span
      className={cn(
        "font-mono text-[10px] leading-[1.6] uppercase tracking-[0.064em]",
        TONES[tone],
        className
      )}
      {...rest}
    >
      {children}
    </span>
  );
}

/** A 1px rule. The system's only separator. */
export function Hairline({ className }: { className?: string }) {
  return <div className={cn("h-px w-full bg-white/10", className)} />;
}
