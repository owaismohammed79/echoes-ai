import type { ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { Drawer } from "vaul";

import { LandmarkThumbnail } from "@/components/echoes/landmark-thumbnail";
import { Hairline, TechLabel } from "@/components/echoes/tech-label";
import { NoiseOverlay } from "@/components/echoes/noise-overlay";
import { Card, CardContent } from "@/components/ui/card";
import type { Fact, Mode } from "@/lib/demo-engine";
import { cn } from "@/lib/utils";
import { classifyVoiceFailure } from "@/lib/voice-status";

function SourceRow({ fact, index }: { fact: Fact; index: number }) {
  return (
    <li className="px-5 py-[14px]">
      <div className="flex items-start gap-[12px]">
        <TechLabel className="mt-[3px] shrink-0">
          {String(index + 1).padStart(2, "0")}
        </TechLabel>
        {/* The Wikipedia pageimage, or the contour mark when there is none. */}
        <LandmarkThumbnail src={fact.imageUrl} alt={fact.title} />
        <div className="min-w-0 flex-1">
          <p className="text-[14px] leading-[1.38] text-white/85">{fact.title}</p>
          <p className="mt-[6px] text-[13px] leading-[1.45] text-white/40">
            {fact.detail}
          </p>
          <div className="mt-[10px] flex flex-wrap items-baseline gap-x-[12px] gap-y-[4px]">
            <TechLabel>cited {fact.id}</TechLabel>
            <TechLabel>{fact.distanceM.toFixed(1)}m</TechLabel>
            <a
              href={fact.url}
              target="_blank"
              rel="noreferrer noopener"
              className="font-mono text-[10px] tracking-[0.064em] text-[#10B981] uppercase underline-offset-2 hover:underline"
            >
              {fact.source} ↗
            </a>
          </div>
        </div>
      </div>
    </li>
  );
}

/**
 * The bottom panel: verified sources, the stage timings, and the refusal state.
 *
 * Built from vaul (shadcn/ui's Drawer primitive) so the drag gesture is real,
 * with the aceternity glass treatment — translucent surface, backdrop blur, and
 * the grain overlay — layered onto a shadcn Card.
 *
 * `open` is controlled by our own button rather than vaul's Drawer.Trigger: the
 * trigger's outside-press handler races its own open state, so a mouse click
 * opened and instantly re-dismissed the sheet. The trigger here is a plain
 * button, which is deterministic. Left as modal (the default) so there is a
 * real backdrop to press: with `modal={false}` vaul renders no overlay, and the
 * only way out of the sheet was the drag handle.
 *
 * Opening the sheet is a mode change. It covers the ring and the orb, so the
 * audio transport is re-hosted inside the sheet instead of being left
 * unreachable underneath it.
 */
export function SourceDrawer({
  open,
  onOpenChange,
  facts,
  mode,
  timings,
  fallback,
  refused,
  audioControl,
  simulation,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  facts: Fact[];
  mode: Mode;
  timings: Record<string, number> | null;
  fallback: string | null;
  refused: boolean;
  /** Pause/resume, re-hosted here because the sheet covers the ring. */
  audioControl: ReactNode;
  simulation: ReactNode;
}) {
  const voice = classifyVoiceFailure(fallback);

  return (
    <Drawer.Root open={open} onOpenChange={onOpenChange}>
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onOpenChange(!open);
        }}
        aria-expanded={open}
        className="flex min-h-[58px] w-full items-center justify-between border-t border-white/10 bg-[#18181B] px-5 text-left"
        aria-label="Verified sources"
      >
          <span className="flex items-baseline gap-[10px]">
            <TechLabel tone={facts.length ? "ink" : "muted"}>
              {mode === "NOTICE"
                ? "Notice mode · no facts"
                : `Verified sources · ${facts.length}`}
            </TechLabel>
            {refused && <TechLabel tone="accent">refused</TechLabel>}
            {voice.badge && <TechLabel tone="accent">{voice.badge}</TechLabel>}
          </span>
          <span className="flex items-center gap-[10px] text-white/45">
            <TechLabel>{open ? "close" : "open"}</TechLabel>
          <ChevronDown
            size={12}
            strokeWidth={1.25}
            className={cn("transition-transform duration-200", open && "rotate-180")}
          />
        </span>
      </button>

      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-x-0 bottom-0 top-24 z-40 bg-[#09090B]/60" />
        <Drawer.Content
          className="fixed inset-x-0 bottom-0 z-50 outline-none"
          aria-describedby={undefined}
        >
          <Drawer.Title className="sr-only">Verified sources</Drawer.Title>
          <Drawer.Handle className="mx-auto mb-[10px] h-[2px] w-[36px] bg-white/20" />

          <Card className="max-h-[58vh] gap-0 overflow-hidden border-0 border-t border-white/10 bg-[#18181B]/85 py-0 backdrop-blur-xl">
            <NoiseOverlay />
            <CardContent className="relative max-h-[58vh] overflow-y-auto px-0 py-0">
              {facts.length > 0 ? (
                <ul className="divide-y divide-white/10">
                  {facts.map((fact, i) => (
                    <SourceRow key={fact.id} fact={fact} index={i} />
                  ))}
                </ul>
              ) : (
                <p className="px-5 py-[16px] text-[13px] leading-[1.5] text-white/40">
                  No qualifying facts were found near you, so this walk runs in
                  Notice mode. Notice mode makes no claims about what is around
                  you — it only offers invitations.
                </p>
              )}

              <Hairline />

              {/* stage timings: which dependency is slow, on the day it is slow */}
              <div className="flex flex-wrap items-baseline gap-x-[18px] gap-y-[6px] px-5 py-[14px]">
                {audioControl}
                <TechLabel>mode {mode.toLowerCase()}</TechLabel>
                {timings &&
                  Object.entries(timings).map(([stage, ms]) => (
                    <TechLabel key={stage}>
                      {stage} {ms}ms
                    </TechLabel>
                  ))}
                {voice.detail && (
                  <TechLabel tone="accent">{voice.detail}</TechLabel>
                )}
              </div>

              {simulation}
            </CardContent>
          </Card>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
