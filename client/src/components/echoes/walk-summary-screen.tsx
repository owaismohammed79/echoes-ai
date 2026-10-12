import { useMemo } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { LandmarkThumbnail } from "@/components/echoes/landmark-thumbnail";
import { formatMMSS } from "@/components/echoes/ScreenLockGuard";
import { Hairline, TechLabel } from "@/components/echoes/tech-label";
import { Card } from "@/components/ui/card";
import { NumberTicker } from "@/components/ui/number-ticker";
import type { Fact, Segment } from "@/lib/demo-engine";
import { cn } from "@/lib/utils";

/**
 * The walk, after it. Duration, the burn it represents, how the listening was
 * paced, and every landmark the walk actually cited — with the same `[[id]]`
 * badges the narrator used, so the summary ties back to the audio rather than
 * floating free of it.
 *
 * The data is a snapshot, not a live read: the controller clears its own state
 * when a walk ends, so the parent hands the summary its numbers at the moment it
 * opens and the screen never renders from a half-reset walk.
 */

export interface WalkSummaryData {
  /** Whole seconds the walk ran. */
  seconds: number;
  /** Whole seconds the screen was on during the walk. */
  screenOnSeconds: number;
  /** The grounded facts the walk surfaced. Empty in Notice mode. */
  facts: Fact[];
  /** The sentences that were narrated, in order. */
  segments: Segment[];
}

/** The standard walking estimate: ~4.5 kcal per minute, rounded. */
function estimateCalories(seconds: number): number {
  return Math.round((seconds / 60) * 4.5);
}

/** "14s", "12m 30s", or "12m" — the compact form the ratio badge reads in. */
function formatShortDuration(seconds: number): string {
  const safe = Math.max(0, Math.round(seconds));
  if (safe < 60) return `${safe}s`;
  const mm = Math.floor(safe / 60);
  const ss = safe % 60;
  return ss === 0 ? `${mm}m` : `${mm}m ${ss}s`;
}

/**
 * MM:SS, counted up through NumberTicker.
 *
 * NumberTicker writes a bare formatted number, so the colon and the leading
 * zeros are the frame around two tickers. The padding is decided from the
 * *final* value: for a seconds count under ten the leading zero is always shown
 * (the ticker animates 0..9 inside it, which reads correctly), and for ten or
 * more it is omitted.
 */
function Duration({ seconds }: { seconds: number }) {
  const mm = Math.floor(seconds / 60);
  const ss = seconds % 60;
  const cls = "text-[30px] leading-none text-white/90 dark:text-white/90";
  return (
    <span className="font-mono tabular-nums">
      {mm < 10 && "0"}
      <NumberTicker value={mm} delay={0} className={cls} />
      <span className="mx-[2px] text-white/30">:</span>
      {ss < 10 && "0"}
      <NumberTicker value={ss} delay={0} className={cls} />
    </span>
  );
}

interface PacingPoint {
  minute: number;
  listened: number;
}

/**
 * The pacing series: segments listened, accumulated. Per-sentence timestamps do
 * not exist in the transcript, so the walk's own duration is divided evenly
 * across the sentences — a straight line drawn from the only two things that are
 * known for certain, where it started and where it ended.
 */
function pacingSeries(seconds: number, segments: number): PacingPoint[] {
  if (segments <= 0) return [];
  const minutes = Math.max(seconds, 1) / 60;
  return Array.from({ length: segments + 1 }, (_, i) => ({
    minute: Number(((i / segments) * minutes).toFixed(2)),
    listened: i,
  }));
}

function PacingTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: PacingPoint }[];
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  return (
    <div className="border border-white/10 bg-[#18181B] px-[10px] py-[6px]">
      <TechLabel tone="muted">{point.minute.toFixed(1)} min</TechLabel>
      <p className="mt-[2px] font-mono text-[11px] tabular-nums text-[#10B981]">
        {point.listened} segments
      </p>
    </div>
  );
}

function TimeTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: { name: string; seconds: number } }[];
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  return (
    <div className="border border-white/10 bg-[#18181B] px-[10px] py-[6px]">
      <TechLabel tone="muted">{point.name}</TechLabel>
      <p className="mt-[2px] font-mono text-[11px] tabular-nums text-white/80">
        {formatShortDuration(point.seconds)}
      </p>
    </div>
  );
}

function LandmarkCard({ fact }: { fact: Fact }) {
  return (
    <article className="flex gap-[12px] border border-white/10 bg-[#101010] p-[12px]">
      {/* The Wikipedia pageimage, or the contour mark when there is none. */}
      <LandmarkThumbnail
        src={fact.imageUrl}
        alt={fact.title}
        className="h-16 w-16"
      />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-[10px] gap-y-[4px]">
          <p className="text-[14px] leading-[1.38] text-white/85">
            {fact.title}
          </p>
          {/* The same [[fact_id]] the narrator cites inline. */}
          <code className="border border-[#10B981]/30 px-[5px] py-[1px] font-mono text-[10px] tracking-[0.064em] text-[#10B981]">
            [[{fact.id}]]
          </code>
        </div>
        <p className="mt-[6px] text-[13px] leading-[1.45] text-white/40">
          {fact.detail}
        </p>
        <div className="mt-[10px] flex flex-wrap items-baseline gap-x-[12px] gap-y-[4px]">
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
    </article>
  );
}

export function WalkSummaryScreen({
  data,
  onStartNew,
  onClose,
}: {
  data: WalkSummaryData;
  /** Back to the permissions/home screen for a fresh walk. */
  onStartNew: () => void;
  /** Close the summary and return to the front door for a fresh walk. */
  onClose: () => void;
}) {
  const calories = estimateCalories(data.seconds);
  const series = useMemo(
    () => pacingSeries(data.seconds, data.segments.length),
    [data.seconds, data.segments.length]
  );
  // Pocket time is the walk that was not watched: whatever the screen was off
  // for. It cannot be negative, even if a screen-on reading overshoots.
  const pocketSeconds = Math.max(0, data.seconds - data.screenOnSeconds);
  const screenFreePct =
    data.seconds > 0
      ? Math.min(100, Math.max(0, Math.round((pocketSeconds / data.seconds) * 100)))
      : 100;
  const timeData = [
    { name: "pocket time", seconds: pocketSeconds },
    { name: "screen-on", seconds: data.screenOnSeconds },
  ];

  return (
    <div className="absolute inset-0 z-50 flex flex-col bg-[#09090B]">
      <header className="flex items-baseline justify-between px-5 pt-[20px] pb-[14px]">
        <TechLabel tone="ink">walk summary</TechLabel>
        <TechLabel tone="faint">{data.segments.length} segments</TechLabel>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-[132px]">
        {/* the score: how much of the walk was listened to, not watched */}
        <div className="mb-[14px] border border-[#10B981]/30 bg-[#10B981]/[0.06] px-5 py-[14px]">
          <TechLabel tone="accent">screen time ratio</TechLabel>
          <p className="mt-[8px] text-[15px] leading-[1.4] text-white/90">
            Screen Time Ratio:{" "}
            <span className="font-mono tabular-nums text-white/70">
              {formatShortDuration(data.screenOnSeconds)}
            </span>{" "}
            /{" "}
            <span className="font-mono tabular-nums text-white/70">
              {formatShortDuration(data.seconds)}
            </span>{" "}
            — <span className="text-[#10B981]">{screenFreePct}% Screen-Free Walk!</span>
          </p>
        </div>

        {/* metrics */}
        <Card className="gap-0 overflow-hidden border-white/10 bg-[#18181B] py-0">
          <div className="grid grid-cols-2">
            <div className="px-5 py-[16px]">
              <TechLabel tone="muted">duration</TechLabel>
              <div className="mt-[10px]">
                <Duration seconds={data.seconds} />
              </div>
              <TechLabel tone="faint" className="mt-[8px] block">
                mm:ss
              </TechLabel>
            </div>
            <div className="border-l border-white/10 px-5 py-[16px]">
              <TechLabel tone="muted">estimated burn</TechLabel>
              <div className="mt-[10px] flex items-baseline gap-[6px]">
                <NumberTicker
                  value={calories}
                  delay={0}
                  className="text-[30px] leading-none text-white/90 dark:text-white/90"
                />
                <TechLabel tone="faint">kcal</TechLabel>
              </div>
              <TechLabel tone="faint" className="mt-[8px] block">
                4.5 kcal / min
              </TechLabel>
            </div>
          </div>
          <Hairline />
          <div className="flex items-baseline justify-between px-5 py-[14px]">
            <TechLabel tone="muted">screen-on time</TechLabel>
            <span className="font-mono text-[13px] tabular-nums text-white/80">
              {formatMMSS(data.screenOnSeconds)}
            </span>
          </div>
        </Card>

        {/* pocket time vs screen time, the ratio above, drawn to scale */}
        <Card className="mt-[14px] gap-0 overflow-hidden border-white/10 bg-[#18181B] py-0">
          <div className="flex items-baseline justify-between px-5 pt-[16px] pb-[10px]">
            <TechLabel tone="ink">pocket vs screen</TechLabel>
            <TechLabel tone="faint">seconds</TechLabel>
          </div>
          <Hairline />
          <div className="h-[124px] w-full px-3 pt-[12px] pb-[8px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={timeData}
                layout="vertical"
                margin={{ top: 4, right: 16, bottom: 0, left: 8 }}
                barCategoryGap={10}
              >
                <XAxis type="number" hide />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={74}
                  tick={{ fill: "rgba(250,250,250,0.45)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  cursor={{ fill: "rgba(250,250,250,0.04)" }}
                  content={<TimeTooltip />}
                />
                <Bar dataKey="seconds" radius={1} isAnimationActive={false}>
                  {/* Pocket time in the accent; screen-on in flat zinc. */}
                  <Cell fill="#10B981" />
                  <Cell fill="#3F3F46" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* pacing */}
        <Card className="mt-[14px] gap-0 overflow-hidden border-white/10 bg-[#18181B] py-0">
          <div className="flex items-baseline justify-between px-5 pt-[16px] pb-[10px]">
            <TechLabel tone="ink">pacing</TechLabel>
            <TechLabel tone="faint">segments listened</TechLabel>
          </div>
          <Hairline />
          <div className="px-3 pt-[14px] pb-[10px]">
            {series.length > 0 ? (
              <div className="h-[132px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={series}
                    margin={{ top: 4, right: 10, bottom: 0, left: -22 }}
                  >
                    <defs>
                      <linearGradient id="echoes-pacing" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#10B981" stopOpacity={0.35} />
                        <stop offset="100%" stopColor="#10B981" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="minute"
                      tickFormatter={(value: number) => `${value.toFixed(0)}m`}
                      tick={{ fill: "rgba(250,250,250,0.35)", fontSize: 9 }}
                      tickLine={false}
                      axisLine={{ stroke: "rgba(250,250,250,0.1)" }}
                    />
                    <YAxis
                      allowDecimals={false}
                      width={30}
                      tick={{ fill: "rgba(250,250,250,0.35)", fontSize: 9 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      content={<PacingTooltip />}
                      cursor={{ stroke: "rgba(250,250,250,0.15)" }}
                    />
                    <Area
                      type="monotone"
                      dataKey="listened"
                      stroke="#10B981"
                      strokeWidth={1.5}
                      fill="url(#echoes-pacing)"
                      isAnimationActive={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="px-2 py-[24px] text-center text-[13px] leading-[1.5] text-white/40">
                No segments were narrated, so there is no pacing to chart.
              </p>
            )}
          </div>
        </Card>

        {/* landmarks */}
        <Card className="mt-[14px] gap-0 overflow-hidden border-white/10 bg-[#18181B] py-0">
          <div className="flex items-baseline justify-between px-5 pt-[16px] pb-[12px]">
            <TechLabel tone="ink">landmark gallery</TechLabel>
            <TechLabel tone="faint">
              {data.facts.length
                ? `${data.facts.length} cited`
                : "none found"}
            </TechLabel>
          </div>
          <Hairline />
          {data.facts.length > 0 ? (
            <div className="grid gap-[10px] p-[12px]">
              {data.facts.map((fact) => (
                <LandmarkCard key={fact.id} fact={fact} />
              ))}
            </div>
          ) : (
            <p className="px-5 py-[16px] text-[13px] leading-[1.5] text-white/40">
              No qualifying facts were found near you, so this walk ran in Notice
              mode. Notice mode makes no claims about what is around you — it only
              offers invitations.
            </p>
          )}
        </Card>
      </div>

      {/* CTAs */}
      <div
        className="absolute inset-x-0 bottom-0 border-t border-white/10 bg-[#09090B]/95 px-5 pt-[14px] backdrop-blur"
        style={{ paddingBottom: "max(14px, env(safe-area-inset-bottom))" }}
      >
        <button
          type="button"
          onClick={onStartNew}
          className="h-[52px] w-full border border-transparent bg-[#10B981] font-mono text-[12px] uppercase tracking-[0.053em] text-[#09090B] transition-opacity hover:opacity-90"
        >
          Start New Walk
        </button>
        <button
          type="button"
          onClick={onClose}
          className={cn(
            "mt-[10px] h-[46px] w-full border border-white/15 font-mono text-[12px] uppercase tracking-[0.053em] text-white/70",
            "transition-colors hover:border-white/30 hover:text-white"
          )}
        >
          Close Summary
        </button>
      </div>
    </div>
  );
}
