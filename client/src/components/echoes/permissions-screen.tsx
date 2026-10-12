import { useCallback, useEffect, useState, type ReactNode } from "react";
import {
  Check,
  CloudRain,
  Compass,
  Loader2,
  MapPin,
  Mic,
  Waves,
  Zap,
  type LucideIcon,
} from "lucide-react";

import { MouseMascot } from "@/components/echoes/mouse-mascot";
import { Hairline, TechLabel } from "@/components/echoes/tech-label";
import { BackgroundBeams } from "@/components/ui/background-beams";
import { Ripple } from "@/components/ui/ripple";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import { primeAudio } from "@/hooks/use-walk-controller";
import type { Phase, WalkPrefs } from "@/lib/demo-engine";
import { cn } from "@/lib/utils";

/**
 * The landing page: what the walk is, how long it runs, how it should feel, and
 * the two permissions it needs. All of it is settled *before* the tap that
 * starts the walk, so the walk itself needs no screen time (AGENTS.md section 3).
 *
 * The walker's choices travel as `WalkPrefs` -- duration and mood -- into the
 * `/api/walk` and `/api/context` payloads, where the mood adapts the narrator's
 * tone and the duration frames the pace.
 *
 * The pre-walk is three states: choosing (`permissions`), asking
 * (`requesting`), and narrating (`generating`). Only the first two are local;
 * the third is the walk's own `arming` phase, so the screen never disagrees with
 * the controller about whether generation has begun.
 */

type Grant = "unknown" | "pending" | "granted" | "denied";

/** Rotated on the waiting screen, in the order the backend does the work. */
const STATUS_PHRASES = [
  "Pinpointing nearby coordinates...",
  "Scanning Wikipedia GeoSearch & Park registry...",
  "Assembling grounded historical narrative...",
];

/** How long each status phrase holds before the next one takes over. */
const PHRASE_MS = 1700;

const DURATION_MIN = 5;
const DURATION_MAX = 45;
const DEFAULT_DURATION = 15;
/** The labelled stops on the timeline, in minutes. */
const DURATION_TICKS = [5, 15, 30, 45] as const;

const DEFAULT_MOOD = "calm";

/** The moods the narrator can be tuned to. Keys match `MOOD_DIRECTIVES` server-side. */
const MOODS: {
  key: string;
  title: string;
  desc: string;
  icon: LucideIcon;
}[] = [
  { key: "energetic", title: "Energetic", desc: "Brisk, vivid, forward-moving.", icon: Zap },
  { key: "calm", title: "Calm", desc: "Unhurried and gentle.", icon: Waves },
  {
    key: "sad",
    title: "Sad / Struggling",
    desc: "Resilience found in the history.",
    icon: CloudRain,
  },
  {
    key: "reflective",
    title: "Reflective / Explore",
    desc: "Curious, opening outward.",
    icon: Compass,
  },
];

/**
 * A dynamic opener line, e.g. "GOOD AFTERNOON • SUNDAY, OCTOBER 11". Built with
 * `Intl.DateTimeFormat` so the date reads in the device's own locale.
 */
function timestampLabel(now: Date): string {
  const hour = now.getHours();
  const greeting =
    hour < 12
      ? "Good morning"
      : hour < 17
        ? "Good afternoon"
        : hour < 21
          ? "Good evening"
          : "Good night";
  const date = new Intl.DateTimeFormat("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
  }).format(now);
  return `${greeting} • ${date}`.toUpperCase();
}

/** Percentage along the duration timeline for a given number of minutes. */
function timelinePercent(minutes: number): number {
  return ((minutes - DURATION_MIN) / (DURATION_MAX - DURATION_MIN)) * 100;
}

/**
 * Snap a dragged value to the nearest allowed stop. The timeline is not evenly
 * spaced (5/15/30/45), so a single native `step` cannot express it -- the range
 * input stays continuous and the value is quantised here instead.
 */
function snapDuration(value: number): number {
  return DURATION_TICKS.reduce<number>(
    (best, tick) =>
      Math.abs(tick - value) < Math.abs(best - value) ? tick : best,
    DURATION_TICKS[0]
  );
}

/**
 * Hand-drawn doodles for the empty top-right corner: a star, a small gold star,
 * a dotted trail and a leaf, in muted emerald and gold. Purely decorative.
 */
function Doodles({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 176 96"
      fill="none"
      aria-hidden="true"
      className={className}
    >
      <path
        d="M14 14l3.2 8.6L26 26l-8.8 3.4L14 38l-3.2-8.6L2 26l8.8-3.4z"
        fill="#10B981"
        opacity="0.45"
      />
      <path
        d="M74 18l1.8 4.8 4.8 1.8-4.8 1.8L74 31l-1.8-4.6-4.8-1.8 4.8-1.8z"
        fill="#D4A24A"
        opacity="0.55"
      />
      <path
        d="M34 62q20-14 40 0t40 0"
        stroke="#10B981"
        strokeOpacity="0.4"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeDasharray="1 6"
      />
      <path
        d="M126 20q12-12 24 0-12 12-24 0z"
        fill="#10B981"
        opacity="0.35"
      />
      <path
        d="M126 20q12-12 24 0"
        stroke="#10B981"
        strokeOpacity="0.55"
        strokeWidth="1"
      />
    </svg>
  );
}

/**
 * Ask for a fix, once. Resolves true on a grant, false on a refusal or a
 * timeout — a permission prompt that is ignored must not hang the screen.
 */
function requestLocation(): Promise<boolean> {
  if (typeof navigator === "undefined" || !navigator.geolocation) {
    return Promise.resolve(false);
  }
  return new Promise<boolean>((resolve) => {
    const timer = window.setTimeout(() => resolve(false), 8000);
    navigator.geolocation.getCurrentPosition(
      () => {
        window.clearTimeout(timer);
        resolve(true);
      },
      () => {
        window.clearTimeout(timer);
        resolve(false);
      },
      { timeout: 7000, maximumAge: 60000, enableHighAccuracy: false }
    );
  });
}

/**
 * Ask for the microphone by opening a stream and closing it again. The tracks
 * are stopped on every path out: holding the mic open to prove we can is the
 * exact privacy cost this screen exists to avoid.
 */
async function requestMicrophone(): Promise<boolean> {
  const media = navigator.mediaDevices;
  if (!media?.getUserMedia) return false;
  try {
    const stream = await media.getUserMedia({ audio: true });
    stream.getTracks().forEach((track) => track.stop());
    return true;
  } catch {
    return false;
  }
}

/** One permission: what it is for, its state, and how to ask for it. */
function PermissionRow({
  icon,
  title,
  detail,
  status,
  onRequest,
}: {
  icon: ReactNode;
  title: string;
  detail: string;
  status: Grant;
  onRequest: () => void;
}) {
  const granted = status === "granted";
  const pending = status === "pending";
  const denied = status === "denied";

  return (
    <div className="flex items-center gap-[14px] px-5 py-[15px]">
      <span
        className={cn(
          "grid h-9 w-9 shrink-0 place-items-center border",
          granted
            ? "border-[#10B981]/40 text-[#10B981]"
            : "border-white/10 text-white/45"
        )}
      >
        {icon}
      </span>

      <div className="min-w-0 flex-1">
        <p className="text-[14px] leading-[1.35] text-white/85">{title}</p>
        <p className="mt-[4px] text-[12px] leading-[1.45] text-white/40">
          {detail}
        </p>
      </div>

      <button
        type="button"
        onClick={onRequest}
        disabled={granted || pending}
        aria-label={`${granted ? "Granted" : denied ? "Retry" : "Allow"} ${title}`}
        className={cn(
          "flex min-w-[74px] shrink-0 items-center justify-center gap-[6px] border px-[10px] py-[6px]",
          "font-mono text-[10px] tracking-[0.064em] uppercase transition-colors",
          granted
            ? "border-[#10B981]/40 text-[#10B981]"
            : "border-white/15 text-white/70 hover:border-white/30 hover:text-white disabled:opacity-60"
        )}
      >
        {granted ? (
          <>
            <Check size={12} strokeWidth={2} />
            granted
          </>
        ) : pending ? (
          <>
            <Loader2 size={12} className="animate-spin" />
            asking
          </>
        ) : denied ? (
          "retry"
        ) : (
          "allow"
        )}
      </button>
    </div>
  );
}

export function PermissionsScreen({
  phase,
  firstSegmentReady,
  onGenerate,
  onStartImmediately,
}: {
  /** The walk's phase. `arming` means the narrative fetch is in flight. */
  phase: Phase;
  /** True once the first segment has arrived from the backend stream. */
  firstSegmentReady: boolean;
  /** Permissions are settled and the user asked for the story. */
  onGenerate: (prefs: WalkPrefs) => void;
  /** Skip the rest of generation and reveal the walk now. */
  onStartImmediately: () => void;
}) {
  const [location, setLocation] = useState<Grant>("unknown");
  const [microphone, setMicrophone] = useState<Grant>("unknown");
  const [busy, setBusy] = useState(false);
  const [phrase, setPhrase] = useState(0);
  const [duration, setDuration] = useState<number>(DEFAULT_DURATION);
  const [mood, setMood] = useState<string>(DEFAULT_MOOD);
  // Read once, on mount: the greeting is about *now*, not a ticking clock.
  const [timestamp] = useState(() => timestampLabel(new Date()));

  const generating = phase === "arming";
  const granted = location === "granted" && microphone === "granted";

  // The status readout advances with the backend's real stages, not on a timer
  // pretending to be one: it holds the last phrase rather than looping, because
  // looping would claim work that is not happening.
  useEffect(() => {
    if (!generating) return;
    setPhrase(0);
    const id = window.setInterval(() => {
      setPhrase((i) => Math.min(i + 1, STATUS_PHRASES.length - 1));
    }, PHRASE_MS);
    return () => window.clearInterval(id);
  }, [generating]);

  const ask = useCallback(async (kind: "location" | "microphone") => {
    if (kind === "location") {
      setLocation("pending");
      setLocation((await requestLocation()) ? "granted" : "denied");
    } else {
      setMicrophone("pending");
      setMicrophone((await requestMicrophone()) ? "granted" : "denied");
    }
  }, []);

  const grantAll = useCallback(async () => {
    setBusy(true);
    const needLocation = location !== "granted";
    const needMicrophone = microphone !== "granted";
    if (needLocation) setLocation("pending");
    if (needMicrophone) setMicrophone("pending");

    const [loc, mic] = await Promise.all([
      needLocation ? requestLocation() : Promise.resolve(true),
      needMicrophone ? requestMicrophone() : Promise.resolve(true),
    ]);

    setLocation(loc ? "granted" : "denied");
    setMicrophone(mic ? "granted" : "denied");
    setBusy(false);
    return loc && mic;
  }, [location, microphone]);

  // Keyboard stepping between the four stops. A controlled range input that
  // snaps would otherwise trap the arrow keys between ticks, since +1 always
  // rounds straight back to where it started.
  const moveDuration = useCallback(
    (delta: number) => {
      const index = DURATION_TICKS.findIndex((tick) => tick === duration);
      const at = index < 0 ? 0 : index;
      const next = DURATION_TICKS[
        Math.min(DURATION_TICKS.length - 1, Math.max(0, at + delta))
      ];
      setDuration(next);
    },
    [duration]
  );

  const handleGenerate = useCallback(() => {
    // Synchronous, inside the gesture, before any await: iOS will not grant
    // playback to a context created after the permission prompts resolve. This
    // is the same discipline as the Start button's handler.
    primeAudio();
    const prefs: WalkPrefs = { duration, mood };
    if (granted) {
      onGenerate(prefs);
      return;
    }
    // Generate once the prompts have settled, granted or not. A refusal is not a
    // dead end: the walk degrades instead — the server fix stands in for a
    // refused location, and a refused mic only costs push-to-talk.
    void (async () => {
      await grantAll();
      onGenerate(prefs);
    })();
  }, [duration, grantAll, granted, mood, onGenerate]);

  // --- generating: the slick waiting state -------------------------------
  if (generating) {
    return (
      <div className="absolute inset-0 z-40 flex flex-col items-center justify-center overflow-hidden bg-gradient-to-b from-[#0A0A0A] to-[#111B15] px-6">
        <BackgroundBeams className="opacity-[0.14]" />

        <div className="relative flex h-[300px] w-[300px] items-center justify-center">
          <Ripple
            mainCircleSize={124}
            numCircles={4}
            mainCircleOpacity={0.2}
            className="opacity-80"
          />
          <span className="relative grid h-[132px] w-[132px] place-items-center rounded-full border border-[#10B981]/30">
            <Loader2 size={22} className="animate-spin text-[#10B981]" />
          </span>
        </div>

        <div className="relative mt-[26px] flex min-h-[3.2em] w-full max-w-[44ch] items-start justify-center text-center">
          {/* Keyed by phrase so each one replays its own reveal rather than
              inheriting the last one's finished state. */}
          <TextGenerateEffect
            key={phrase}
            words={STATUS_PHRASES[phrase]}
            className="font-mono text-[12px] uppercase tracking-[0.064em] text-white/60"
          />
        </div>

        {firstSegmentReady && (
          <button
            type="button"
            onClick={onStartImmediately}
            className="relative mt-[26px] border border-[#10B981] px-5 py-[10px] font-mono text-[11px] uppercase tracking-[0.053em] text-[#10B981] transition-colors hover:bg-[#10B981]/10"
          >
            Start Walk Immediately
          </button>
        )}
      </div>
    );
  }

  // --- permissions / requesting: the landing page ------------------------
  return (
    <div className="absolute inset-0 z-40 overflow-y-auto bg-gradient-to-b from-[#0A0A0A] to-[#111B15]">
      <div className="relative mx-auto flex min-h-full w-full max-w-[440px] flex-col px-5 pt-6 pb-10">
        {/* Header: the mascot watches the cursor; the doodles fill the corner. */}
        <header className="relative flex items-start justify-between">
          <MouseMascot className="h-20 w-20" />
          <Doodles className="pointer-events-none absolute -top-1 right-0 h-24 w-44" />
        </header>

        <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.22em] text-[#10B981]/80">
          {timestamp}
        </p>

        <h1 className="mt-3 font-serif text-[34px] leading-[1.08] text-white/95">
          Let&rsquo;s uncover <em className="italic text-[#10B981]">hidden echoes.</em>
        </h1>

        <p className="mt-3 max-w-[42ch] text-[13px] leading-[1.55] text-white/50">
          Gemma narrates the past. Audio streams based on your path. Your phone
          stays in your pocket.
        </p>

        {/* Duration: a draggable timeline, ticks at 5 / 15 / 30 / 45 minutes. */}
        <section className="mt-7 rounded-sm border border-white/10 bg-[#101010] p-4">
          <div className="flex items-baseline justify-between">
            <h2 className="font-serif text-[18px] text-white/90">How long</h2>
            <span className="font-mono text-[13px] tabular-nums text-[#10B981]">
              {duration} min
            </span>
          </div>

          <div className="relative mt-5 h-1.5 w-full rounded-full bg-white/10">
            <div
              className="absolute inset-y-0 left-0 rounded-full bg-[#10B981]"
              style={{ width: `${timelinePercent(duration)}%` }}
            />
            {DURATION_TICKS.map((tick) => (
              <span
                key={tick}
                className="absolute top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/25"
                style={{ left: `${timelinePercent(tick)}%` }}
              />
            ))}
            {/* The glowing highlight indicator, sitting on the emerald track. */}
            <span
              className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#10B981] shadow-[0_0_14px_rgba(16,185,129,0.85)] ring-4 ring-[#10B981]/20"
              style={{ left: `${timelinePercent(duration)}%` }}
            />
            <input
              type="range"
              min={DURATION_MIN}
              max={DURATION_MAX}
              step={1}
              value={duration}
              onChange={(event) => setDuration(snapDuration(Number(event.target.value)))}
              onKeyDown={(event) => {
                if (event.key === "ArrowRight" || event.key === "ArrowUp") {
                  event.preventDefault();
                  moveDuration(1);
                } else if (
                  event.key === "ArrowLeft" ||
                  event.key === "ArrowDown"
                ) {
                  event.preventDefault();
                  moveDuration(-1);
                } else if (event.key === "Home") {
                  event.preventDefault();
                  setDuration(DURATION_MIN);
                } else if (event.key === "End") {
                  event.preventDefault();
                  setDuration(DURATION_MAX);
                }
              }}
              aria-label="Walk duration in minutes"
              className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
            />
          </div>

          {/*
            The labels are absolutely positioned at the same percentages as the
            tick dots and centred with -translate-x-1/2, so each label sits flush
            under its dot. `justify-between` cannot do this: the stops are not
            evenly spaced, so it would anchor the first label's edge rather than
            its centre to the first tick.
          */}
          <div className="relative mt-3 h-4">
            {DURATION_TICKS.map((tick) => (
              <span
                key={tick}
                className="absolute -translate-x-1/2 font-mono text-[10px] uppercase tracking-[0.064em] text-white/35"
                style={{ left: `${timelinePercent(tick)}%` }}
              >
                {tick}m
              </span>
            ))}
          </div>
        </section>

        {/* Mood: two columns of cards; the selected one glows emerald. */}
        <section className="mt-7">
          <h2 className="font-serif text-[18px] text-white/90">Pick a mood</h2>
          <div className="mt-3 grid grid-cols-2 gap-3">
            {MOODS.map(({ key, title, desc, icon: Icon }) => {
              const selected = mood === key;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setMood(key)}
                  aria-pressed={selected}
                  className={cn(
                    "flex flex-col gap-2 border p-3 text-left transition-all",
                    selected
                      ? "border-[#10B981]/40 bg-[#10B981]/10 ring-1 ring-[#10B981]"
                      : "border-white/10 bg-[#101010] hover:border-white/25"
                  )}
                >
                  <Icon
                    size={18}
                    strokeWidth={1.6}
                    className={selected ? "text-[#10B981]" : "text-white/45"}
                  />
                  <span className="text-[13px] font-semibold text-white/90">
                    {title}
                  </span>
                  <span className="text-[11px] leading-[1.4] text-white/40">
                    {desc}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        {/* Permissions, then the tap that starts the walk. */}
        <div className="mt-7 overflow-hidden rounded-sm border border-white/10 bg-[#101010]">
          <div className="px-5 pt-[16px] pb-[12px]">
            <p className="text-[13px] leading-[1.5] text-white/55">
              Two permissions, then a story. Neither is stored.
            </p>
          </div>

          <Hairline />

          <PermissionRow
            icon={<MapPin size={15} strokeWidth={1.5} />}
            title="Location Access"
            detail="Required for hyper-local context."
            status={location}
            onRequest={() => void ask("location")}
          />

          <Hairline />

          <PermissionRow
            icon={<Mic size={15} strokeWidth={1.5} />}
            title="Microphone Access"
            detail="Required for hands-free questions."
            status={microphone}
            onRequest={() => void ask("microphone")}
          />

          <Hairline />

          <div className="px-5 py-[16px]">
            <button
              type="button"
              onClick={handleGenerate}
              disabled={busy}
              className="h-[52px] w-full bg-[#10B981] font-sans text-[13px] font-medium uppercase tracking-[0.08em] text-[#0A0A0A] transition-opacity hover:opacity-90 disabled:opacity-60"
            >
              {busy ? "Requesting…" : granted ? "Begin the walk" : "Grant & begin"}
            </button>
            <TechLabel tone="faint" className="mt-[10px] block text-center">
              {granted ? "both permissions granted" : "location and microphone"}
            </TechLabel>
          </div>
        </div>
      </div>
    </div>
  );
}
