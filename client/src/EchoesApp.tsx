import { useCallback, useEffect, useRef, useState } from "react";

import { AmbientState } from "@/components/echoes/ambient-state";
import { ConversationalOrb } from "@/components/echoes/conversational-orb";
import { PermissionsScreen } from "@/components/echoes/permissions-screen";
import { LlmOutageModal } from "@/components/echoes/llm-outage-modal";
import { PocketModeOverlay } from "@/components/echoes/pocket-mode";
import { ScreenLockGuard } from "@/components/echoes/ScreenLockGuard";
import { SimulationStrip } from "@/components/echoes/simulation-strip";
import { SourceDrawer } from "@/components/echoes/source-drawer";
import { StartCta } from "@/components/echoes/start-cta";
import { Hairline, TechLabel } from "@/components/echoes/tech-label";
import {
  WalkSummaryScreen,
  type WalkSummaryData,
} from "@/components/echoes/walk-summary-screen";
import { BorderBeam } from "@/components/ui/border-beam";
import { NumberTicker } from "@/components/ui/number-ticker";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import { useWalkController } from "@/hooks/use-walk-controller";
import { type Activity, type WalkEngine } from "@/lib/demo-engine";
import { realEngine } from "@/lib/real-engine";
import { useScreenWakeLock } from "@/lib/wake-lock";
import { cn } from "@/lib/utils";
import { classifyVoiceFailure } from "@/lib/voice-status";

/** The centre word per state. One word, plus a mono qualifier. */
const STATE_COPY: Record<Activity | "live", { title: string; sub: string }> = {
  ready: { title: "Ready", sub: "tap start walk" },
  live: { title: "Ready", sub: "speak any time" },
  speaking: { title: "Speaking", sub: "grounded narrative" },
  paused: { title: "Paused", sub: "tap ring to resume" },
  // While the button is held, the sub-line names the engine rather than
  // repeating the instruction the orb is already showing.
  listening: { title: "Listening", sub: "whisper large v3" },
  thinking: { title: "Thinking", sub: "transcribing" },
  answering: { title: "Answering", sub: "grounded answer" },
};

const IDLE_COPY = "A voice that tells you where you are standing, and cites its sources.";

export default function EchoesApp({ engine = realEngine }: { engine?: WalkEngine }) {
  const walk = useWalkController(engine);
  const live = walk.phase === "live";
  // The guard watches the screen only while a walk is actually running: once the
  // walk completes the screen-on clock stops and the summary takes over.
  const guardActive = live && !walk.completed;

  // Pocket Mode: the ultra-dark overlay for a walk being listened to with the
  // phone away. Kept here so it can be dismissed by the walk ending, and so the
  // wake lock keeps the tab alive underneath it.
  const [pocket, setPocket] = useState(false);
  useEffect(() => {
    if (!live) setPocket(false);
  }, [live]);

  // Hold the screen awake for the length of the walk, re-acquiring it whenever
  // the page comes back to the foreground.
  useScreenWakeLock(guardActive);

  // The permissions screen is the app's front door: the walker always lands
  // here first and the walk only initialises once they submit their mood,
  // duration and permissions. It stays mounted until the walk goes live.
  const [onboarding, setOnboarding] = useState(true);
  // Bumped on every return to the front door, so the permissions screen (and
  // the mascot inside it) remounts fresh rather than reusing a stale layout.
  const [resetKey, setResetKey] = useState(0);
  useEffect(() => {
    if (live) setOnboarding(false);
  }, [live]);

  // The summary is opened two ways: the walk finishing on its own, or a tap on
  // End Walk. It reads from a snapshot, because endWalk() clears the controller's
  // state — a summary rendered live would watch its own numbers disappear.
  const [summary, setSummary] = useState<WalkSummaryData | null>(null);
  const summaryShown = useRef(false);

  useEffect(() => {
    if (!walk.completed) {
      summaryShown.current = false;
      return;
    }
    if (summaryShown.current) return;
    summaryShown.current = true;
    setSummary({
      seconds: walk.seconds,
      screenOnSeconds: walk.screenOnSeconds,
      facts: walk.facts,
      segments: walk.segments,
    });
  }, [
    walk.completed,
    walk.seconds,
    walk.screenOnSeconds,
    walk.facts,
    walk.segments,
  ]);

  const handleEndWalk = useCallback(() => {
    setSummary({
      seconds: walk.seconds,
      screenOnSeconds: walk.screenOnSeconds,
      facts: walk.facts,
      segments: walk.segments,
    });
    walk.endWalk();
  }, [walk]);

  const handleStartNew = useCallback(() => {
    setSummary(null);
    summaryShown.current = false;
    walk.endWalk();
    setOnboarding(true);
    setResetKey((k) => k + 1);
  }, [walk]);

  const voice = classifyVoiceFailure(walk.fallback);
  // Once the walk is live, an idle ring means "the queue is done, ask me
  // something" — not the pre-walk invitation to start.
  const copy =
    live && walk.activity === "ready"
      ? STATE_COPY.live
      : (STATE_COPY[walk.activity] ?? STATE_COPY.ready);
  const total = walk.segments.length || 1;
  // The dial counts segments the walk has *reached*, so it only ever moves
  // forwards. Counting the current segment only while `speaking` made it rewind
  // every time the audio paused or push-to-talk interrupted it.
  const progress = !live ? 0 : walk.completed ? 1 : Math.min(1, (walk.index + 1) / total);
  const current = walk.segments[Math.min(walk.index, walk.segments.length - 1)];

  // The transcript follows the voice, not the walk index: `spokenText` is
  // whatever the engine is saying right now, which is a walk segment or a
  // push-to-talk answer. Reading and speaking stay in the same register.
  const transcriptText =
    live && current ? walk.spokenText || current.text : IDLE_COPY;
  const transcriptWords = transcriptText.split(/\s+/).filter(Boolean).length;
  // Words only crawl in while sound is actually moving. Paused, thinking and
  // completed all show the sentence whole, because re-reading it is the point.
  const revealing =
    live && (walk.activity === "speaking" || walk.activity === "answering");

  return (
    <div className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-[#09090B] text-white">
      {/* TOP — the name, quiet; the walk's technical readout on the right */}
      <header className="flex items-center justify-between px-5 pb-[18px] pt-[22px]">
        <div className="flex items-center gap-2 md:gap-3">
          <img
            src="/assets/echoes-logo.svg"
            alt="Echoes Logo"
            className="h-7 w-7 md:h-9 md:w-9"
          />
          <h1 className="text-[14px] tracking-tight text-white/40 md:text-[18px]">
            Echoes
          </h1>
        </div>
        <div className="flex items-baseline gap-[12px]">
          {walk.fix && (
            <TechLabel>
              {walk.fix.lat.toFixed(3)}, {walk.fix.lon.toFixed(3)}
            </TechLabel>
          )}
          <TechLabel tone={live ? "ink" : "muted"} className="md:text-[12px]">
            {live ? walk.mode.toLowerCase() : "no walk"}
          </TechLabel>

          {/* magicui Number Ticker: the walk's own clock, counting up in
              seconds. Real data, so it belongs in the technical readout. */}
          {live && (
            <span className="flex items-baseline gap-[3px]">
              <NumberTicker
                value={walk.seconds}
                delay={0}
                className="font-mono text-[10px] tracking-[0.064em] text-white/45 tabular-nums dark:text-white/45"
              />
              <TechLabel tone="faint">s</TechLabel>
            </span>
          )}
        </div>
      </header>

      {/* CENTER — one object, plus the sentence being spoken */}
      <main
        className={cn(
          "flex min-h-0 flex-1 flex-col items-center justify-center px-6",
          // The landing bottom is only the CTA; the live bottom also holds the
          // orb and the drawer, so each reserves just what it needs beneath the
          // transcript instead of one padding sized for the larger of the two.
          live ? "pb-[196px]" : "pb-[112px]"
        )}
      >
        <button
          type="button"
          onClick={walk.togglePause}
          disabled={!live}
          aria-label={walk.activity === "paused" ? "Resume walk" : "Pause walk"}
          className="disabled:cursor-default"
        >
          <AmbientState
            activity={live ? walk.activity : "ready"}
            progress={live ? progress : 0}
            title={live ? copy.title : STATE_COPY.ready.title}
            sub={live ? copy.sub : STATE_COPY.ready.sub}
          />
        </button>

        {/* the transcript: what is being said, in text. Also the a11y surface.
            It is the only panel in the system with a full 1px border, which is
            what lets the beam travel around it. */}
        <div className="relative mt-[24px] w-full max-w-[38ch] border border-white/10 px-[16px] pt-[12px] pb-[16px]">
          {/* magicui Border Beam: a light that circuits the panel only while
              audio is actually moving, in the single accent. */}
          {live && walk.activity === "speaking" && (
            <BorderBeam
              size={40}
              duration={7}
              borderWidth={1}
              colorFrom="#10B981"
              colorTo="#10B98100"
            />
          )}

          <div className="mb-[12px] flex items-baseline justify-between">
            <TechLabel>
              {!live
                ? "awaiting walk"
                : walk.completed
                  ? "walk complete"
                  : `segment ${String(walk.index + 1).padStart(2, "0")} / ${String(
                      walk.segments.length
                    ).padStart(2, "0")}`}
            </TechLabel>
            <span className="flex items-baseline gap-[12px]">
              {/* A microphone that failed is stated, never swallowed: the
                  answer the user hears was not built from their words. */}
              {walk.mic && <TechLabel tone="accent">{walk.mic}</TechLabel>}
              {walk.notice && <TechLabel tone="accent">{walk.notice}</TechLabel>}
              {voice.badge && <TechLabel tone="accent">{voice.badge}</TechLabel>}
              {live && !walk.completed && (
                <button
                  type="button"
                  onClick={() => setPocket(true)}
                  className="font-mono text-[10px] tracking-[0.064em] text-white/45 uppercase transition-colors hover:text-white/80"
                >
                  pocket mode
                </button>
              )}
              {live && (
                <button
                  type="button"
                  onClick={handleEndWalk}
                  className="font-mono text-[10px] tracking-[0.064em] text-white/45 uppercase transition-colors hover:text-white/80"
                >
                  end walk
                </button>
              )}
            </span>
          </div>

          <Hairline className="mb-[12px]" />

          <p
            aria-live="polite"
            className="min-h-[3.9em] text-[16px] leading-[1.45] text-white/60"
          >
            {/* aceternity Text Generate Effect, driven by the engine's real
                word boundaries instead of its own timer. */}
            <TextGenerateEffect
              words={transcriptText}
              revealed={revealing ? walk.spoken : transcriptWords}
              duration={0.34}
            />
          </p>
        </div>
      </main>

      {/* BOTTOM — CTA before the walk, the orb and the drawer during it */}
      <div className="absolute inset-x-0 bottom-0 flex flex-col items-center">
        {!live && (
          // Full width on purpose: a shrink-to-fit wrapper here collapses the
          // CTA to its own padding, because the button's `w-full` would be a
          // percentage of a box that is itself sized *by* that button.
          <div className="mb-[18px] w-full px-5">
            <StartCta arming={walk.phase === "arming"} onStart={() => setOnboarding(true)} />
          </div>
        )}

        {/* Reading and speaking are mutually exclusive on a 390px screen: the
            sheet covers the orb, so the orb leaves rather than hiding under it. */}
        {live && !walk.drawer && (
          <ConversationalOrb
            activity={walk.activity}
            listening={walk.listening}
            levelRef={walk.levelRef}
            muted={walk.muted}
            mic={walk.mic}
            onToggleMute={walk.toggleMute}
          />
        )}

        {live && (
          <SourceDrawer
            open={walk.drawer}
            onOpenChange={walk.setDrawer}
            facts={walk.facts}
            mode={walk.mode}
            timings={walk.timings}
            fallback={walk.fallback}
            refused={walk.refused}
            audioControl={
              walk.activity === "speaking" || walk.activity === "paused" ? (
                <button
                  type="button"
                  onClick={walk.togglePause}
                  className="font-mono text-[10px] tracking-[0.064em] text-white/70 uppercase transition-colors hover:text-white"
                >
                  {walk.activity === "paused" ? "resume audio" : "pause audio"}
                </button>
              ) : null
            }
            simulation={
              <SimulationStrip
                mode={walk.mode}
                onModeChange={walk.setMode}
                voice={walk.voicePath}
                onVoiceChange={walk.setVoicePath}
                disabled={false}
              />
            }
          />
        )}

        <div style={{ height: "max(14px, env(safe-area-inset-bottom))" }} />
      </div>

      {/* The pre-walk: permissions, then the generating screen. Mounted only
          until the walk is live, so it never fights the live UI for the screen. */}
      {onboarding && !live && (
        <PermissionsScreen
          key={resetKey}
          phase={walk.phase}
          firstSegmentReady={walk.segments.length > 0}
          onGenerate={(prefs) => walk.start(prefs)}
          onStartImmediately={() => setOnboarding(false)}
        />
      )}

      {/* The phone-in-pocket guard: the screen-on clock the summary reports on,
          and the reminder when a live walk is reopened mid-walk. Sits under the
          summary so the two never fight for the screen. */}
      <ScreenLockGuard
        active={guardActive}
        onScreenOnChange={walk.setScreenOnSeconds}
      />

      {summary && (
        <WalkSummaryScreen
          data={summary}
          onStartNew={handleStartNew}
          onClose={handleStartNew}
        />
      )}

      {/* Pocket Mode sits above the live UI, below nothing: while it is up the
          only input is the two-tap unlock. */}
      <PocketModeOverlay
        open={live && pocket}
        onClose={() => setPocket(false)}
      />

      {/* The narration cascade is down: nothing was said, so offer a Retry
          rather than a broken walk. */}
      {walk.outage && (
        <LlmOutageModal
          message={walk.outage}
          onRetry={walk.retryOutage}
          onDismiss={walk.dismissOutage}
        />
      )}
    </div>
  );
}
