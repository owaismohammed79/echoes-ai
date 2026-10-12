import { Hairline, TechLabel } from "@/components/echoes/tech-label";
import { MouseMascot } from "@/components/echoes/mouse-mascot";
import { cn } from "@/lib/utils";

/**
 * The narration-outage modal.
 *
 * Shown when `/api/walk` answers `LLM_TEMPORARY_OUTAGE`: the Gemma -> Groq
 * cascade spent its whole retry budget, so no sentence was ever streamed. That
 * is a delay, not a broken walk, and the copy says so — with a single Retry.
 */

/** The line the walker reads. Fixed, so the message is always the same one. */
const BREATHER_COPY =
  "Gemma is taking a quick breather! 🦔 Please tap Retry in a few moments.";

export interface LlmOutageModalProps {
  /** The server's own message, shown as detail when it adds anything. */
  message: string;
  onRetry: () => void;
  onDismiss: () => void;
}

export function LlmOutageModal({
  message,
  onRetry,
  onDismiss,
}: LlmOutageModalProps) {
  const detail = message && message !== BREATHER_COPY ? message : null;

  return (
    <div
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="llm-outage-message"
      className="absolute inset-0 z-50 flex items-center justify-center bg-[#09090B]/90 px-5 backdrop-blur-sm"
    >
      <div className="w-full max-w-[440px] border border-white/10 bg-[#101010] p-5 shadow-[0_0_60px_rgba(16,185,129,0.08)]">
        <TechLabel tone="accent">gemma outage</TechLabel>

        <Hairline className="my-[14px]" />

        <div className="flex items-center gap-[18px]">
          <MouseMascot className="h-20 w-20" />
          <div>
            <p
              id="llm-outage-message"
              className="text-[15px] leading-[1.45] text-white/85"
            >
              {BREATHER_COPY}
            </p>
            {detail && (
              <p className="mt-[8px] font-mono text-[10px] leading-[1.5] tracking-[0.04em] text-white/35">
                {detail}
              </p>
            )}
          </div>
        </div>

        <div className="mt-[18px] flex gap-[10px]">
          <button
            type="button"
            onClick={onRetry}
            className={cn(
              "h-[48px] flex-1 border border-[#10B981] font-mono text-[12px] uppercase tracking-[0.053em] text-[#10B981]",
              "transition-colors hover:bg-[#10B981]/10"
            )}
          >
            Retry
          </button>
          <button
            type="button"
            onClick={onDismiss}
            className="h-[48px] border border-white/10 px-[18px] font-mono text-[12px] tracking-[0.053em] text-white/45 uppercase transition-colors hover:text-white/80"
          >
            Not now
          </button>
        </div>
      </div>
    </div>
  );
}
