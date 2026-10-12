import { useCallback, useEffect, useRef } from "react";

import { MouseMascot } from "@/components/echoes/mouse-mascot";

/**
 * Pocket Mode.
 *
 * An ultra-dark, low-power overlay for a walk that is being listened to with
 * the phone in a pocket. It replaces the live UI — which is bright, busy, and
 * pointless when nobody is looking at it — with an almost-black screen, a dim
 * glow, and one instruction. Black pixels cost the least power on OLED, which is
 * the whole point of a screen the walker is not reading.
 *
 * Unlocking is a deliberate two-tap gesture rather than a single tap: a phone in
 * a pocket gets jostled constantly, and a single touch would unlock it by
 * accident.
 */

/** Two taps inside this window count as the unlock gesture. */
const UNLOCK_WINDOW_MS = 700;

export interface PocketModeOverlayProps {
  open: boolean;
  onClose: () => void;
}

export function PocketModeOverlay({ open, onClose }: PocketModeOverlayProps) {
  // The first tap's timestamp. Reset whenever the overlay opens, so a stray tap
  // from a previous session cannot pair with the first one of this one.
  const lastTapRef = useRef(0);
  useEffect(() => {
    if (open) lastTapRef.current = 0;
  }, [open]);

  const handleTap = useCallback(() => {
    const now = Date.now();
    if (now - lastTapRef.current <= UNLOCK_WINDOW_MS) {
      lastTapRef.current = 0;
      onClose();
      return;
    }
    lastTapRef.current = now;
  }, [onClose]);

  if (!open) return null;

  return (
    <button
      type="button"
      onClick={handleTap}
      aria-label="Pocket mode. Tap the screen twice to unlock."
      className="absolute inset-0 z-50 flex h-full w-full flex-col items-center justify-center gap-[26px] bg-black px-8 text-center"
    >
      {/* The mascot, dimmed to a faint glow so it costs almost nothing and
          still reads as the app rather than a dead screen. */}
      <MouseMascot className="h-16 w-16 opacity-25 [filter:drop-shadow(0_0_20px_rgba(16,185,129,0.35))]" />
      <div>
        <p className="font-mono text-[10px] tracking-[0.16em] text-[#10B981]/60 uppercase">
          pocket mode
        </p>
        <p className="mt-[10px] text-[15px] leading-[1.5] text-white/45">
          Phone active in pocket. Tap screen twice to unlock.
        </p>
      </div>
    </button>
  );
}
