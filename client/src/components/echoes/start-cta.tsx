import { ShimmerButton } from "@/components/ui/shimmer-button";
import { primeAudio } from "@/hooks/use-walk-controller";

/**
 * The walk's entry point, and the only filled chromatic surface in the app.
 *
 * The magicui ShimmerButton supplies the sweep. What matters more than the
 * shimmer is the click handler: `primeAudio()` runs synchronously inside the
 * tap, before React has a chance to schedule anything, because iOS will not
 * grant audio playback to a context created after an await.
 */
export function StartCta({
  arming,
  onStart,
}: {
  arming: boolean;
  onStart: () => void;
}) {
  return (
    <ShimmerButton
      onClick={() => {
        // Order is the whole point: unlock, then start the (async) work.
        primeAudio();
        onStart();
      }}
      disabled={arming}
      aria-label="Start walk"
      shimmerColor="#FFFFFF59"
      shimmerDuration="4s"
      borderRadius="1px"
      background="#10B981"
      className="mx-auto h-[56px] w-full max-w-[380px] border-transparent px-0 font-mono text-[12px] tracking-[0.053em] text-[#09090B] uppercase disabled:opacity-60"
    >
      {arming ? "Locating…" : "Start Walk"}
    </ShimmerButton>
  );
}
