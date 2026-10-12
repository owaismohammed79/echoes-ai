import { Hairline, TechLabel } from "@/components/echoes/tech-label";
import type { Mode, VoicePath } from "@/lib/demo-engine";

function Row({
  label,
  value,
  onClick,
}: {
  label: string;
  value: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex min-h-[42px] w-full items-baseline justify-between px-5 text-left"
    >
      <TechLabel>{label}</TechLabel>
      <TechLabel tone="ink">{value}</TechLabel>
    </button>
  );
}

/**
 * Present only because there is no backend attached yet. It lets the two
 * behaviours that are otherwise unreachable in a demo — the deterministic
 * refusal into Notice mode, and the voice fallback — be seen rather than
 * taken on trust. Delete this with `demoEngine`.
 */
export function SimulationStrip({
  mode,
  onModeChange,
  voice,
  onVoiceChange,
  disabled,
}: {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  voice: VoicePath;
  onVoiceChange: (voice: VoicePath) => void;
  disabled: boolean;
}) {
  return (
    <div className={disabled ? "opacity-40" : undefined}>
      <Hairline />
      <div className="px-5 pb-[2px] pt-[12px]">
        <TechLabel tone="faint">simulation · no backend attached</TechLabel>
      </div>
      <Row
        label="Fact path"
        value={mode === "STORY" ? "facts found" : "no facts → notice"}
        onClick={() => onModeChange(mode === "STORY" ? "NOTICE" : "STORY")}
      />
      <Row
        label="Voice path"
        value={voice === "native" ? "native" : "remote → fallback"}
        onClick={() => onVoiceChange(voice === "native" ? "remote" : "native")}
      />
      <div className="h-[10px]" />
    </div>
  );
}
