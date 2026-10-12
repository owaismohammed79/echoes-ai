/**
 * Voice failure triage.
 *
 * There are two genuinely different failures and the screen must not conflate
 * them, because the user's recovery is different in each case:
 *
 *  - the *remote* voice failed (ElevenLabs rate limit, 5xx, unconfigured, open
 *    circuit) while the browser voice still works  -> the walk continues, and
 *    the screen says so, which is the on-screen form of `use_native_tts: true`;
 *  - the *browser* voice failed too (autoplay policy, no engine) -> there is no
 *    audio at all, so the screen must say "text only" rather than imply a voice
 *    is speaking when none is.
 *
 * Deliberate cancellation is not a failure and never reaches this module.
 */

const REMOTE_FAILURES = new Set([
  "tts_rate_limited",
  "tts_provider_error",
  "tts_unconfigured",
  "tts_circuit_open",
]);

export interface VoiceStatus {
  /** Short badge for the transcript row. Null when everything is fine. */
  badge: string | null;
  /** Full readout for the source drawer. Null when everything is fine. */
  detail: string | null;
  /** True when there is no audio at all and the words exist only as text. */
  textOnly: boolean;
}

export function classifyVoiceFailure(reason: string | null): VoiceStatus {
  if (!reason) return { badge: null, detail: null, textOnly: false };

  if (REMOTE_FAILURES.has(reason)) {
    return {
      badge: "native voice",
      detail: `use_native_tts: true · ${reason}`,
      textOnly: false,
    };
  }

  return {
    badge: "text only",
    detail: `no audio available · ${reason}`,
    textOnly: true,
  };
}
