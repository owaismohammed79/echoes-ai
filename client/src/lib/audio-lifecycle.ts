/**
 * Audio lifecycle helpers that have to work with the screen locked.
 *
 * Mobile browsers freeze JS timers in the background, but they do honour
 * `HTMLAudioElement.onended` and network fetches kicked off from that event.
 * Everything here is therefore either a pure decision or bytes for that
 * element — never a `setTimeout` that would go to sleep in a pocket.
 */

/** Server-dictated gap between narration cycles, in milliseconds. */
export const AMBIENT_PAUSE_MS = 75_000;

/** Post-walk Q&A window, played as a bed so the mic timeout survives lock. */
export const QA_GRACE_MS = 45_000;

export function shouldResumeOnVisible(args: {
  visibilityState: DocumentVisibilityState | string;
  walkActive: boolean;
  userPaused: boolean;
  elementPaused: boolean;
}): boolean {
  return (
    args.visibilityState === "visible" &&
    args.walkActive &&
    !args.userPaused &&
    args.elementPaused
  );
}

/** Hardware lock: stopping every track is what turns the mic indicator off. */
export function stopAllTracks(stream: MediaStream | null | undefined): void {
  stream?.getTracks().forEach((track) => track.stop());
}

/**
 * Near-silent brown-ish noise, long enough to keep iOS from dropping the
 * media session. Digital silence is treated as idle; a few LSBs of noise is not.
 */
export function makeAmbientBed(
  durationMs: number,
  sampleRate = 8000
): Blob {
  const seconds = Math.max(0.05, durationMs / 1000);
  const samples = Math.floor(seconds * sampleRate);
  const dataSize = samples * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeAscii = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, dataSize, true);

  let seed = 1;
  let last = 0;
  for (let i = 0; i < samples; i++) {
    seed = (seed * 16807) % 2147483647;
    const white = (seed / 2147483647) * 2 - 1;
    last = last * 0.98 + white * 0.02;
    view.setInt16(44 + i * 2, Math.max(-80, Math.min(80, last * 80)), true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}
