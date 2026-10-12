/**
 * Push-to-talk capture.
 *
 * RECORDING is a real recording: a `MediaRecorder` over a microphone stream, so
 * the clip handed to `transcribe` is the clip the user actually spoke rather
 * than a timer pretending. Groq's whisper-large-v3 wants a file, and this is it.
 *
 * When the microphone is unavailable the state machine still runs — the user
 * still gets an answer, from the transcript they'd have given anyway — but the
 * failure is *returned* and surfaced on screen. A prompt that says "recording"
 * while nothing is being captured is the same class of lie as a citation
 * without a source.
 */

export interface ClipCapture {
  /** Stop, release the microphone, and hand back what was recorded. */
  stop: () => Promise<Blob | null>;
  /** Drop everything without producing a clip. */
  discard: () => void;
}

export type CaptureStart = ClipCapture | { error: string };

/**
 * A recorder that never fires `stop` must not hang the release: the walk is
 * waiting on this call and the user is standing in the street.
 */
const STOP_TIMEOUT_MS = 1500;

export async function startCapture(): Promise<CaptureStart> {
  const media = navigator.mediaDevices;
  if (!media?.getUserMedia || typeof MediaRecorder === "undefined") {
    return { error: "mic_unsupported" };
  }

  let stream: MediaStream;
  try {
    stream = await media.getUserMedia({ audio: true });
  } catch (error) {
    const name = (error as { name?: string }).name;
    return {
      error:
        name === "NotAllowedError" || name === "SecurityError"
          ? "mic_denied"
          : "mic_unavailable",
    };
  }

  // The microphone light must go out on every path out of here, including the
  // failures below — a walk that leaves the mic open is a privacy bug.
  const release = () => stream.getTracks().forEach((track) => track.stop());

  let recorder: MediaRecorder;
  try {
    recorder = new MediaRecorder(stream);
  } catch {
    release();
    return { error: "mic_unavailable" };
  }

  const chunks: Blob[] = [];
  const stopped = new Promise<void>((resolve) => {
    recorder.onstop = () => resolve();
    recorder.onerror = () => resolve();
  });

  recorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) chunks.push(event.data);
  };

  try {
    recorder.start();
  } catch {
    release();
    return { error: "mic_unavailable" };
  }

  const halt = () => {
    try {
      if (recorder.state !== "inactive") recorder.stop();
    } catch {
      /* already stopped */
    }
  };

  return {
    stop: async () => {
      halt();
      await Promise.race([
        stopped,
        new Promise((resolve) => setTimeout(resolve, STOP_TIMEOUT_MS)),
      ]);
      release();
      if (!chunks.length) return null;
      return new Blob(chunks, { type: chunks[0].type || "audio/webm" });
    },
    discard: () => {
      halt();
      release();
    },
  };
}
