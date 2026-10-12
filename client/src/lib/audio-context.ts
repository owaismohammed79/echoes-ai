/**
 * The app's single AudioContext.
 *
 * One context for the whole process, created and resumed inside a user gesture.
 * Two reasons it is shared rather than made per-caller:
 *
 *  - constructing a context per tap leaks one per walk and eventually exhausts
 *    the browser's limit, after which `resume()` fails silently — the failure
 *    this exists to prevent;
 *  - on iOS a context created *after* an `await` (outside the unlocking gesture)
 *    stays suspended, and a suspended context runs no graph — so a VAD with its
 *    own context would meter silence forever and never detect a barge-in.
 *
 * Both the playback unlock and the voice-activity analyser therefore read from
 * here, so the one context that was unlocked by the Start tap is the one that
 * meters the microphone.
 */

let ctx: AudioContext | null = null;

/** The shared context, created and resumed on first call. Null if unsupported. */
export function getAudioContext(): AudioContext | null {
  try {
    const Ctx =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctx) return null;
    if (!ctx) ctx = new Ctx();
    if (ctx.state !== "running") void ctx.resume();
    return ctx;
  } catch {
    /* autoplay policy refused the context; the element path may still work */
    return null;
  }
}
