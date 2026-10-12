/**
 * Media Session.
 *
 * A walk happens with the phone in a pocket and the screen off, so the lock
 * screen is part of the interface, not an extra: it carries the title and the
 * transport controls that would otherwise require unlocking the device
 * mid-walk.
 *
 * Everything here is feature-detected and swallowed on failure. Safari before
 * 15 has no Media Session at all and several actions are unimplemented
 * elsewhere; `setActionHandler` *throws* for an unsupported action rather than
 * ignoring it, so each binding is guarded separately. A missing lock screen must
 * never take the walk down with it.
 */

const WALK_TITLE = "Echoes Walk";
const WALK_ARTIST = "Gemma Historian";

export interface WalkSessionHandlers {
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
}

function session(): MediaSession | undefined {
  if (typeof navigator === "undefined") return undefined;
  return navigator.mediaSession;
}

/** Publish the walk's identity and bind its transport controls. */
export function publishWalkSession(handlers: WalkSessionHandlers): void {
  const ms = session();
  if (!ms) return;

  try {
    const Metadata = (
      window as unknown as { MediaMetadata?: typeof MediaMetadata }
    ).MediaMetadata;
    if (Metadata) {
      ms.metadata = new Metadata({
        title: WALK_TITLE,
        artist: WALK_ARTIST,
        album: "Cited narration",
      });
    }
  } catch {
    /* no metadata support; the controls below may still bind */
  }

  const bind = (action: MediaSessionAction, run: () => void) => {
    try {
      ms.setActionHandler(action, run);
    } catch {
      /* this browser does not implement that action */
    }
  };

  bind("play", handlers.onPlay);
  bind("pause", handlers.onPause);
  bind("stop", handlers.onStop);

  setWalkSessionState("playing");
}

export function setWalkSessionState(
  state: "playing" | "paused" | "none"
): void {
  const ms = session();
  if (!ms) return;
  try {
    ms.playbackState = state;
  } catch {
    /* not settable here */
  }
}

/** Hand the lock screen back when the walk ends. */
export function clearWalkSession(): void {
  const ms = session();
  if (!ms) return;
  const unbind = (action: MediaSessionAction) => {
    try {
      ms.setActionHandler(action, null);
    } catch {
      /* never bound, or unsupported */
    }
  };
  unbind("play");
  unbind("pause");
  unbind("stop");
  try {
    ms.metadata = null;
  } catch {
    /* not settable here */
  }
  setWalkSessionState("none");
}
