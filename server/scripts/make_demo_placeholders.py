"""Generate the demo clips as valid silent WAVs, sized to their script.

Run this to (re)create the placeholders that ``app/services/demo.py`` points at:

    python scripts/make_demo_placeholders.py

Why a generator rather than checked-in silence: the kill-switch needs *files* at
those paths, and a silent file that was produced by a script whose input is the
script text is auditable in a way an opaque blob is not. Each clip is a real,
decodable WAV of roughly the duration its sentence takes to speak, so the queue,
the transport and the client timings all behave as they would with a recording.

These are placeholders and the manifest says so (``"recorded": false``).
``/api/health`` reports that flag, so a demo deployment cannot quietly ship
silence thinking it shipped a recording. Replace them with real recordings and
set ``"recorded": true``.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.demo import (  # noqa: E402  (path set above)
    DEMO_CLIP_DIR,
    DEMO_SEGMENTS,
    MANIFEST_NAME,
    REMINDER_CLIP,
    static_root,
)
from app.services.prompts import PEDESTRIAN_REMINDER  # noqa: E402

#: Telephony-grade mono. Enough for a placeholder; replace with the real master.
RATE = 8000
CHANNELS = 1
SAMPLE_WIDTH = 2

#: ~170 words/minute with short words, so the clip length tracks the script.
CHARS_PER_SECOND = 14.0


def seconds_for(text: str) -> float:
    """Roughly how long `text` takes to say."""
    return max(1.0, len(text) / CHARS_PER_SECOND)


def write_silence(path: Path, seconds: float) -> int:
    """Write `seconds` of valid, silent PCM. Returns the byte size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(RATE * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(RATE)
        handle.writeframes(b"\x00\x00" * frames)
    return path.stat().st_size


def main() -> int:
    root = static_root()
    demo_dir = root / DEMO_CLIP_DIR

    clips = []
    for index, segment in enumerate(DEMO_SEGMENTS):
        seconds = seconds_for(segment.text)
        size = write_silence(demo_dir / segment.clip, seconds)
        clips.append(
            {
                "index": index,
                "clip": segment.clip,
                "seconds": round(seconds, 2),
                "bytes": size,
            }
        )
        print(f"  {segment.clip:18} {seconds:5.2f}s  {size:>8} bytes")

    reminder_seconds = seconds_for(PEDESTRIAN_REMINDER)
    reminder_size = write_silence(root / REMINDER_CLIP, reminder_seconds)
    print(f"  {REMINDER_CLIP:18} {reminder_seconds:5.2f}s  {reminder_size:>8} bytes")

    manifest = {
        "recorded": False,
        "note": (
            "Generated silent placeholders. Replace with the recorded demo audio "
            "and set recorded=true; /api/health reports this flag."
        ),
        "rate_hz": RATE,
        "channels": CHANNELS,
        "sample_width_bytes": SAMPLE_WIDTH,
        "reminder": REMINDER_CLIP,
        "segments": clips,
    }
    manifest_path = demo_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  {MANIFEST_NAME:18} written (recorded=false)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
