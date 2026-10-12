"""The kill-switch: a recorded walk that costs nothing.

Implements AGENTS.md 4 ("If budgets are exceeded, seamlessly fall back to a
clearly labeled recorded demo mode").

Two things this module refuses to do:

* **It never calls a provider.** No Gemma, no ElevenLabs, no Wikipedia. Demo mode
  exists because those calls are no longer affordable, so a demo path that made
  one would defeat itself.
* **It never pretends to be live.** Every frame carries ``demo: true`` and the
  response carries ``X-Demo-Mode: True``, so the client can label it. A recorded
  walk presented as a generated one would be the same class of dishonesty as an
  uncited sentence.

The facts below describe real, well-known landmarks with their real Wikipedia
sources, so the *content* is not fabricated -- only the manner of its delivery
is canned. That keeps the demo honest about the one thing the product is for.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from app.models import FactObject
from app.services.prompts import PEDESTRIAN_REMINDER

logger = logging.getLogger("echoes.demo")

__all__ = [
    "DEMO_ANSWER",
    "DEMO_FACTS",
    "DEMO_MODE",
    "DEMO_SEGMENTS",
    "DEMO_TOTAL_CHARS",
    "REMINDER_CLIP",
    "DemoManifest",
    "DemoSegment",
    "clip_paths",
    "demo_walk_frames",
    "load_manifest",
    "missing_clips",
    "static_root",
]

#: The mode every demo frame reports. STORY rather than NOTICE: the demo has
#: facts, so claiming Notice mode would misrepresent what Notice mode means.
DEMO_MODE = "STORY"

def static_root() -> Path:
    """``server/static`` -- resolved from this file, not the working directory."""
    return Path(__file__).resolve().parents[2] / "static"


DEMO_CLIP_DIR = "demo"
REMINDER_CLIP = "reminder.wav"
MANIFEST_NAME = "manifest.json"

#: What push-to-talk answers with under the kill-switch.
#:
#: A recorded line rather than a generated one, and deliberately an *assertion
#: about the system* rather than about the walker's surroundings. Notice Mode's
#: rules (AGENTS.md 2) do not stop applying because the budget ran out: a canned
#: line claiming anything about what is in front of the user would be worse than
#: saying nothing.
DEMO_ANSWER = (
    "Echoes is running its recorded demonstration, so I cannot answer questions "
    "about where you are right now."
)


@dataclass(frozen=True)
class DemoSegment:
    """One pre-recorded sentence: its text, its citation, its clip."""

    text: str
    citations: tuple[str, ...]
    source_urls: tuple[str, ...]
    clip: str


#: Mock FactObjects, in the exact shape the live pipeline produces. The ids match
#: the citations in DEMO_SEGMENTS, so the validator's contract (every sentence
#: cites a fact that exists) holds in the demo too.
DEMO_FACTS: tuple[FactObject, ...] = (
    FactObject(
        fact_id="a1c4f2",
        title="Nelson's Column",
        snippet=(
            "A 52-metre granite column raised in 1843 to honour Admiral Horatio "
            "Nelson, watching over the square from a Corinthian capital."
        ),
        source_url="https://en.wikipedia.org/wiki/Nelson%27s_Column",
        distance_m=21.6,
    ),
    FactObject(
        fact_id="7d90be",
        title="St Martin-in-the-Fields",
        snippet=(
            "A Greek Revival church on the north-east corner, its portico raised "
            "in the 1720s over an earlier Tudor building."
        ),
        source_url="https://en.wikipedia.org/wiki/St_Martin-in-the-Fields",
        distance_m=180.4,
    ),
    FactObject(
        fact_id="c33a18",
        title="The Trafalgar Fountains",
        snippet=(
            "Two fountain basins added in 1845 were rebuilt in 1939 with pumps "
            "that can lift 46 litres a second."
        ),
        source_url="https://en.wikipedia.org/wiki/Trafalgar_Square",
        distance_m=62.1,
    ),
)

DEMO_SEGMENTS: tuple[DemoSegment, ...] = (
    DemoSegment(
        text=(
            "This is a recorded demonstration of Echoes, playing because the "
            "day's API budget is spent. You are a few steps from a granite "
            "column raised in 1843 for Admiral Nelson."
        ),
        citations=("a1c4f2",),
        source_urls=("https://en.wikipedia.org/wiki/Nelson%27s_Column",),
        clip="segment-01.wav",
    ),
    DemoSegment(
        text=(
            "Further along the north-east corner stands St Martin-in-the-Fields, "
            "its portico added in the 1720s over a Tudor church."
        ),
        citations=("7d90be",),
        source_urls=("https://en.wikipedia.org/wiki/St_Martin-in-the-Fields",),
        clip="segment-02.wav",
    ),
    DemoSegment(
        text=(
            "The two fountain basins beside you were rebuilt in 1939, and their "
            "pumps can lift forty-six litres of water every second."
        ),
        citations=("c33a18",),
        source_urls=("https://en.wikipedia.org/wiki/Trafalgar_Square",),
        clip="segment-03.wav",
    ),
)

#: What a live walk of this length would have cost ElevenLabs. Reported so the
#: saving is visible rather than assumed.
DEMO_TOTAL_CHARS = sum(len(s.text) for s in DEMO_SEGMENTS)


@dataclass(frozen=True)
class DemoManifest:
    """``static/demo/manifest.json``, or a truthful default when it is absent."""

    recorded: bool = False
    note: str = ""
    present: bool = False


def load_manifest() -> DemoManifest:
    """Read the demo manifest. A malformed manifest must not break the walk."""
    path = static_root() / DEMO_CLIP_DIR / MANIFEST_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DemoManifest(
            recorded=False, note="manifest missing; clips are placeholders", present=False
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        logger.warning("demo manifest unreadable: %s", exc)
        return DemoManifest(recorded=False, note=f"unreadable: {exc}", present=False)
    return DemoManifest(
        recorded=bool(raw.get("recorded", False)),
        note=str(raw.get("note", "")),
        present=True,
    )


def clip_paths() -> tuple[Path, ...]:
    """Every audio file demo mode expects to serve."""
    root = static_root()
    demo_dir = root / DEMO_CLIP_DIR
    return tuple([demo_dir / s.clip for s in DEMO_SEGMENTS] + [root / REMINDER_CLIP])


def missing_clips() -> tuple[str, ...]:
    """Clips that are referenced but not on disk.

    Surfaced through ``/api/health`` because a kill-switch that fires into 404s
    is worse than no kill-switch: the user hears silence and cannot tell why.
    """
    root = static_root()
    return tuple(str(p.relative_to(root)) for p in clip_paths() if not p.is_file())


def demo_walk_frames(
    index_offset: int = 0, session_id: str | None = None
) -> Iterator[dict[str, Any]]:
    """The demo walk as NDJSON frames, in the same shape the live walk streams.

    Same shape on purpose: the kill-switch must be invisible to the client apart
    from the label. A separate demo contract would be a second thing to keep in
    sync, and would rot the first time the live frames changed.
    """
    for index, segment in enumerate(DEMO_SEGMENTS):
        yield {
            "type": "sentence",
            "index": index + index_offset,
            "text": segment.text,
            "citations": list(segment.citations),
            "source_urls": list(segment.source_urls),
            # The clip is pre-recorded, so no provider was asked to speak.
            "use_native_tts": False,
            "mode": DEMO_MODE,
            "audio_url": f"/static/{DEMO_CLIP_DIR}/{segment.clip}",
            "demo": True,
        }

    yield {
        "type": "reminder",
        "text": PEDESTRIAN_REMINDER,
        "audio_url": f"/static/{REMINDER_CLIP}",
        "demo": True,
    }
    yield {
        "type": "done",
        "mode": DEMO_MODE,
        "degraded": False,
        "used_static_fallback": True,
        "session_id": session_id,
        "rejections": [],
        "demo": True,
    }
