"""What kind of thing a fact is, and how much a walk wants to hear about it.

AGENTS.md section 1 forbids hard filters that "might drop valid adjacent
history" and asks for *ranking signals* instead. This module is that signal. A
Wikipedia article's own title and categories are enough to tell a temple from a
school, so the walk can spend the handful of slots it has on the things a
walker would actually notice.

Measured at 12.9638, 77.5480, the nearest twenty articles are a locality, a
district, a division, a school and four metro stations. The temples, gardens and
basilica a walker would notice sit a little further out, and the material handed
to the narrator was the district and the school. Ordering by kind is what puts
the landmarks back in the narration -- without dropping anything: a demoted fact
is still there, and is still what a walk narrates when the neighbourhood really
is only a school and a revenue boundary.

Three passes, in this order, because each source of evidence fails differently:

1. **Categories, for events.** "The Stones in the Park" is a concert named after
a park, and "1973 UCI Cyclo-cross World Championships" is not a place at all.
What gives them away is being filed under a year ("1969 in London") or under a
concert, championship or tribute -- and a title that merely mentions a park
hides that, so this pass has to come first.
2. **The title.** Precise when it speaks ("Bengaluru division", "Hyde Park,
London"), which is why a school filed under "Catholic schools in India" is still
typed as a school rather than as a religious place.
3. **Categories, for places.** Not precise: a category says where a subject sits
as often as what it is. "Cities and towns in Tumakuru district" and "Memorials
to C. Rajagopalachari" describe a city and a neighbourhood. So this pass uses a
narrower vocabulary, checks settlement kinds before landmark kinds, and ignores
the words that mark containment -- otherwise a village is a revenue boundary and
Rajajinagar is a monument.
"""

from __future__ import annotations

import re
from typing import Final, Iterable

__all__ = ["CATEGORY_PATTERNS", "KIND_PATTERNS", "KIND_PRIORITY", "PLACE", "classify", "priority"]

#: What a fact is when nothing above it matched.
PLACE: Final[str] = "place"


def _words(*terms: str) -> re.Pattern[str]:
    """One pattern per kind, tolerant of the plurals a category name uses.

    Wikipedia's own category list says "Neighbourhoods in Bengaluru" and
    "Botanical gardens", so a pattern that only matched the singular would type
    a neighbourhood as an unclassified place.
    """
    return re.compile(r"\b(?:" + "|".join(terms) + r")(?:s|es)?\b", re.IGNORECASE)


#: ``(kind, pattern)`` for a title, most specific first. A "War Memorial Library"
#: is a memorial before it is a library, which is why order matters and why this
#: is a tuple rather than a dict. Events are checked before any place kind: a
#: title naming a marathon, tournament or tribute is an event article near enough
#: every time, and a park in its title ("Royal Parks Half Marathon") does not
#: change that.
KIND_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "event",
        _words(
            "championship", "tournament", "marathon", "festival", "protest",
            "tribute", "pilgrimage", "parade",
        ),
    ),
    (
        "water",
        _words(
            "lake", "lakelet", "pond", "river", "canal", "reservoir",
            "tank", "stream", "waterfall", "backwater", "lakefront",
        ),
    ),
    (
        "religious",
        _words(
            "temple", "mandir", "church", "cathedral", "basilica", "chapel",
            "mosque", "masjid", "dargah", "gurdwara", "synagogue",
            "monastery", "shrine", "ashram", "stupa", "pagoda", "vihara",
        ),
    ),
    (
        "monument",
        _words(
            "monument", "memorial", "statue", "sculpture", "obelisk", "column",
            "cenotaph", "fountain", "arch", "gate", "clock\\s+tower",
            "watch\\s+tower", "martyr",
        ),
    ),
    (
        "hill",
        _words("hill", "hillock", "peak", "mountain", "ridge", "summit", "valley"),
    ),
    (
        "station",
        _words("railway\\s+station"),
    ),
    (
        "park",
        _words(
            "park", "garden", "botanical", "zoo", "arboretum", "green",
            "common", "maidan", "promenade", "walkway",
        ),
    ),
    (
        "building",
        _words(
            "fort", "palace", "haveli", "mahal", "museum", "librar(?:y|ies)",
            "theatre", "theater", "opera", "planetarium", "gallery",
            "observatory", "stadium", "courthouse", "town\\s+hall",
            "building", "stepwell", "aqueduct",
        ),
    ),
    (
        "locality",
        _words(
            "neighbourhood", "neighborhood", "suburb", "localit(?:y|ies)",
            "hamlet", "village", "quarter", "extension",
        ),
    ),
    (
        "institution",
        _words(
            "school", "college", "universit(?:y|ies)", "institute", "academy",
            "hospital", "clinic", "organi[sz]ation", "agenc(?:y|ies)", "board",
            "commission", "bank", "software", "technolog(?:y|ies)",
            "laborator(?:y|ies)",
        ),
    ),
    (
        "administrative",
        _words(
            "district", "division", "constituenc(?:y|ies)", "assembl(?:y|ies)",
            "legislature", "municipal", "municipalit(?:y|ies)", "corporation",
            "taluk", "tehsil", "council", "province", "governorate", "ward",
            "zone",
        ),
    ),
)

#: Kinds matched on *categories*, in a different order and with a narrower
#: vocabulary than the title pass -- see the module docstring. Settlement kinds
#: come before landmark kinds, and the words that mark containment
#: (district, division, ward, zone) are absent, because in a category name they
#: describe where the subject sits rather than what it is.
CATEGORY_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "locality",
        _words(
            "neighbourhood", "neighborhood", "suburb", "localit(?:y|ies)",
            "village", "town", "hamlet", "street", "road",
        ),
    ),
    (
        "water",
        _words("lake", "pond", "river", "canal", "reservoir", "tank", "stream"),
    ),
    (
        "religious",
        _words(
            "temple", "church", "cathedral", "basilica", "mosque", "gurdwara",
            "synagogue", "monastery", "shrine",
        ),
    ),
    (
        "monument",
        _words("monument", "memorial", "statue", "obelisk", "cenotaph", "fountain"),
    ),
    ("hill", _words("hill", "peak", "mountain", "ridge")),
    ("station", _words("railway\\s+station")),
    ("park", _words("park", "garden", "botanical", "zoo")),
    (
        "building",
        _words("fort", "palace", "museum", "librar(?:y|ies)", "theatre", "building"),
    ),
    (
        "institution",
        _words("school", "college", "universit(?:y|ies)", "institute", "academy", "hospital"),
    ),
)

#: A category that files an article under a *year* is about something that
#: happened: "1969 in London", "1973 in cycling". The year has to be followed
#: by "in", so "Buildings and structures completed in 1917" -- which is how a
#: plain landmark is filed -- is not an event.
_EVENT_YEAR: Final[re.Pattern[str]] = re.compile(r"\b(?:1[6-9]\d{2}|20\d{2}) in \w")

#: Event categories whose wording does not also name the venue: a concert hall is
#: a building ("Concert halls in London"), a concert at one is not.
_EVENT_CATEGORY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:concerts?\s+at|festivals?|championships?|tournaments?|marathons?|"
    r"protests?|tributes?|pilgrimages?|parades?)\b",
    re.IGNORECASE,
)

#: A work of art, not a happening. Henry Moore's "The Arch 1979-1980" is filed
#: under "1980 in London" as well as "Outdoor sculptures in London", and the
#: year alone would demote a sculpture a walker can stand in front of.
_WORK_CATEGORY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:sculpture|statue|painting|artwork|memorial|monument|fountain|"
    r"building|structure|installation|mural)(?:s|es)?\b",
    re.IGNORECASE,
)

#: Kinds a walk wants, and the order it wants them in, lowest first. Distance
#: still decides *inside* a tier, so the nearest temple leads: this only stops a
#: school or a revenue division from taking a slot a landmark should have.
KIND_PRIORITY: Final[dict[str, int]] = {
    "water": 0,
    "religious": 0,
    "monument": 0,
    "hill": 0,
    "station": 0,
    "park": 0,
    "building": 0,
    "locality": 1,
    PLACE: 2,
    "institution": 3,
    "administrative": 3,
    "event": 3,
}


def classify(title: str, categories: Iterable[str] = ()) -> str:
    """The kind of thing a fact is, from its title and its Wikipedia categories.

    The three passes and the reason for their order are in the module docstring.
    """
    blob = " ".join(category for category in categories if category)

    if blob and (_EVENT_YEAR.search(blob) or _EVENT_CATEGORY.search(blob)):
        if not _WORK_CATEGORY.search(blob):
            return "event"

    for kind, pattern in KIND_PATTERNS:
        if pattern.search(title):
            return kind

    if blob:
        for kind, pattern in CATEGORY_PATTERNS:
            if pattern.search(blob):
                return kind

    return PLACE


def priority(kind: str | None) -> int:
    """Where a kind sorts. An unknown kind is a place, not a demotion."""
    return KIND_PRIORITY.get(kind or PLACE, KIND_PRIORITY[PLACE])
