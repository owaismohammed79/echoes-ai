"""Infrastructure / commercial deny-list.

AGENTS.md section 1 calls for a *negative ranking signal* rather than a hard
filter, and PRD section 3 lists the terms to push down. The two governing
documents disagree on the exact set:

* Part 1 spec: ``metro``, ``station``, ``constituency``, ``district``,
  ``wholesale``, ``cutting``
* AGENTS.md section 4: ``metro``, ``station``, ``constituency``, ``district``,
  ``assembly``

:data:`DENY_LIST` is deliberately the **union**. Over-filtering costs at most
one fact and falls back to the safe NOTICE path; under-filtering leaks
infrastructure or commercial results into narrated audio, which AGENTS.md
treats as a hard failure. See server/NOTES.md for the reconciliation.

One category is asked for by name, though: a **named railway station** is not a
junction but a place with a date and an architect, and AGENTS.md's own wording
-- push results down, do not drop *valid adjacent history* -- is the reason
:func:`is_allowed_title` can be asked to let those through. Metro, bus and tram
stops stay denied either way: they are where the walker is standing, not
something to say about it.
"""

from __future__ import annotations

import re
from typing import Final

DENY_LIST: Final[frozenset[str]] = frozenset(
    {
        "metro",
        "station",
        "constituency",
        "district",
        "assembly",
        "wholesale",
        "cutting",
    }
)

_DENY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(DENY_LIST)) + r")\b",
    re.IGNORECASE,
)

#: A station article that is about the railway itself, rather than a stop on a
#: city network. ``Mysuru Road Bus Station`` and ``Charing Cross tube station``
#: are not this.
_RAILWAY_STATION: Final[re.Pattern[str]] = re.compile(
    r"\brailway\s+station\b", re.IGNORECASE
)

#: Stops that stay denied even where railway stations are allowed.
_JUNCTION: Final[re.Pattern[str]] = re.compile(
    r"\b(?:metro|bus|tram|monorail|subway|tube|underground|ferry)\b",
    re.IGNORECASE,
)


def is_allowed_title(title: str, *, allow_railway_stations: bool = False) -> bool:
    """Return True if ``title`` survives the infrastructure deny-list.

    Matching is case-insensitive and anchored on word boundaries, so
    "Wharton Station" and "District Line" are rejected while "Bridgetown"
    and "Cutting Green Park" are not falsely matched on interior substrings.

    ``allow_railway_stations`` is the one documented exception (see the module
    docstring): a title that names a railway station is a candidate fact, so
    long as it is not also a metro, bus, tram or ferry stop.
    """
    if allow_railway_stations and _RAILWAY_STATION.search(title):
        return _JUNCTION.search(title) is None
    return _DENY_PATTERN.search(title) is None


__all__ = ["DENY_LIST", "is_allowed_title"]
