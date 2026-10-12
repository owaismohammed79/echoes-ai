"""Tests for fact typing and walk ranking.

Run from ``server/``:  python -m pytest -q

The point of these tests is the ordering, not the label: what a walk narrates is
decided by which facts reach the prompt, and the kinds are what decide that.
"""

from __future__ import annotations

import pytest

from app.services.fact_kind import KIND_PRIORITY, PLACE, classify, priority

# ---------------------------------------------------------------------------
# Typing from a title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "kind"),
    [
        ("Sankey Tank", "water"),
        ("Ulsoor Lake", "water"),
        ("Vrishabhavathi River", "water"),
        ("Gavi Gangadhareshwara Temple", "religious"),
        ("St. Mary's Basilica, Bengaluru", "religious"),
        ("Jama Masjid", "religious"),
        ("Nelson's Column", "monument"),
        ("India Gate", "monument"),
        ("Daly Memorial Hall", "monument"),
        ("Chamundi Hill", "hill"),
        ("KSR Bengaluru City railway station", "station"),
        ("Cubbon Park", "park"),
        ("Bangalore Fort", "building"),
        ("Vidhana Soudha building", "building"),
        ("Vijayanagar, Bengaluru", "place"),  # typed from its categories instead
        ("Venkat International Public School", "institution"),
        ("Bengaluru division", "administrative"),
        ("Rajajinagar Assembly constituency", "administrative"),
    ],
)
def test_classify_reads_the_title(title: str, kind: str) -> None:
    assert classify(title) == kind


def test_classify_falls_back_to_the_categories() -> None:
    """``Lal Bagh`` says nothing about itself; its categories do."""
    assert classify("Lal Bagh", ["Botanical gardens in India", "Lal Bagh"]) == "park"
    assert classify("Chamarajpet", ["Neighbourhoods in Bengaluru"]) == "locality"


def test_categories_never_override_a_precise_title() -> None:
    """A Catholic school is a school: the title decides before categories do."""
    kind = classify(
        "St Joseph's Boys' High School, Bengaluru",
        ["Catholic schools in India", "Christianity in Bengaluru"],
    )
    assert kind == "institution"


def test_a_settlement_category_beats_an_incidental_landmark_category() -> None:
    """Rajajinagar is a neighbourhood named after a man, not a memorial to one."""
    kind = classify(
        "Rajajinagar",
        ["Memorials to C. Rajagopalachari", "Neighbourhoods in West Bengaluru"],
    )
    assert kind == "locality"


def test_containment_words_in_a_category_do_not_type_the_subject() -> None:
    """"Cities and towns in Tumakuru district" says where, not what."""
    assert classify("Tumkur", ["Cities and towns in Tumakuru district"]) == "locality"
    assert classify("Mudigere, Tumkur", ["Villages in Tumakuru district"]) == "locality"


def test_an_event_is_not_a_place() -> None:
    """A concert named after a park, and a championship, are not landmarks.

    Both have place-like titles, which is why the event check reads categories
    before the title pass does.
    """
    assert classify("The Stones in the Park", ["1969 in British music"]) == "event"
    assert (
        classify(
            "1973 UCI Cyclo-cross World Championships", ["1973 in cycle racing"]
        )
        == "event"
    )
    assert classify("Royal Parks Half Marathon", ["Parks in London"]) == "event"
    assert priority("event") == priority("administrative")


def test_a_landmark_filed_under_a_year_is_still_a_landmark() -> None:
    """"Completed in 1917" is how a monument is filed, not an event."""
    kind = classify(
        "Daly Memorial Hall",
        ["Buildings and structures completed in 1917", "Libraries in Bengaluru"],
    )
    assert kind == "monument"


def test_a_venue_is_not_an_event() -> None:
    """Hyde Park hosted the Olympics; it is not the Olympics."""
    kind = classify(
        "Hyde Park, London",
        [
            "Olympic swimming venues",
            "Great Exhibition",
            "Venues of the 2012 Summer Olympics",
        ],
    )
    assert kind == "park"


def test_a_containment_phrase_does_not_type_the_subject() -> None:
    """A category listing where a subject sits is not evidence of what it is."""
    kind = classify(
        "The Serpentine",
        [
            "Lakes of London",
            "Parks and open spaces in the City of Westminster",
            "Hyde Park, London",
        ],
    )
    assert kind == "water"


def test_a_year_category_does_not_demote_a_work() -> None:
    """Henry Moore's "The Arch 1979-1980" is a sculpture, not a happening.

    It is filed under "1980 in London" and under "Outdoor sculptures in
    London": the year alone would push a landmark a walker can stand in front
    of below the revenue boundaries.
    """
    kind = classify(
        "The Arch 1979-1980",
        ["1980 in London", "Outdoor sculptures in London", "Sculptures by Henry Moore"],
    )
    assert kind == "monument"


def test_the_most_specific_kind_wins() -> None:
    assert classify("War Memorial Library") == "monument"
    assert classify("Karnataka Legislative Assembly") == "administrative"


def test_every_kind_has_a_priority() -> None:
    """A new kind without a priority would silently sort as a place."""
    from app.services.fact_kind import CATEGORY_PATTERNS, KIND_PATTERNS

    for kind, _ in (*KIND_PATTERNS, *CATEGORY_PATTERNS):
        assert kind in KIND_PRIORITY


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_landmarks_outrank_institutions_and_boundaries() -> None:
    assert priority("water") < priority("locality")
    assert priority("religious") < priority("institution")
    assert priority("administrative") == priority("institution")
    assert priority(PLACE) < priority("administrative")


def test_an_unknown_kind_is_not_a_demotion() -> None:
    assert priority(None) == priority(PLACE)
    assert priority("something-new") == priority(PLACE)
