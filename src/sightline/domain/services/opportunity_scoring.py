"""Opportunity scoring.

The spec (S4.2) requires an ``opportunity_score`` in 0-1 and leaves the formula to us. This
module is that formula, and it is deliberately a pure domain service: no I/O, no LLM, fully
deterministic, and unit-testable in isolation. The Analysis agent *uses* this score, it does
not invent one - keeping a business rule out of a prompt where it could silently drift.

Model
-----
An opportunity is attractive when the prize is large, the fight is winnable, and we are not
already winning it. Three independent components, combined as a weighted sum:

===============  ======  ==================================================================
Component        Weight  Meaning
===============  ======  ==================================================================
demand           0.35    How much traffic exists. Log-scaled, because search volume is
                         heavy-tailed: the gap between 100 and 1,000 searches matters far
                         more than the gap between 100,000 and 101,000.
attainability    0.25    ``1 - difficulty``. Discounts queries dominated by entrenched
                         competitors.
visibility_gap   0.40    How much headroom remains. Dominant weight, because this product
                         is about *closing visibility gaps* - a high-volume, easy query we
                         already rank first for is not an opportunity.
===============  ======  ==================================================================

The ``UNKNOWN`` visibility case scores a neutral 0.5 rather than 1.0. A retrieval failure
must not be laundered into a maximum-opportunity signal - that would rank the queries we know
least about at the top of the report.
"""

from __future__ import annotations

import math

from sightline.domain.errors import DomainValidationError
from sightline.domain.value_objects.enums import VisibilityStatus
from sightline.domain.value_objects.scores import (
    CompetitiveDifficulty,
    OpportunityScore,
    SearchVolume,
)

WEIGHT_DEMAND = 0.35
WEIGHT_ATTAINABILITY = 0.25
WEIGHT_VISIBILITY_GAP = 0.40

#: Volume at which the demand component saturates at 1.0.
VOLUME_SATURATION = 100_000

#: SERP position beyond which we treat visibility as effectively zero, so the full gap
#: applies. Position 11 is the first result off page one.
POSITION_HORIZON = 10

#: Gap credited when retrieval never established visibility either way.
UNKNOWN_VISIBILITY_GAP = 0.5

if not math.isclose(WEIGHT_DEMAND + WEIGHT_ATTAINABILITY + WEIGHT_VISIBILITY_GAP, 1.0):
    raise DomainValidationError(  # pragma: no cover - guards against a bad edit
        "Opportunity score weights must sum to 1.0, otherwise the score leaves its 0-1 range."
    )


def demand_component(search_volume: SearchVolume) -> float:
    """Log-scaled search volume in 0-1, saturating at :data:`VOLUME_SATURATION`."""
    volume = min(int(search_volume), VOLUME_SATURATION)
    return math.log10(1 + volume) / math.log10(1 + VOLUME_SATURATION)


def attainability_component(difficulty: CompetitiveDifficulty) -> float:
    """Inverse of competitive difficulty, in 0-1."""
    return 1.0 - difficulty.as_ratio


def visibility_gap_component(
    visibility_status: VisibilityStatus,
    visibility_position: int | None,
) -> float:
    """Remaining headroom in 0-1: 0.0 when already ranked first, 1.0 when absent."""
    match visibility_status:
        case VisibilityStatus.NOT_VISIBLE:
            return 1.0
        case VisibilityStatus.UNKNOWN:
            return UNKNOWN_VISIBILITY_GAP
        case VisibilityStatus.VISIBLE:
            # Guaranteed non-None by the DiscoveredQuery invariant; defended anyway because
            # this service is also called before an entity exists.
            position = visibility_position if visibility_position is not None else 1
            return min(1.0, max(0, position - 1) / POSITION_HORIZON)


def calculate_opportunity_score(
    *,
    search_volume: SearchVolume,
    competitive_difficulty: CompetitiveDifficulty,
    visibility_status: VisibilityStatus,
    visibility_position: int | None = None,
) -> OpportunityScore:
    """Combine demand, attainability and visibility gap into a single 0-1 score.

    Keyword-only by design: four numeric-ish arguments in a row are trivially transposable at
    a call site, and a silently transposed score is the kind of bug that never surfaces.
    """
    weighted = (
        WEIGHT_DEMAND * demand_component(search_volume)
        + WEIGHT_ATTAINABILITY * attainability_component(competitive_difficulty)
        + WEIGHT_VISIBILITY_GAP * visibility_gap_component(visibility_status, visibility_position)
    )
    # Clamped rather than constructed directly: floating-point accumulation can land a
    # hair outside [0, 1] even though the weights sum to exactly 1.0.
    return OpportunityScore.clamped(weighted)
