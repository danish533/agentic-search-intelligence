"""Bounded numeric value objects.

The assessment spec pins explicit ranges to three fields (``opportunity_score`` 0-1,
``competitive_difficulty`` 0-100, relevance scores 0-1). Encoding those ranges as value
objects means the invariant is enforced once, at construction, rather than re-checked
defensively at every use site - and it makes the scoring formula unit-testable in isolation.

Bounds are declared as ``ClassVar`` so they stay class constants: annotating them as plain
attributes would make them dataclass *fields*, putting them in ``__slots__`` and in equality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from sightline.domain.errors import DomainValidationError

_SCORE_PRECISION = 4


def _require_real_number(value: object, owner: str) -> float:
    """Reject non-numeric input, including ``bool`` (which is an ``int`` subclass)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainValidationError(f"{owner} requires a number, got {type(value).__name__}.")
    return float(value)


def _require_int(value: object, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DomainValidationError(f"{owner} requires an int, got {type(value).__name__}.")
    return value


@dataclass(frozen=True, slots=True, order=True)
class OpportunityScore:
    """How attractive a query is to target, on 0-1 (spec S4.2).

    Produced solely by :mod:`sightline.domain.services.opportunity_scoring`.
    """

    MINIMUM: ClassVar[float] = 0.0
    MAXIMUM: ClassVar[float] = 1.0

    value: float

    def __post_init__(self) -> None:
        numeric = _require_real_number(self.value, "OpportunityScore")
        if not self.MINIMUM <= numeric <= self.MAXIMUM:
            raise DomainValidationError(
                f"OpportunityScore must be within [{self.MINIMUM}, {self.MAXIMUM}], "
                f"got {self.value}."
            )
        object.__setattr__(self, "value", round(numeric, _SCORE_PRECISION))

    def __float__(self) -> float:
        return self.value

    @classmethod
    def clamped(cls, value: float) -> OpportunityScore:
        """Build from a value that may overshoot the range, clamping to the nearest bound.

        Correct where an upstream computation is mathematically allowed to exceed the bounds
        (weighted sums, LLM-supplied scores) and clamping - not raising - is the right answer.
        """
        numeric = _require_real_number(value, "OpportunityScore")
        return cls(min(max(numeric, cls.MINIMUM), cls.MAXIMUM))


@dataclass(frozen=True, slots=True, order=True)
class RelevanceScore:
    """How strongly an insight is supported by retrieved evidence, on 0-1 (spec S4.2)."""

    MINIMUM: ClassVar[float] = 0.0
    MAXIMUM: ClassVar[float] = 1.0

    value: float

    def __post_init__(self) -> None:
        numeric = _require_real_number(self.value, "RelevanceScore")
        if not self.MINIMUM <= numeric <= self.MAXIMUM:
            raise DomainValidationError(
                f"RelevanceScore must be within [{self.MINIMUM}, {self.MAXIMUM}], got {self.value}."
            )
        object.__setattr__(self, "value", round(numeric, _SCORE_PRECISION))

    def __float__(self) -> float:
        return self.value

    @classmethod
    def clamped(cls, value: float) -> RelevanceScore:
        numeric = _require_real_number(value, "RelevanceScore")
        return cls(min(max(numeric, cls.MINIMUM), cls.MAXIMUM))


@dataclass(frozen=True, slots=True, order=True)
class CompetitiveDifficulty:
    """How hard a query is to rank for, on 0-100 (spec S4.2)."""

    MINIMUM: ClassVar[int] = 0
    MAXIMUM: ClassVar[int] = 100

    value: int

    def __post_init__(self) -> None:
        numeric = _require_int(self.value, "CompetitiveDifficulty")
        if not self.MINIMUM <= numeric <= self.MAXIMUM:
            raise DomainValidationError(
                f"CompetitiveDifficulty must be within [{self.MINIMUM}, {self.MAXIMUM}], "
                f"got {self.value}."
            )

    def __int__(self) -> int:
        return self.value

    @property
    def as_ratio(self) -> float:
        """Difficulty normalised to 0-1, for use inside weighted formulas."""
        return self.value / self.MAXIMUM

    @classmethod
    def clamped(cls, value: float) -> CompetitiveDifficulty:
        numeric = _require_real_number(value, "CompetitiveDifficulty")
        return cls(int(min(max(round(numeric), cls.MINIMUM), cls.MAXIMUM)))


@dataclass(frozen=True, slots=True, order=True)
class SearchVolume:
    """Estimated monthly search volume. Non-negative, unbounded above."""

    MINIMUM: ClassVar[int] = 0

    value: int

    def __post_init__(self) -> None:
        numeric = _require_int(self.value, "SearchVolume")
        if numeric < self.MINIMUM:
            raise DomainValidationError(f"SearchVolume must be non-negative, got {self.value}.")

    def __int__(self) -> int:
        return self.value

    @classmethod
    def clamped(cls, value: float) -> SearchVolume:
        numeric = _require_real_number(value, "SearchVolume")
        return cls(max(cls.MINIMUM, round(numeric)))
