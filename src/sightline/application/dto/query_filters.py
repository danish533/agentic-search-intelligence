"""Filter DTO for the discovered-queries listing (spec S4.2)."""

from __future__ import annotations

from dataclasses import dataclass

from sightline.domain.value_objects.enums import VisibilityStatus


@dataclass(frozen=True, slots=True)
class QueryFilters:
    """``?min_score=`` and ``?status=`` from the spec, as a single typed argument.

    Both are optional; ``None`` means "do not constrain on this dimension".
    """

    min_score: float | None = None
    visibility_status: VisibilityStatus | None = None

    @property
    def is_empty(self) -> bool:
        return self.min_score is None and self.visibility_status is None
