"""The DiscoveredQuery entity: one sub-query the planner produced and retrieval measured.

This is the entity surfaced by ``GET /api/v1/profiles/{uuid}/queries`` and it carries the
exact field set the spec pins in S4.2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sightline.domain.errors import DomainValidationError
from sightline.domain.value_objects.enums import VisibilityStatus
from sightline.domain.value_objects.scores import (
    CompetitiveDifficulty,
    OpportunityScore,
    SearchVolume,
)

_MIN_SERP_POSITION = 1


@dataclass(frozen=True, slots=True)
class DiscoveredQuery:
    """A measured sub-query: its market size, its difficulty, and the brand's visibility in it."""

    profile_uuid: UUID
    run_uuid: UUID
    query_text: str
    estimated_search_volume: SearchVolume
    competitive_difficulty: CompetitiveDifficulty
    opportunity_score: OpportunityScore
    visibility_status: VisibilityStatus
    visibility_position: int | None = None

    #: The observations this measurement was derived from, as already-serialised JSON. Held on
    #: the entity for the same reason ``PipelineRun`` holds its report: the query owns the
    #: evidence for its own verdict, and a later recheck compares against it.
    evidence: Mapping[str, Any] | None = None

    uuid: UUID = field(default_factory=uuid4)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        query_text = self.query_text.strip()
        if not query_text:
            raise DomainValidationError("DiscoveredQuery.query_text must not be empty.")
        object.__setattr__(self, "query_text", query_text)

        # A position is meaningful only when the domain was actually found. Enforcing the
        # pairing here stops the API from ever reporting "not visible at position 3".
        if self.visibility_status is VisibilityStatus.VISIBLE:
            if self.visibility_position is None:
                raise DomainValidationError(
                    "A query marked 'visible' must carry a visibility_position."
                )
            if self.visibility_position < _MIN_SERP_POSITION:
                raise DomainValidationError(
                    f"visibility_position must be >= {_MIN_SERP_POSITION}, "
                    f"got {self.visibility_position}."
                )
        elif self.visibility_position is not None:
            raise DomainValidationError(
                f"A query marked '{self.visibility_status.value}' must not carry a "
                "visibility_position."
            )

    def cited_domains(self) -> tuple[str, ...]:
        """Domains observed in this query's result set, from the stored evidence."""
        if not self.evidence:
            return ()
        domains = self.evidence.get("cited_domains") or []
        return tuple(str(domain) for domain in domains)

    @property
    def domain_visible(self) -> bool:
        """The boolean the API contract exposes (spec S4.2).

        Derived rather than stored: ``UNKNOWN`` (retrieval failed) must report ``False``
        without being confused with a measured absence.
        """
        return self.visibility_status is VisibilityStatus.VISIBLE
