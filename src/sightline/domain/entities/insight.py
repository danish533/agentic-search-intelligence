"""The Insight entity: a conclusion drawn by the Analysis/Synthesis agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from sightline.domain.errors import DomainValidationError
from sightline.domain.value_objects.scores import RelevanceScore


@dataclass(frozen=True, slots=True)
class Insight:
    """A single finding, scored for relevance and traceable to the queries that support it.

    ``supporting_query_uuids`` is what keeps the Analysis agent honest: an insight must point
    at the normalized records it was drawn from, so a reviewer can audit a conclusion instead
    of taking the model's word for it.
    """

    run_uuid: UUID
    headline: str
    detail: str
    relevance_score: RelevanceScore
    supporting_query_uuids: tuple[UUID, ...] = ()
    uuid: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        headline = self.headline.strip()
        if not headline:
            raise DomainValidationError("Insight.headline must not be empty.")
        object.__setattr__(self, "headline", headline)
        object.__setattr__(self, "detail", self.detail.strip())
