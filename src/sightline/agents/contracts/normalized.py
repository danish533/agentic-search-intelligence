"""Extraction / Normalization output contract.

The normalizer's sole responsibility is turning raw provider payloads into this clean shape
(spec S3.2). It draws no conclusions: it records what was observed - was the domain present,
at what position, who else was cited, how big is the query - and stops there. Deciding what
any of that *means* is the Analysis agent's job.

These are Pydantic models rather than dataclasses because they are serialised twice: into the
Analysis agent's prompt, and into the ``evidence`` JSONB column (via :meth:`NormalizedQuery.
to_evidence`) that lets a recheck say what actually changed rather than only that a score moved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sightline.domain.value_objects.enums import RetrievalKind, VisibilityStatus


class NormalizedObservation(BaseModel):
    """One measurement, from one retrieval capability, for one sub-query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_text: str
    kind: RetrievalKind
    source_endpoint: str

    visibility_status: VisibilityStatus = Field(
        description="Whether the profile's own domain was found in this result set."
    )
    position: int | None = Field(
        default=None, description="1-indexed rank, when the domain was found and ranked."
    )
    cited_domains: tuple[str, ...] = Field(
        default=(),
        description="Every domain present in the result set, in rank order. The competitive "
        "picture the Analysis agent reasons over.",
    )
    competitor_positions: dict[str, int] = Field(
        default_factory=dict,
        description="Positions of the profile's declared competitors, where present.",
    )

    search_volume: int | None = None
    competition_index: int | None = Field(
        default=None, description="0-100 competition index, when the capability reports one."
    )

    from_mock: bool = False


class NormalizedQuery(BaseModel):
    """Every observation for a single sub-query, merged across capabilities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_text: str
    observations: tuple[NormalizedObservation, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.observations

    def best_visibility(self) -> tuple[VisibilityStatus, int | None]:
        """Collapse observations into one visibility verdict for the query.

        A domain found by *any* capability counts as visible, at its best position. Absence is
        only reported when at least one capability actually looked and did not find it -
        otherwise the answer is ``UNKNOWN``, because a failed retrieval is not evidence of
        absence.
        """
        best_position: int | None = None
        saw_result = False

        for observation in self.observations:
            if observation.visibility_status is VisibilityStatus.VISIBLE:
                saw_result = True
                if observation.position is not None and (
                    best_position is None or observation.position < best_position
                ):
                    best_position = observation.position
            elif observation.visibility_status is VisibilityStatus.NOT_VISIBLE:
                saw_result = True

        if not saw_result:
            return VisibilityStatus.UNKNOWN, None
        if best_position is not None:
            return VisibilityStatus.VISIBLE, best_position

        visible_without_rank = any(
            observation.visibility_status is VisibilityStatus.VISIBLE
            for observation in self.observations
        )
        # Cited by an AI answer but not ranked organically: still visible, position unknown.
        # Reported at position 1, since a citation is the top of that surface.
        return (
            (VisibilityStatus.VISIBLE, 1)
            if visible_without_rank
            else (VisibilityStatus.NOT_VISIBLE, None)
        )

    def search_volume(self) -> int | None:
        for observation in self.observations:
            if observation.search_volume is not None:
                return observation.search_volume
        return None

    def competition_index(self) -> int | None:
        for observation in self.observations:
            if observation.competition_index is not None:
                return observation.competition_index
        return None

    def to_evidence(self) -> dict[str, Any]:
        """Serialise the observations for storage on the query row.

        Kept as plain JSON-safe primitives so it round-trips through ``JSONB``. This is what
        makes a recheck answer the useful question - *which competitor appeared or dropped
        out* - instead of only reporting that a number moved.
        """
        return {
            "captured_at": datetime.now(UTC).isoformat(),
            "cited_domains": list(self.all_cited_domains()),
            "observations": [
                {
                    "kind": observation.kind.value,
                    "source_endpoint": observation.source_endpoint,
                    "visibility_status": observation.visibility_status.value,
                    "position": observation.position,
                    "cited_domains": list(observation.cited_domains),
                    "competitor_positions": dict(observation.competitor_positions),
                    "search_volume": observation.search_volume,
                    "competition_index": observation.competition_index,
                    "from_mock": observation.from_mock,
                }
                for observation in self.observations
            ],
        }

    def all_cited_domains(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for observation in self.observations:
            for domain in observation.cited_domains:
                seen.setdefault(domain, None)
        return tuple(seen)
