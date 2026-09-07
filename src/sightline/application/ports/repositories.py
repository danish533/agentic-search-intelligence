"""Repository ports.

Each protocol describes what the application needs of persistence, in domain vocabulary.
Concrete SQLAlchemy adapters live in the infrastructure ring and are injected by the
composition root - nothing in this ring knows Postgres exists.

Getters raise the corresponding ``EntityNotFoundError`` rather than returning ``None``: a
missing aggregate is an exceptional condition the API maps to 404. Finders that legitimately
express absence are named ``find_*`` and return ``None``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from sightline.application.dto.pagination import Page, PageRequest
from sightline.application.dto.query_filters import QueryFilters
from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.entities.insight import Insight
from sightline.domain.entities.pipeline_run import PipelineRun
from sightline.domain.entities.profile import Profile
from sightline.domain.entities.recommendation import Recommendation


class ProfileRepository(Protocol):
    """Persistence for the Profile aggregate."""

    async def add(self, profile: Profile) -> None: ...

    async def get(self, profile_uuid: UUID) -> Profile:
        """Raise :class:`ProfileNotFoundError` when absent."""
        ...

    async def find_by_domain(self, domain: str) -> Profile | None:
        """Return the profile registered for a normalized domain, if any."""
        ...


class PipelineRunRepository(Protocol):
    """Persistence for the PipelineRun aggregate."""

    async def add(self, run: PipelineRun) -> None: ...

    async def get(self, run_uuid: UUID) -> PipelineRun:
        """Raise :class:`PipelineRunNotFoundError` when absent."""
        ...

    async def save(self, run: PipelineRun) -> None:
        """Persist mutations to an existing run (status transitions, counters)."""
        ...

    async def find_latest_for_profile(self, profile_uuid: UUID) -> PipelineRun | None:
        """Most recent run by start time - the basis for the 'most recent run status' stat."""
        ...

    async def count_for_profile(self, profile_uuid: UUID) -> int: ...


class DiscoveredQueryRepository(Protocol):
    """Persistence for discovered queries, including the filtered listing of spec S4.2."""

    async def add_many(self, queries: Sequence[DiscoveredQuery]) -> None: ...

    async def get(self, query_uuid: UUID) -> DiscoveredQuery:
        """Raise :class:`DiscoveredQueryNotFoundError` when absent."""
        ...

    async def save(self, query: DiscoveredQuery) -> None:
        """Overwrite a query's measurements - used by the recheck endpoint."""
        ...

    async def list_for_run(
        self,
        run_uuid: UUID,
        *,
        filters: QueryFilters,
        page_request: PageRequest,
    ) -> Page[DiscoveredQuery]:
        """Filtered, paginated, ordered by opportunity score descending.

        Sorting is a repository concern, not a caller concern: the spec mandates the order,
        and doing it in SQL keeps it correct across pages.
        """
        ...

    async def average_opportunity_score(self, profile_uuid: UUID) -> float | None:
        """Mean opportunity score across every query ever discovered for a profile.

        ``None`` when the profile has no queries yet - distinct from a genuine mean of 0.0.
        """
        ...


class InsightRepository(Protocol):
    """Persistence for analysis insights."""

    async def add_many(self, insights: Sequence[Insight]) -> None: ...

    async def list_for_run(self, run_uuid: UUID) -> Sequence[Insight]:
        """Ordered by relevance score descending."""
        ...


class RecommendationRepository(Protocol):
    """Persistence for content recommendations."""

    async def add_many(self, recommendations: Sequence[Recommendation]) -> None: ...

    async def list_for_run(self, run_uuid: UUID) -> Sequence[Recommendation]:
        """Ordered by priority (high first), then insertion order."""
        ...
