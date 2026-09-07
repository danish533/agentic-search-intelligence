"""SQLAlchemy implementation of ``DiscoveredQueryRepository``."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from sightline.application.dto.pagination import Page, PageRequest
from sightline.application.dto.query_filters import QueryFilters
from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.errors import DiscoveredQueryNotFoundError
from sightline.infrastructure.persistence.mappers import (
    apply_query_changes,
    query_to_domain,
    query_to_model,
)
from sightline.infrastructure.persistence.models.discovered_query import DiscoveredQueryModel


class SqlAlchemyDiscoveredQueryRepository:
    """Discovered-query persistence, including the filtered listing of spec S4.2."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_many(self, queries: Sequence[DiscoveredQuery]) -> None:
        self._session.add_all([query_to_model(query) for query in queries])

    async def get(self, query_uuid: UUID) -> DiscoveredQuery:
        model = await self._session.get(DiscoveredQueryModel, query_uuid)
        if model is None:
            raise DiscoveredQueryNotFoundError(query_uuid)
        return query_to_domain(model)

    async def save(self, query: DiscoveredQuery) -> None:
        model = await self._session.get(DiscoveredQueryModel, query.uuid)
        if model is None:
            raise DiscoveredQueryNotFoundError(query.uuid)
        apply_query_changes(model, query)

    @staticmethod
    def _filtered(run_uuid: UUID, filters: QueryFilters) -> Select[tuple[DiscoveredQueryModel]]:
        statement = select(DiscoveredQueryModel).where(DiscoveredQueryModel.run_uuid == run_uuid)
        if filters.min_score is not None:
            statement = statement.where(DiscoveredQueryModel.opportunity_score >= filters.min_score)
        if filters.visibility_status is not None:
            statement = statement.where(
                DiscoveredQueryModel.visibility_status == filters.visibility_status
            )
        return statement

    async def list_for_run(
        self,
        run_uuid: UUID,
        *,
        filters: QueryFilters,
        page_request: PageRequest,
    ) -> Page[DiscoveredQuery]:
        statement = self._filtered(run_uuid, filters)

        total = await self._session.scalar(select(func.count()).select_from(statement.subquery()))

        rows = await self._session.scalars(
            statement.order_by(
                DiscoveredQueryModel.opportunity_score.desc(),
                # A deterministic tiebreak. Without it, rows with equal scores can be ordered
                # differently between two pages, so an item is shown twice and another is
                # never shown at all.
                DiscoveredQueryModel.uuid.asc(),
            )
            .offset(page_request.offset)
            .limit(page_request.limit)
        )

        return Page(
            items=[query_to_domain(model) for model in rows],
            total_items=int(total or 0),
            page=page_request.page,
            per_page=page_request.per_page,
        )

    async def average_opportunity_score(self, profile_uuid: UUID) -> float | None:
        average = await self._session.scalar(
            select(func.avg(DiscoveredQueryModel.opportunity_score)).where(
                DiscoveredQueryModel.profile_uuid == profile_uuid
            )
        )
        # None means "no queries yet", which is distinct from a genuine mean of 0.0 and must
        # not be flattened into one by a `or 0.0`.
        return float(average) if average is not None else None
