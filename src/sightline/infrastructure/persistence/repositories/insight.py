"""SQLAlchemy implementation of ``InsightRepository``."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sightline.domain.entities.insight import Insight
from sightline.infrastructure.persistence.mappers import insight_to_domain, insight_to_model
from sightline.infrastructure.persistence.models.insight import InsightModel


class SqlAlchemyInsightRepository:
    """Insight persistence bound to one session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_many(self, insights: Sequence[Insight]) -> None:
        # The enumeration index preserves the Analysis agent's own ordering, which is the
        # tiebreak when several insights share a relevance score.
        self._session.add_all(
            [insight_to_model(insight, position=index) for index, insight in enumerate(insights)]
        )

    async def list_for_run(self, run_uuid: UUID) -> Sequence[Insight]:
        rows = await self._session.scalars(
            select(InsightModel)
            .where(InsightModel.run_uuid == run_uuid)
            .order_by(InsightModel.relevance_score.desc(), InsightModel.position.asc())
        )
        return [insight_to_domain(model) for model in rows]
