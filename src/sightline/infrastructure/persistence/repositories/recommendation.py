"""SQLAlchemy implementation of ``RecommendationRepository``."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sightline.domain.entities.recommendation import Recommendation
from sightline.infrastructure.persistence.mappers import (
    recommendation_to_domain,
    recommendation_to_model,
)
from sightline.infrastructure.persistence.models.recommendation import RecommendationModel


class SqlAlchemyRecommendationRepository:
    """Recommendation persistence bound to one session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_many(self, recommendations: Sequence[Recommendation]) -> None:
        self._session.add_all(
            [
                recommendation_to_model(recommendation, position=index)
                for index, recommendation in enumerate(recommendations)
            ]
        )

    async def list_for_run(self, run_uuid: UUID) -> Sequence[Recommendation]:
        rows = await self._session.scalars(
            select(RecommendationModel)
            .where(RecommendationModel.run_uuid == run_uuid)
            # The native enum is declared high, medium, low - so PostgreSQL's own ordering
            # already yields high-priority first, with no CASE expression or sort column.
            .order_by(RecommendationModel.priority.asc(), RecommendationModel.position.asc())
        )
        return [recommendation_to_domain(model) for model in rows]
