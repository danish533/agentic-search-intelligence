"""List content recommendations from a profile's most recent run (spec S4.2)."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sightline.application.ports.unit_of_work import UnitOfWorkFactory
from sightline.domain.entities.recommendation import Recommendation


class ListRecommendationsUseCase:
    """Recommendations for a profile's latest run, ordered high priority first."""

    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def execute(self, profile_uuid: UUID) -> Sequence[Recommendation]:
        async with self._unit_of_work() as uow:
            await uow.profiles.get(profile_uuid)
            latest = await uow.runs.find_latest_for_profile(profile_uuid)
            if latest is None:
                return []
            return await uow.recommendations.list_for_run(latest.uuid)
