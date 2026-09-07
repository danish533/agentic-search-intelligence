"""Retrieve a profile with its run statistics (spec S4.1)."""

from __future__ import annotations

from uuid import UUID

from sightline.application.dto.results import ProfileSummary
from sightline.application.ports.unit_of_work import UnitOfWorkFactory


class GetProfileUseCase:
    """Loads a profile plus the summary stats the endpoint reports."""

    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def execute(self, profile_uuid: UUID) -> ProfileSummary:
        """Raise :class:`ProfileNotFoundError` (HTTP 404) when the profile does not exist."""
        async with self._unit_of_work() as uow:
            profile = await uow.profiles.get(profile_uuid)
            total_runs = await uow.runs.count_for_profile(profile_uuid)
            latest = await uow.runs.find_latest_for_profile(profile_uuid)
            average = await uow.queries.average_opportunity_score(profile_uuid)

        return ProfileSummary(
            profile=profile,
            total_runs=total_runs,
            latest_run_uuid=latest.uuid if latest else None,
            latest_run_status=latest.status if latest else None,
            latest_run_at=latest.started_at.isoformat() if latest else None,
            average_opportunity_score=average,
        )
