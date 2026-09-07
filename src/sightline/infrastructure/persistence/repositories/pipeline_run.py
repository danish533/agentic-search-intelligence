"""SQLAlchemy implementation of ``PipelineRunRepository``."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from sightline.domain.entities.pipeline_run import PipelineRun
from sightline.domain.errors import PipelineRunNotFoundError
from sightline.infrastructure.persistence.mappers import (
    apply_run_changes,
    run_to_domain,
    run_to_model,
)
from sightline.infrastructure.persistence.models.pipeline_run import PipelineRunModel


class SqlAlchemyPipelineRunRepository:
    """Pipeline-run persistence bound to one session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, run: PipelineRun) -> None:
        self._session.add(run_to_model(run))

    async def get(self, run_uuid: UUID) -> PipelineRun:
        model = await self._session.get(PipelineRunModel, run_uuid)
        if model is None:
            raise PipelineRunNotFoundError(run_uuid)
        return run_to_domain(model)

    async def save(self, run: PipelineRun) -> None:
        """Persist mutations to an existing run.

        Loads the attached instance and copies onto it, so the change participates in the
        session's flush rather than issuing a separate merge round trip.
        """
        model = await self._session.get(PipelineRunModel, run.uuid)
        if model is None:
            raise PipelineRunNotFoundError(run.uuid)
        apply_run_changes(model, run)

    async def find_latest_for_profile(self, profile_uuid: UUID) -> PipelineRun | None:
        result = await self._session.scalars(
            select(PipelineRunModel)
            .where(PipelineRunModel.profile_uuid == profile_uuid)
            .order_by(PipelineRunModel.started_at.desc(), PipelineRunModel.uuid.desc())
            .limit(1)
        )
        model = result.first()
        return run_to_domain(model) if model is not None else None

    async def count_for_profile(self, profile_uuid: UUID) -> int:
        total = await self._session.scalar(
            select(func.count())
            .select_from(PipelineRunModel)
            .where(PipelineRunModel.profile_uuid == profile_uuid)
        )
        return int(total or 0)
