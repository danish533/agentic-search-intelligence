"""Re-measure a single query (spec S4.2).

Re-runs the retrieval -> normalization -> analysis subgraph for one query and writes the fresh
measurements back onto the existing row. The previous values are carried out in the result so
the caller can see what changed - which is the point of a recheck after publishing content.
"""

from __future__ import annotations

from uuid import UUID

from sightline.application.dto.pipeline import RecheckRequest
from sightline.application.dto.results import RecheckResult
from sightline.application.ports.pipeline_engine import PipelineEngine
from sightline.application.ports.unit_of_work import UnitOfWorkFactory
from sightline.domain.value_objects.enums import RunStatus
from sightline.observability.correlation import current_correlation_id, new_correlation_id
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)


class RecheckQueryUseCase:
    """Re-measures one query and persists the update."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        engine: PipelineEngine,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._engine = engine

    async def execute(self, query_uuid: UUID) -> RecheckResult:
        """Raise :class:`DiscoveredQueryNotFoundError` when the query does not exist."""
        async with self._unit_of_work() as uow:
            query = await uow.queries.get(query_uuid)
            profile = await uow.profiles.get(query.profile_uuid)

        previous_score = query.opportunity_score.value
        previous_status = query.visibility_status.value
        previous_position = query.visibility_position
        previous_domains = set(query.cited_domains())

        outcome = await self._engine.recheck(
            RecheckRequest(
                profile=profile,
                query=query,
                run_uuid=query.run_uuid,
                correlation_id=current_correlation_id() or new_correlation_id(),
            )
        )

        # A degraded recheck leaves the stored measurement untouched. Overwriting a good
        # measurement with an unmeasurable one would destroy information on a transient
        # failure, which is the opposite of what a recheck is for.
        if outcome.status is RunStatus.COMPLETED:
            async with self._unit_of_work() as uow:
                await uow.queries.save(outcome.query)
                await uow.commit()
            _logger.info(
                "query.rechecked",
                query_uuid=str(query_uuid),
                previous_score=previous_score,
                new_score=outcome.query.opportunity_score.value,
                previous_status=previous_status,
                new_status=outcome.query.visibility_status.value,
            )
        else:
            _logger.warning(
                "query.recheck_degraded",
                query_uuid=str(query_uuid),
                status=outcome.status.value,
                reason=(outcome.degradation_reason.value if outcome.degradation_reason else None),
            )

        # Only meaningful when the recheck actually measured something; a degraded recheck
        # returns the original query unchanged, so the diff would be empty by construction.
        current_domains = set(outcome.query.cited_domains())
        gained = tuple(sorted(current_domains - previous_domains)) if current_domains else ()
        lost = tuple(sorted(previous_domains - current_domains)) if current_domains else ()

        return RecheckResult(
            query=outcome.query,
            previous_opportunity_score=previous_score,
            previous_visibility_status=previous_status,
            previous_visibility_position=previous_position,
            domains_gained=gained,
            domains_lost=lost,
            status=outcome.status,
            insights=outcome.insights,
            total_tokens=outcome.total_tokens,
            error_message=outcome.error_message,
        )
