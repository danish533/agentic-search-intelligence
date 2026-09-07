"""Trigger the full DAG for a profile (spec S4.2).

Owns the run's transaction boundaries, in two deliberate stages:

1. The run record is committed as ``RUNNING`` **before** the DAG starts. If the process dies
   mid-run, there is still a row saying a run began and never finished - which is the record
   an operator needs. Writing everything at the end would leave no trace of a crashed run.
2. Results are written in one further transaction, so a run's record, queries, insights and
   recommendations all land together or not at all.
"""

from __future__ import annotations

from uuid import UUID

from sightline.application.dto.pipeline import PipelineRequest
from sightline.application.dto.results import RunResult
from sightline.application.ports.pipeline_engine import PipelineEngine
from sightline.application.ports.unit_of_work import UnitOfWorkFactory
from sightline.domain.entities.pipeline_run import PipelineRun
from sightline.domain.value_objects.enums import DegradationReason, RunStatus
from sightline.observability.correlation import current_correlation_id, new_correlation_id
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)


class RunPipelineUseCase:
    """Executes the DAG for a profile and persists everything it produced."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        engine: PipelineEngine,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._engine = engine

    async def execute(self, profile_uuid: UUID, question: str | None = None) -> RunResult:
        async with self._unit_of_work() as uow:
            profile = await uow.profiles.get(profile_uuid)

        resolved_question = (question or "").strip() or profile.default_research_question()
        run = PipelineRun(
            profile_uuid=profile.uuid,
            # Reuses the request's correlation id, so the HTTP request and every node of the
            # run it triggered share one identifier in the logs.
            correlation_id=current_correlation_id() or new_correlation_id(),
            question=resolved_question,
        )

        async with self._unit_of_work() as uow:
            await uow.runs.add(run)
            await uow.commit()

        outcome = await self._engine.run(
            PipelineRequest(
                profile=profile,
                question=resolved_question,
                run_uuid=run.uuid,
                correlation_id=run.correlation_id,
            )
        )

        async with self._unit_of_work() as uow:
            persisted = await uow.runs.get(run.uuid)
            persisted.record_plan(outcome.planned_retrieval_count)
            persisted.record_retrieval_outcome(outcome.successful_retrieval_count)
            persisted.record_normalized(outcome.normalized_record_count)
            persisted.add_tokens(outcome.total_tokens)
            persisted.attach_output(
                report=dict(outcome.report.structured), metrics=dict(outcome.metrics_summary)
            )

            match outcome.status:
                case RunStatus.COMPLETED:
                    persisted.mark_completed()
                case RunStatus.PARTIAL:
                    # The engine always sets a reason alongside PARTIAL; the default keeps the
                    # transition total rather than relying on that invariant holding forever.
                    persisted.mark_partial(
                        outcome.degradation_reason or DegradationReason.INSUFFICIENT_RETRIEVALS,
                        outcome.error_message,
                    )
                case _:
                    persisted.mark_failed(outcome.error_message or "pipeline failed")

            await uow.runs.save(persisted)
            if outcome.queries:
                await uow.queries.add_many(outcome.queries)
            if outcome.insights:
                await uow.insights.add_many(outcome.insights)
            if outcome.recommendations:
                await uow.recommendations.add_many(outcome.recommendations)
            await uow.commit()

        _logger.info(
            "run.persisted",
            run_uuid=str(persisted.uuid),
            status=persisted.status.value,
            queries=len(outcome.queries),
            recommendations=len(outcome.recommendations),
        )
        return RunResult(run=persisted, outcome=outcome)
