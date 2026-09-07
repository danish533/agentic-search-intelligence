"""LangGraph implementation of the ``PipelineEngine`` port.

The boundary between the application ring and the DAG. It owns the two compiled graphs, the
per-run observability scope, and the translation from LangGraph's final state into the
application DTOs - so nothing outside this module needs to know that LangGraph is what runs
underneath.

**It does not raise for a failed run.** A run that could not complete comes back as a
``FAILED`` or ``PARTIAL`` outcome carrying a reason, because the API owes the caller a run
record rather than a stack trace (spec S3.5). The only thing that escapes is a programming
error, which should escape.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import Any

from sightline.agents.contracts.plan import PlannedRetrieval, RetrievalPlan
from sightline.agents.dependencies import NodeDependencies
from sightline.agents.graph.builder import build_pipeline_graph, build_recheck_graph
from sightline.agents.graph.state import PipelineState, initial_state
from sightline.agents.nodes.degradation import classify_degradation
from sightline.application.dto.pipeline import (
    PipelineOutcome,
    PipelineRequest,
    RecheckOutcome,
    RecheckRequest,
    ReportDocument,
)
from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.value_objects.enums import QueryIntent, RetrievalKind, RunStatus
from sightline.observability.correlation import correlation_scope
from sightline.observability.logging import get_logger
from sightline.observability.metrics import RunMetrics, metrics_scope

_logger = get_logger(__name__)

#: A recheck re-measures every surface, so the caller gets a like-for-like comparison against
#: the original measurement rather than a narrower one.
_RECHECK_KINDS = [
    RetrievalKind.ORGANIC_SERP,
    RetrievalKind.AI_OVERVIEW,
    RetrievalKind.LLM_VISIBILITY,
    RetrievalKind.KEYWORD_METRICS,
]


def _empty_report(headline: str, summary: str) -> ReportDocument:
    return ReportDocument(
        headline=headline,
        summary=summary,
        structured={"generated_at": datetime.now(UTC).isoformat(), "error": summary},
    )


class LangGraphPipelineEngine:
    """Executes the agent DAG and translates its result for the application ring."""

    def __init__(self, deps: NodeDependencies) -> None:
        self._deps = deps
        # Compiled once at startup. Compilation validates the topology, so a malformed graph
        # fails when the process boots rather than on the first request.
        self._pipeline = build_pipeline_graph(deps)
        self._recheck = build_recheck_graph(deps)

    async def run(self, request: PipelineRequest) -> PipelineOutcome:
        metrics = RunMetrics()

        with (
            correlation_scope(
                correlation_id=request.correlation_id, run_uuid=str(request.run_uuid)
            ),
            metrics_scope(metrics),
        ):
            _logger.info(
                "pipeline.started",
                profile_domain=request.profile.domain,
                question=request.question,
            )
            try:
                final: dict[str, Any] = await self._pipeline.ainvoke(
                    initial_state(
                        profile=request.profile,
                        question=request.question,
                        run_uuid=request.run_uuid,
                        correlation_id=request.correlation_id,
                    )
                )
            except Exception as exc:
                # The graph is built so that data-level failures route to degradation, so
                # reaching here means something structural broke. Report it as a failed run
                # rather than propagating: the caller still needs a run record.
                _logger.exception("pipeline.crashed", error_type=type(exc).__name__)
                return PipelineOutcome(
                    status=RunStatus.FAILED,
                    report=_empty_report(
                        "Pipeline failed", f"The pipeline could not complete: {exc}"
                    ),
                    metrics_summary=metrics.summary(),
                    total_tokens=metrics.total_tokens,
                    error_message=str(exc),
                )

            outcome = self._to_outcome(final, metrics)
            _logger.info(
                "pipeline.finished",
                status=outcome.status.value,
                queries=len(outcome.queries),
                insights=len(outcome.insights),
                total_tokens=outcome.total_tokens,
                degradation_reason=(
                    outcome.degradation_reason.value if outcome.degradation_reason else None
                ),
                metrics=metrics.summary(),
            )
            return outcome

    def _to_outcome(self, final: dict[str, Any], metrics: RunMetrics) -> PipelineOutcome:
        state: PipelineState = final  # type: ignore[assignment]

        outcomes = state.get("retrieval_outcomes", [])
        queries = state.get("queries", [])
        report = state.get("report")
        degradation_reason = state.get("degradation_reason")
        errors = state.get("errors", [])

        # Token totals come from the state channels, which the fan-out reducer summed across
        # every branch - the metrics collector only sees calls that pass through the DataForSEO
        # client, not LLM usage.
        metrics.record_tokens(
            prompt_tokens=state.get("prompt_tokens", 0),
            completion_tokens=state.get("completion_tokens", 0),
        )

        if report is None:
            status = RunStatus.FAILED
            report = _empty_report(
                "Pipeline produced no report",
                "The pipeline finished without producing a report document.",
            )
        elif degradation_reason is not None:
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.COMPLETED

        return PipelineOutcome(
            status=status,
            report=report,
            queries=tuple(queries),
            insights=tuple(state.get("insights", [])),
            recommendations=tuple(state.get("recommendations", [])),
            planned_retrieval_count=len(plan.sub_queries) if (plan := state.get("plan")) else 0,
            successful_retrieval_count=sum(1 for item in outcomes if item.succeeded),
            normalized_record_count=len(state.get("normalized", [])),
            total_tokens=metrics.total_tokens,
            metrics_summary=metrics.summary(),
            degradation_reason=degradation_reason,
            error_message="; ".join(errors[:5]) if errors else None,
        )

    async def recheck(self, request: RecheckRequest) -> RecheckOutcome:
        metrics = RunMetrics()
        original = request.query

        plan = RetrievalPlan(
            interpretation=f"Recheck of a single query: {original.query_text}",
            sub_queries=[
                PlannedRetrieval(
                    query_text=original.query_text,
                    intent=QueryIntent.COMMERCIAL,
                    rationale="Re-measuring a previously discovered query on request.",
                    retrieval_kinds=_RECHECK_KINDS,
                )
            ],
        )

        with (
            correlation_scope(
                correlation_id=request.correlation_id, run_uuid=str(request.run_uuid)
            ),
            metrics_scope(metrics),
        ):
            _logger.info("recheck.started", query=original.query_text)
            state = initial_state(
                profile=request.profile,
                question=f"Re-measure visibility for '{original.query_text}'.",
                run_uuid=request.run_uuid,
                correlation_id=request.correlation_id,
            )
            state["plan"] = plan

            try:
                final: dict[str, Any] = await self._recheck.ainvoke(state)
            except Exception as exc:
                _logger.exception("recheck.crashed", error_type=type(exc).__name__)
                return RecheckOutcome(
                    status=RunStatus.FAILED,
                    query=original,
                    metrics_summary=metrics.summary(),
                    error_message=str(exc),
                )

            result: PipelineState = final  # type: ignore[assignment]
            metrics.record_tokens(
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
            )

            measured = result.get("queries", [])
            if not measured:
                reason = classify_degradation(result)
                _logger.warning("recheck.degraded", reason=reason.value)
                return RecheckOutcome(
                    status=RunStatus.PARTIAL,
                    query=original,
                    total_tokens=metrics.total_tokens,
                    metrics_summary=metrics.summary(),
                    degradation_reason=reason,
                    error_message="; ".join(result.get("errors", [])[:3]) or None,
                )

            updated = self._merge_recheck(original, measured[0])
            _logger.info(
                "recheck.finished",
                query=updated.query_text,
                visibility=updated.visibility_status.value,
                opportunity_score=updated.opportunity_score.value,
            )
            return RecheckOutcome(
                status=RunStatus.COMPLETED,
                query=updated,
                insights=tuple(result.get("insights", [])),
                total_tokens=metrics.total_tokens,
                metrics_summary=metrics.summary(),
            )

    @staticmethod
    def _merge_recheck(original: DiscoveredQuery, measured: DiscoveredQuery) -> DiscoveredQuery:
        """Carry the fresh measurements onto the existing query's identity.

        The normalization node mints a new entity, which is right for a new run and wrong for
        a recheck: the caller asked to re-measure *this* query, and the API contract, the
        recommendations pointing at it, and its history all key on its existing uuid.
        """
        return dataclasses.replace(
            measured,
            uuid=original.uuid,
            profile_uuid=original.profile_uuid,
            run_uuid=original.run_uuid,
            discovered_at=datetime.now(UTC),
        )
