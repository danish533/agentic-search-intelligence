"""Degradation node.

Sole responsibility: classify *why* a run could not complete normally, so the Report agent can
say so and the API can answer ``status: "partial"`` (spec S3.5, "return partial results with a
clear error flag").

It deliberately does not build the report. Assembling output is the Report agent's job, and
duplicating it here would create a second document format that drifts from the first. This
node records a reason; the report renders it.

It contains no LLM call and no I/O, so it is reachable no matter which dependency failed.
"""

from __future__ import annotations

from sightline.agents.dependencies import NodeDependencies
from sightline.agents.graph.names import NodeName
from sightline.agents.graph.state import PipelineState
from sightline.application.ports.errors import CircuitOpenError
from sightline.domain.value_objects.enums import DegradationReason
from sightline.observability.logging import get_logger
from sightline.observability.metrics import current_metrics
from sightline.observability.tracing import node_span

_logger = get_logger(__name__)


def classify_degradation(state: PipelineState) -> DegradationReason:
    """Determine the most specific reason this run degraded.

    Ordered most-specific first: an open circuit explains a total retrieval failure better
    than "everything failed" does, and the on-call engineer reading the run record needs the
    former.
    """
    outcomes = state.get("retrieval_outcomes", [])
    plan = state.get("plan")

    if plan is None or not plan.sub_queries:
        return DegradationReason.NO_PLAN_PRODUCED

    if any(outcome.error_type == CircuitOpenError.__name__ for outcome in outcomes):
        return DegradationReason.UPSTREAM_CIRCUIT_OPEN

    if outcomes and not any(outcome.succeeded for outcome in outcomes):
        return DegradationReason.ALL_RETRIEVALS_FAILED

    if not state.get("normalized"):
        return DegradationReason.NORMALIZATION_EMPTY

    if state.get("analysis") is None:
        return DegradationReason.ANALYSIS_UNAVAILABLE

    return DegradationReason.INSUFFICIENT_RETRIEVALS


async def degradation_node(state: PipelineState, *, deps: NodeDependencies) -> PipelineState:
    outcomes = state.get("retrieval_outcomes", [])
    succeeded = sum(1 for outcome in outcomes if outcome.succeeded)

    async with node_span(
        NodeName.DEGRADATION,
        metrics=current_metrics(),
        inputs={"branches": len(outcomes), "succeeded": succeeded},
    ) as span:
        reason = classify_degradation(state)
        _logger.warning(
            "pipeline.degraded",
            reason=reason.value,
            branches=len(outcomes),
            succeeded=succeeded,
            errors=state.get("errors", [])[:5],
        )
        span.set_output(degradation_reason=reason.value)

    return PipelineState(degradation_reason=reason)
