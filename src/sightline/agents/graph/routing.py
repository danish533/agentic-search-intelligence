"""Conditional routing.

Spec S3.1 requires "conditional routing where appropriate (e.g. route to a fallback path if a
retrieval node fails validation)", and spec S6 grades the graph on being **non-linear**. These
functions are where that non-linearity lives.

Routing is deliberately *outside* the nodes. A node that decided its own successor would be
doing two jobs, and the graph's shape would only be discoverable by reading every node body.
Keeping the decisions here means the topology is declared in one place and can be rendered as
a diagram straight from the compiled graph.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.types import Send

from sightline.agents.contracts.plan import RetrievalPlan
from sightline.agents.graph.names import NodeName
from sightline.agents.graph.state import PipelineState, RetrievalBranchState
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)


def _fan_out(state: PipelineState, plan: RetrievalPlan) -> list[Send]:
    """Dispatch one concurrent retrieval branch per planned sub-query.

    ``Send`` is what makes retrieval genuinely parallel rather than a loop inside one node:
    LangGraph runs each branch as its own unit of work, and the state's reducer channel
    accumulates their results as they land.

    The plan is passed in rather than re-read from state so that the caller's ``is_usable``
    check is what narrows the type - no assertion, which would vanish under ``python -O``.
    """
    profile = state["profile"]
    run_uuid = state["run_uuid"]

    return [
        Send(
            NodeName.RETRIEVAL,
            RetrievalBranchState(
                profile=profile,
                sub_query=sub_query,
                run_uuid=run_uuid,
                branch_index=index,
            ),
        )
        for index, sub_query in enumerate(plan.sub_queries)
    ]


def route_after_planner(state: PipelineState) -> str | list[Send]:
    """Fan out into retrieval, or divert to the deterministic fallback planner."""
    plan = state.get("plan")
    if plan is None or not plan.is_usable:
        _logger.warning("routing.planner_to_fallback", reason="no usable plan")
        return NodeName.PLANNER_FALLBACK
    return _fan_out(state, plan)


def route_after_fallback(state: PipelineState) -> str | list[Send]:
    """Fan out from the fallback plan, or degrade if even that produced nothing."""
    plan = state.get("plan")
    if plan is None or not plan.is_usable:
        _logger.error("routing.fallback_empty", reason="fallback produced no sub-queries")
        return NodeName.DEGRADATION
    return _fan_out(state, plan)


def make_route_after_retrieval(min_successful: int) -> Callable[[PipelineState], str]:
    """Build the post-retrieval router, bound to the configured success threshold.

    A closure rather than a settings lookup inside the function: the threshold is fixed for
    the life of the graph, and binding it at build time keeps the router a pure function of
    state, which is what makes it testable without any configuration at all.
    """

    def route_after_retrieval(state: PipelineState) -> str:
        outcomes = state.get("retrieval_outcomes", [])
        succeeded = sum(1 for outcome in outcomes if outcome.succeeded)

        if succeeded < min_successful:
            _logger.warning(
                "routing.retrieval_to_degradation",
                succeeded=succeeded,
                required=min_successful,
                branches=len(outcomes),
            )
            return NodeName.DEGRADATION
        return NodeName.NORMALIZATION

    return route_after_retrieval


def route_after_normalization(state: PipelineState) -> str:
    """Proceed to analysis only if normalization produced something to reason about."""
    if not state.get("queries"):
        _logger.warning("routing.normalization_to_degradation", reason="no usable queries")
        return NodeName.DEGRADATION
    return NodeName.ANALYSIS


def route_after_analysis(state: PipelineState) -> str:
    """Proceed to the report, or degrade if the analysis stage produced nothing."""
    if state.get("analysis") is None:
        _logger.warning("routing.analysis_to_degradation", reason="analysis unavailable")
        return NodeName.DEGRADATION
    return NodeName.REPORT
