"""Graph state.

LangGraph merges each node's return value into this state. Channels annotated with a reducer
**accumulate** rather than overwrite, which is what makes the parallel retrieval fan-out
correct: several branches complete concurrently and each returns its own single-element list,
and the reducer concatenates them instead of the last writer winning.

Channels without a reducer are last-write-wins, which is right for the single-writer fields -
only the planner writes ``plan``, only the report node writes ``report``.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict
from uuid import UUID

from sightline.agents.contracts.analysis import AnalysisResult
from sightline.agents.contracts.normalized import NormalizedQuery
from sightline.agents.contracts.plan import PlannedRetrieval, RetrievalPlan
from sightline.agents.contracts.retrieval import RetrievalOutcome
from sightline.application.dto.pipeline import ReportDocument
from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.entities.insight import Insight
from sightline.domain.entities.profile import Profile
from sightline.domain.entities.recommendation import Recommendation
from sightline.domain.value_objects.enums import DegradationReason


class PipelineState(TypedDict, total=False):
    """State threaded through the full pipeline graph."""

    # --- Set once by the ingest node, read by everything downstream --------------
    profile: Profile
    question: str
    run_uuid: UUID
    correlation_id: str

    # --- Query Planner ------------------------------------------------------------
    plan: RetrievalPlan | None
    used_fallback_plan: bool

    # --- Retrieval (fan-out: accumulated across concurrent branches) ---------------
    retrieval_outcomes: Annotated[list[RetrievalOutcome], operator.add]

    # --- Extraction / Normalization ------------------------------------------------
    normalized: list[NormalizedQuery]
    queries: list[DiscoveredQuery]

    # --- Analysis / Synthesis --------------------------------------------------------
    analysis: AnalysisResult | None
    insights: list[Insight]
    recommendations: list[Recommendation]

    # --- Report ------------------------------------------------------------------
    report: ReportDocument | None

    # --- Cross-cutting -------------------------------------------------------------
    degradation_reason: DegradationReason | None
    # Accumulated, not overwritten: a fan-out branch that fails must not erase the record of
    # another branch that failed differently.
    errors: Annotated[list[str], operator.add]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]


class RetrievalBranchState(TypedDict):
    """The payload one fan-out branch receives.

    A retrieval branch needs its own sub-query and the brand context, and nothing else. Sending
    a narrow payload instead of the whole pipeline state keeps branches independent - a branch
    physically cannot read another branch's results, so it cannot accidentally depend on them.
    """

    profile: Profile
    sub_query: PlannedRetrieval
    run_uuid: UUID
    branch_index: int


def initial_state(
    *,
    profile: Profile,
    question: str,
    run_uuid: UUID,
    correlation_id: str,
) -> PipelineState:
    """Build the starting state.

    Reducer-backed channels are seeded explicitly: ``operator.add`` needs a left operand, and
    an absent channel would fail on the first branch that tries to accumulate into it.
    """
    return PipelineState(
        profile=profile,
        question=question,
        run_uuid=run_uuid,
        correlation_id=correlation_id,
        plan=None,
        used_fallback_plan=False,
        retrieval_outcomes=[],
        normalized=[],
        queries=[],
        analysis=None,
        insights=[],
        recommendations=[],
        report=None,
        degradation_reason=None,
        errors=[],
        prompt_tokens=0,
        completion_tokens=0,
    )
