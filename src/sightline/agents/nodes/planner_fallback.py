"""Planner fallback node.

The fallback path spec S3.1 asks for: "route to a fallback path if a retrieval node fails
validation". Reached when the Query Planner could not produce a usable plan - the provider was
down, the key was wrong, or the model returned something that would not satisfy the schema.

Deliberately contains **no LLM call**. A fallback that depends on the component that just
failed is not a fallback. This builds a plan from the profile alone, using templates that hold
for any brand in any industry, so the run degrades to a narrower but still genuinely useful
answer instead of returning nothing.
"""

from __future__ import annotations

from sightline.agents.contracts.plan import PlannedRetrieval, RetrievalPlan
from sightline.agents.dependencies import NodeDependencies
from sightline.agents.graph.names import NodeName
from sightline.agents.graph.state import PipelineState
from sightline.domain.entities.profile import Profile
from sightline.domain.value_objects.enums import QueryIntent, RetrievalKind
from sightline.observability.logging import get_logger
from sightline.observability.metrics import current_metrics
from sightline.observability.tracing import node_span

_logger = get_logger(__name__)

_FULL_COVERAGE = [
    RetrievalKind.ORGANIC_SERP,
    RetrievalKind.AI_OVERVIEW,
    RetrievalKind.LLM_VISIBILITY,
    RetrievalKind.KEYWORD_METRICS,
]


#: Why this plan was produced. The template itself is neutral - it is used both when the LLM
#: planner has failed and, offline, when there is no LLM at all - so the caller states the
#: reason rather than the builder asserting one that may not be true.
FALLBACK_REASON = "Generated without an LLM because the Query Planner was unavailable."


def build_fallback_plan(
    profile: Profile, *, limit: int, reason: str = FALLBACK_REASON
) -> RetrievalPlan:
    """Derive a plan from the profile alone, deterministically.

    The template set mirrors what the LLM planner is asked for - brand defence, category
    demand, comparison intent - so the fallback answers the same question, just less
    imaginatively.
    """
    category = profile.industry.strip() or "software"
    brand = profile.name

    candidates: list[PlannedRetrieval] = [
        PlannedRetrieval(
            query_text=f"best {category}",
            intent=QueryIntent.COMMERCIAL,
            rationale="Category demand: the query most likely to decide a purchase.",
            retrieval_kinds=_FULL_COVERAGE,
        ),
        PlannedRetrieval(
            query_text=f"{brand} alternatives",
            intent=QueryIntent.COMPARISON,
            rationale="Comparison intent: where competitors capture existing brand demand.",
            retrieval_kinds=_FULL_COVERAGE,
        ),
        PlannedRetrieval(
            query_text=f"{brand} review",
            intent=QueryIntent.COMMERCIAL,
            rationale="Brand evaluation: what buyers find when researching the brand itself.",
            retrieval_kinds=_FULL_COVERAGE,
        ),
        PlannedRetrieval(
            query_text=f"{category} for small business",
            intent=QueryIntent.COMMERCIAL,
            rationale="Segment demand: a common qualifier on category searches.",
            retrieval_kinds=[
                RetrievalKind.ORGANIC_SERP,
                RetrievalKind.AI_OVERVIEW,
                RetrievalKind.KEYWORD_METRICS,
            ],
        ),
        PlannedRetrieval(
            query_text=brand,
            intent=QueryIntent.NAVIGATIONAL,
            rationale="Brand defence: confirms the brand owns its own name.",
            retrieval_kinds=[RetrievalKind.ORGANIC_SERP, RetrievalKind.KEYWORD_METRICS],
        ),
    ]

    for competitor in profile.competitors[:2]:
        competitor_name = competitor.split(".")[0]
        candidates.append(
            PlannedRetrieval(
                query_text=f"{brand} vs {competitor_name}",
                intent=QueryIntent.COMPARISON,
                rationale=f"Head-to-head comparison against declared competitor {competitor}.",
                retrieval_kinds=_FULL_COVERAGE,
            )
        )

    return RetrievalPlan(
        interpretation=(
            f"Measure {brand}'s search and AI-answer visibility across its core {category} "
            f"queries. {reason}"
        ),
        sub_queries=candidates[:limit],
    )


async def planner_fallback_node(state: PipelineState, *, deps: NodeDependencies) -> PipelineState:
    profile = state["profile"]

    async with node_span(
        NodeName.PLANNER_FALLBACK,
        metrics=current_metrics(),
        inputs={"profile_domain": profile.domain},
    ) as span:
        plan = build_fallback_plan(profile, limit=deps.pipeline.max_planned_queries)
        _logger.warning(
            "planner.fallback_engaged",
            planned=len(plan.sub_queries),
            reason="query_planner produced no usable plan",
        )
        span.set_output(
            planned=len(plan.sub_queries),
            queries=[sub.query_text for sub in plan.sub_queries],
        )

    return PipelineState(plan=plan, used_fallback_plan=True)
