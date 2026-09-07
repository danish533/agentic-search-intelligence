"""Report node.

Sole responsibility: assemble the final output (spec S3.2) - the structured JSON and the
human-readable summary spec S4.2 asks for. It draws no new conclusions; every claim it renders
came from the Analysis agent or from a measurement.

It has a deterministic path for when the LLM is unavailable. That is not belt-and-braces: this
node is also the terminus of the degradation route, so it may be reached precisely because a
provider is down. A report that required a working LLM to report an LLM failure would be
useless exactly when it is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sightline.agents.contracts.analysis import AnalysisResult
from sightline.agents.contracts.report import ReportDraft
from sightline.agents.dependencies import NodeDependencies
from sightline.agents.graph.names import NodeName
from sightline.agents.graph.state import PipelineState
from sightline.agents.prompts import report as prompts
from sightline.application.dto.pipeline import ReportDocument
from sightline.application.ports.llm_provider import LLMError
from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.entities.insight import Insight
from sightline.domain.entities.profile import Profile
from sightline.domain.entities.recommendation import Recommendation
from sightline.domain.value_objects.enums import DegradationReason, VisibilityStatus
from sightline.observability.logging import get_logger
from sightline.observability.metrics import current_metrics
from sightline.observability.tracing import node_span

_logger = get_logger(__name__)

_TOP_QUERIES_IN_REPORT = 10

_DEGRADATION_NOTES: dict[DegradationReason, str] = {
    DegradationReason.NO_PLAN_PRODUCED: (
        "the query planner produced no usable plan, so a template plan was used instead"
    ),
    DegradationReason.ALL_RETRIEVALS_FAILED: (
        "every retrieval call failed, so no live measurements were obtained"
    ),
    DegradationReason.INSUFFICIENT_RETRIEVALS: (
        "too few retrieval calls succeeded to support a full analysis"
    ),
    DegradationReason.NORMALIZATION_EMPTY: ("retrieved responses contained no usable results"),
    DegradationReason.ANALYSIS_UNAVAILABLE: (
        "the analysis stage was unavailable, so findings were not generated"
    ),
    DegradationReason.UPSTREAM_CIRCUIT_OPEN: (
        "the search data provider was unavailable and its circuit breaker was open"
    ),
}


def _query_summaries(queries: Sequence[DiscoveredQuery]) -> list[dict[str, Any]]:
    ordered = sorted(queries, key=lambda query: query.opportunity_score.value, reverse=True)
    return [
        {
            "query_text": query.query_text,
            "opportunity_score": query.opportunity_score.value,
            "estimated_search_volume": int(query.estimated_search_volume),
            "competitive_difficulty": int(query.competitive_difficulty),
            "domain_visible": query.domain_visible,
            "visibility_position": query.visibility_position,
            "visibility_status": query.visibility_status.value,
        }
        for query in ordered
    ]


def _coverage(queries: Sequence[DiscoveredQuery]) -> dict[str, int]:
    counts = dict.fromkeys((status.value for status in VisibilityStatus), 0)
    for query in queries:
        counts[query.visibility_status.value] += 1
    return counts


def build_deterministic_draft(
    *,
    profile: Profile,
    queries: Sequence[DiscoveredQuery],
    insights: Sequence[Insight],
    degradation_note: str | None,
) -> ReportDraft:
    """Assemble a report without an LLM, from measurements alone."""
    coverage = _coverage(queries)
    gaps = [query for query in queries if not query.domain_visible]
    top_gaps = sorted(gaps, key=lambda query: query.opportunity_score.value, reverse=True)[:3]

    headline = (
        f"{profile.name}: {len(gaps)} visibility gap(s) across {len(queries)} measured queries"
        if queries
        else f"{profile.name}: no queries could be measured"
    )

    lines = [
        f"{profile.name} ({profile.domain}) was measured across {len(queries)} queries. "
        f"It is visible for {coverage[VisibilityStatus.VISIBLE.value]}, absent from "
        f"{coverage[VisibilityStatus.NOT_VISIBLE.value]}, and "
        f"{coverage[VisibilityStatus.UNKNOWN.value]} could not be determined."
    ]
    if top_gaps:
        listed = ", ".join(f"'{query.query_text}'" for query in top_gaps)
        lines.append(f"The highest-opportunity gaps are {listed}.")
    if degradation_note:
        lines.append(f"Coverage was incomplete: {degradation_note}.")
    lines.append(
        "This summary was assembled without a language model, so it reports measurements "
        "directly rather than interpreting them."
    )

    return ReportDraft(
        headline=headline,
        executive_summary=" ".join(lines),
        key_findings=[insight.headline for insight in insights[:5]]
        or [
            f"'{query.query_text}' - opportunity {query.opportunity_score.value:.2f}, "
            f"{'visible' if query.domain_visible else 'not visible'}"
            for query in _sorted_top(queries)
        ],
        recommended_next_steps=[
            f"Create content targeting '{query.query_text}' "
            f"({int(query.estimated_search_volume)} monthly searches)"
            for query in top_gaps
        ],
    )


def _sorted_top(queries: Sequence[DiscoveredQuery]) -> list[DiscoveredQuery]:
    return sorted(queries, key=lambda query: query.opportunity_score.value, reverse=True)[:5]


def render_summary(draft: ReportDraft) -> str:
    """Render the draft as the human-readable markdown the API returns."""
    parts = [f"# {draft.headline}", "", draft.executive_summary]
    if draft.key_findings:
        parts += ["", "## Key findings", ""]
        parts += [f"- {finding}" for finding in draft.key_findings]
    if draft.recommended_next_steps:
        parts += ["", "## Recommended next steps", ""]
        parts += [f"{index}. {step}" for index, step in enumerate(draft.recommended_next_steps, 1)]
    return "\n".join(parts)


def build_structured_report(
    *,
    profile: Profile,
    question: str,
    draft: ReportDraft,
    analysis: AnalysisResult | None,
    queries: Sequence[DiscoveredQuery],
    insights: Sequence[Insight],
    recommendations: Sequence[Recommendation],
    degradation_reason: DegradationReason | None,
) -> dict[str, Any]:
    """The JSON half of the report. Every value is a primitive, so it round-trips through JSONB."""
    return {
        "headline": draft.headline,
        "question": question,
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": {
            "name": profile.name,
            "domain": profile.domain,
            "industry": profile.industry,
            "competitors": list(profile.competitors),
        },
        "coverage": {
            "queries_measured": len(queries),
            **_coverage(queries),
            "degraded": degradation_reason is not None,
            "degradation_reason": degradation_reason.value if degradation_reason else None,
        },
        "executive_summary": draft.executive_summary,
        "key_findings": list(draft.key_findings),
        "recommended_next_steps": list(draft.recommended_next_steps),
        "competitive_summary": analysis.competitive_summary if analysis else "",
        "top_opportunities": _query_summaries(queries)[:_TOP_QUERIES_IN_REPORT],
        "insights": [
            {
                "uuid": str(insight.uuid),
                "headline": insight.headline,
                "detail": insight.detail,
                "relevance_score": insight.relevance_score.value,
                "supporting_query_uuids": [str(uuid) for uuid in insight.supporting_query_uuids],
            }
            for insight in insights
        ],
        "recommendations": [
            {
                "uuid": str(recommendation.uuid),
                "target_query_uuid": str(recommendation.target_query_uuid),
                "content_type": recommendation.content_type.value,
                "title": recommendation.title,
                "rationale": recommendation.rationale,
                "target_keywords": list(recommendation.target_keywords),
                "priority": recommendation.priority.value,
            }
            for recommendation in recommendations
        ],
    }


async def report_node(state: PipelineState, *, deps: NodeDependencies) -> PipelineState:
    profile = state["profile"]
    question = state["question"]
    queries = state.get("queries", [])
    insights = state.get("insights", [])
    recommendations = state.get("recommendations", [])
    analysis = state.get("analysis")
    degradation_reason = state.get("degradation_reason")
    note = _DEGRADATION_NOTES.get(degradation_reason) if degradation_reason else None

    async with node_span(
        NodeName.REPORT,
        metrics=current_metrics(),
        inputs={
            "queries": len(queries),
            "insights": len(insights),
            "degraded": degradation_reason is not None,
        },
    ) as span:
        prompt_tokens = completion_tokens = 0
        rendered_by = "llm"

        try:
            completion = await deps.llm.complete_structured(
                system_prompt=prompts.SYSTEM_PROMPT,
                user_prompt=prompts.build_user_prompt(
                    profile, question, analysis, _query_summaries(queries), note
                ),
                schema=ReportDraft,
            )
            draft = completion.value
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
        except LLMError as exc:
            _logger.warning("report.llm_failed_using_deterministic", error=str(exc))
            rendered_by = "deterministic"
            draft = build_deterministic_draft(
                profile=profile, queries=queries, insights=insights, degradation_note=note
            )

        document = ReportDocument(
            headline=draft.headline,
            summary=render_summary(draft),
            structured=build_structured_report(
                profile=profile,
                question=question,
                draft=draft,
                analysis=analysis,
                queries=queries,
                insights=insights,
                recommendations=recommendations,
                degradation_reason=degradation_reason,
            ),
        )

        span.set_output(
            rendered_by=rendered_by,
            headline=draft.headline,
            summary_chars=len(document.summary),
            degraded=degradation_reason is not None,
        )

    return PipelineState(
        report=document,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
