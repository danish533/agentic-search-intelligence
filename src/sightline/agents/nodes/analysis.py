"""Analysis / Synthesis node.

Sole responsibility: reason over normalized data to produce insights (spec S3.2). It performs
no I/O and formats no output document.

Model-supplied query references are resolved against the queries that were actually measured.
A reference that does not resolve is **dropped with a warning**, never guessed at: attaching a
recommendation to the wrong query would be a plausible-looking, silently incorrect deliverable,
which is worse than one recommendation fewer.
"""

from __future__ import annotations

from uuid import UUID

from sightline.agents.contracts.analysis import AnalysisResult
from sightline.agents.dependencies import NodeDependencies
from sightline.agents.graph.names import NodeName
from sightline.agents.graph.state import PipelineState
from sightline.agents.prompts import analysis as prompts
from sightline.application.ports.llm_provider import LLMError
from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.entities.insight import Insight
from sightline.domain.entities.recommendation import Recommendation
from sightline.domain.value_objects.scores import RelevanceScore
from sightline.observability.logging import get_logger
from sightline.observability.metrics import current_metrics
from sightline.observability.tracing import node_span

_logger = get_logger(__name__)


def _index_by_text(queries: list[DiscoveredQuery]) -> dict[str, DiscoveredQuery]:
    """Index measured queries by normalised text, for resolving model references."""
    return {query.query_text.casefold().strip(): query for query in queries}


def _resolve(reference: str, index: dict[str, DiscoveredQuery]) -> DiscoveredQuery | None:
    return index.get(reference.casefold().strip())


async def analysis_node(state: PipelineState, *, deps: NodeDependencies) -> PipelineState:
    profile = state["profile"]
    question = state["question"]
    run_uuid: UUID = state["run_uuid"]
    normalized = state.get("normalized", [])
    queries = state.get("queries", [])

    async with node_span(
        NodeName.ANALYSIS,
        metrics=current_metrics(),
        inputs={"normalized_queries": len(normalized)},
    ) as span:
        try:
            completion = await deps.llm.complete_structured(
                system_prompt=prompts.SYSTEM_PROMPT,
                user_prompt=prompts.build_user_prompt(profile, question, normalized),
                schema=AnalysisResult,
            )
        except LLMError as exc:
            _logger.warning("analysis.llm_failed", error=str(exc), provider=deps.llm.name)
            span.set_output(insights=0, recommendations=0, failed=True)
            return PipelineState(analysis=None, errors=[f"analysis: {exc}"])

        result = completion.value
        index = _index_by_text(queries)

        insights = [
            Insight(
                run_uuid=run_uuid,
                headline=draft.headline,
                detail=draft.detail,
                relevance_score=RelevanceScore.clamped(draft.relevance_score),
                supporting_query_uuids=tuple(
                    resolved.uuid
                    for reference in draft.supporting_queries
                    if (resolved := _resolve(reference, index)) is not None
                ),
            )
            for draft in result.insights
        ]

        recommendations: list[Recommendation] = []
        unresolved = 0
        for draft in result.recommendations:
            target = _resolve(draft.target_query, index)
            if target is None:
                unresolved += 1
                _logger.warning(
                    "analysis.recommendation_dropped",
                    reason="target_query did not match any measured query",
                    target_query=draft.target_query,
                    title=draft.title,
                )
                continue
            recommendations.append(
                Recommendation(
                    run_uuid=run_uuid,
                    target_query_uuid=target.uuid,
                    content_type=draft.content_type,
                    title=draft.title,
                    rationale=draft.rationale,
                    target_keywords=tuple(draft.target_keywords),
                    priority=draft.priority,
                )
            )

        span.set_output(
            insights=len(insights),
            recommendations=len(recommendations),
            dropped_recommendations=unresolved,
        )

        return PipelineState(
            analysis=result,
            insights=insights,
            recommendations=recommendations,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        )
