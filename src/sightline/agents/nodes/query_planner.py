"""Query Planner node.

Sole responsibility: decide which searches are needed (spec S3.2). It touches no API and
interprets no results - its entire output is a :class:`RetrievalPlan`.

It does not raise on failure. A planner that cannot produce a usable plan returns
``plan=None``, and the graph's conditional edge routes to the deterministic fallback. Raising
here would abort a run that is entirely recoverable.
"""

from __future__ import annotations

from sightline.agents.contracts.plan import RetrievalPlan
from sightline.agents.dependencies import NodeDependencies
from sightline.agents.graph.names import NodeName
from sightline.agents.graph.state import PipelineState
from sightline.agents.prompts import query_planner as prompts
from sightline.application.ports.llm_provider import LLMError
from sightline.observability.logging import get_logger
from sightline.observability.metrics import current_metrics
from sightline.observability.tracing import node_span

_logger = get_logger(__name__)


async def query_planner_node(state: PipelineState, *, deps: NodeDependencies) -> PipelineState:
    profile = state["profile"]
    question = state["question"]
    limit = deps.pipeline.max_planned_queries

    async with node_span(
        NodeName.QUERY_PLANNER,
        metrics=current_metrics(),
        inputs={"question": question, "max_planned_queries": limit},
    ) as span:
        try:
            completion = await deps.llm.complete_structured(
                system_prompt=prompts.SYSTEM_PROMPT,
                user_prompt=prompts.build_user_prompt(profile, question, limit),
                schema=RetrievalPlan,
            )
        except LLMError as exc:
            _logger.warning("planner.llm_failed", error=str(exc), provider=deps.llm.name)
            span.set_output(planned=0, failed=True)
            return PipelineState(plan=None, errors=[f"query_planner: {exc}"])

        plan = completion.value
        if len(plan.sub_queries) > limit:
            # Enforced here rather than trusted to the prompt: a model that over-produces
            # would otherwise multiply the run's API spend without any code change consenting.
            plan = plan.model_copy(update={"sub_queries": plan.sub_queries[:limit]})

        span.set_output(
            planned=len(plan.sub_queries),
            interpretation=plan.interpretation,
            queries=[sub.query_text for sub in plan.sub_queries],
        )

        return PipelineState(
            plan=plan if plan.is_usable else None,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        )
