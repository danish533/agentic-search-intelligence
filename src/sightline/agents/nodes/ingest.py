"""Ingest node.

Sole responsibility: normalise the run's inputs before any agent sees them.

It exists so that no downstream node has to ask "was a question supplied?". Spec S2 frames the
input as a natural-language question while S4.2 makes the entry point a bodyless profile
trigger; reconciling those two is one job, done once, here.
"""

from __future__ import annotations

from sightline.agents.dependencies import NodeDependencies
from sightline.agents.graph.names import NodeName
from sightline.agents.graph.state import PipelineState
from sightline.observability.metrics import current_metrics
from sightline.observability.tracing import node_span


async def ingest_node(state: PipelineState, *, deps: NodeDependencies) -> PipelineState:
    profile = state["profile"]
    question = (state.get("question") or "").strip()

    async with node_span(
        NodeName.INGEST,
        metrics=current_metrics(),
        inputs={"profile_domain": profile.domain, "question_supplied": bool(question)},
    ) as span:
        resolved = question or profile.default_research_question()
        span.set_output(question=resolved, competitors=len(profile.competitors))

    return PipelineState(question=resolved)
