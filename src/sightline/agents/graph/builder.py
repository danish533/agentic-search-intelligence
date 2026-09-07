"""Graph construction.

Two graphs, sharing one set of node implementations:

``build_pipeline_graph``
    The full DAG. Spec S4.2's required sequence - planner, retrieval, normalization, analysis,
    report - plus the conditional routing and fallback paths spec S3.1 asks for.

``build_recheck_graph``
    The retrieval -> normalization -> analysis slice, for ``POST /queries/{uuid}/recheck``.
    It reuses the very same node functions, which is what makes recheck a genuine re-entry
    into the pipeline rather than a second implementation that drifts from the first. Building
    for this from the start is why the nodes take their dependencies by injection and read
    only the state keys they need.

Dependencies are bound at build time with ``functools.partial``. Nodes therefore remain plain
async functions of state, callable directly in a unit test with no graph and no runtime.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sightline.agents.dependencies import NodeDependencies
from sightline.agents.graph.names import NodeName
from sightline.agents.graph.routing import (
    make_route_after_retrieval,
    route_after_analysis,
    route_after_fallback,
    route_after_normalization,
    route_after_planner,
)
from sightline.agents.graph.state import PipelineState
from sightline.agents.nodes.analysis import analysis_node
from sightline.agents.nodes.degradation import degradation_node
from sightline.agents.nodes.ingest import ingest_node
from sightline.agents.nodes.normalization import normalization_node
from sightline.agents.nodes.planner_fallback import planner_fallback_node
from sightline.agents.nodes.query_planner import query_planner_node
from sightline.agents.nodes.report import report_node
from sightline.agents.nodes.retrieval import retrieval_node, single_retrieval_node

#: LangGraph parameterises graphs by state, runtime context, input and output. This system
#: uses one state type throughout and no runtime context, so the parameters are pinned once
#: here rather than repeated at every signature.
type SightlineGraph = StateGraph[PipelineState, None, PipelineState, PipelineState]
type CompiledSightlineGraph = CompiledStateGraph[PipelineState, None, PipelineState, PipelineState]


def build_pipeline_graph(deps: NodeDependencies) -> CompiledSightlineGraph:
    """Compile the full pipeline DAG."""
    graph: SightlineGraph = StateGraph(PipelineState)

    graph.add_node(NodeName.INGEST, partial(ingest_node, deps=deps))
    graph.add_node(NodeName.QUERY_PLANNER, partial(query_planner_node, deps=deps))
    graph.add_node(NodeName.PLANNER_FALLBACK, partial(planner_fallback_node, deps=deps))
    graph.add_node(NodeName.RETRIEVAL, partial(retrieval_node, deps=deps))
    graph.add_node(NodeName.NORMALIZATION, partial(normalization_node, deps=deps))
    graph.add_node(NodeName.ANALYSIS, partial(analysis_node, deps=deps))
    graph.add_node(NodeName.DEGRADATION, partial(degradation_node, deps=deps))
    graph.add_node(NodeName.REPORT, partial(report_node, deps=deps))

    graph.set_entry_point(NodeName.INGEST)
    graph.add_edge(NodeName.INGEST, NodeName.QUERY_PLANNER)

    # Branch 1: a usable plan fans out into parallel retrieval; an unusable one diverts to the
    # deterministic fallback planner, which then fans out the same way.
    graph.add_conditional_edges(
        NodeName.QUERY_PLANNER,
        route_after_planner,
        [NodeName.RETRIEVAL, NodeName.PLANNER_FALLBACK],
    )
    graph.add_conditional_edges(
        NodeName.PLANNER_FALLBACK,
        route_after_fallback,
        [NodeName.RETRIEVAL, NodeName.DEGRADATION],
    )

    # Branch 2: evaluated once, after every fan-out branch has joined.
    graph.add_conditional_edges(
        NodeName.RETRIEVAL,
        make_route_after_retrieval(deps.pipeline.min_successful_retrievals),
        [NodeName.NORMALIZATION, NodeName.DEGRADATION],
    )

    # Branch 3 and 4: each stage can still find itself with nothing usable to hand on.
    graph.add_conditional_edges(
        NodeName.NORMALIZATION,
        route_after_normalization,
        [NodeName.ANALYSIS, NodeName.DEGRADATION],
    )
    graph.add_conditional_edges(
        NodeName.ANALYSIS,
        route_after_analysis,
        [NodeName.REPORT, NodeName.DEGRADATION],
    )

    # Degradation converges back into the report: a degraded run still produces a document,
    # flagged as partial, rather than an error (spec S3.5).
    graph.add_edge(NodeName.DEGRADATION, NodeName.REPORT)
    graph.add_edge(NodeName.REPORT, END)

    return graph.compile()


def build_recheck_graph(deps: NodeDependencies) -> CompiledSightlineGraph:
    """Compile the single-query recheck subgraph (spec S4.2)."""
    graph: SightlineGraph = StateGraph(PipelineState)

    graph.add_node(NodeName.RETRIEVAL, partial(single_retrieval_node, deps=deps))
    graph.add_node(NodeName.NORMALIZATION, partial(normalization_node, deps=deps))
    graph.add_node(NodeName.ANALYSIS, partial(analysis_node, deps=deps))

    graph.set_entry_point(NodeName.RETRIEVAL)
    graph.add_conditional_edges(
        NodeName.RETRIEVAL,
        # One branch, so one success is the threshold regardless of the pipeline setting.
        make_route_after_retrieval(1),
        {NodeName.NORMALIZATION: NodeName.NORMALIZATION, NodeName.DEGRADATION: END},
    )
    graph.add_conditional_edges(
        NodeName.NORMALIZATION,
        route_after_normalization,
        {NodeName.ANALYSIS: NodeName.ANALYSIS, NodeName.DEGRADATION: END},
    )
    graph.add_edge(NodeName.ANALYSIS, END)

    return graph.compile()


def render_mermaid(graph: CompiledSightlineGraph) -> str:
    """Render a compiled graph as Mermaid, for the README diagram spec S3.1 requires.

    Generated from the compiled graph rather than hand-drawn, so the diagram cannot fall out
    of step with the topology it documents.
    """
    return graph.get_graph().draw_mermaid()
