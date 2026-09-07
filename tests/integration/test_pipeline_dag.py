"""End-to-end DAG execution.

**Spec S5 mandated test #1: "one happy-path run".**
**Spec S5 mandated test #2: "one simulated API failure with successful retry/fallback".**

Exercises the real graph - real nodes, real routing, real validation gate, real retry and
circuit-breaker code - against the offline LLM and the mock transport. No network, no database,
no API key.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from sightline.agents.engine import LangGraphPipelineEngine
from sightline.agents.graph.builder import build_pipeline_graph, build_recheck_graph
from sightline.agents.graph.names import NodeName
from sightline.application.dto.pipeline import PipelineRequest, RecheckRequest
from sightline.application.ports.llm_provider import LLMError
from sightline.domain.entities.profile import Profile
from sightline.domain.value_objects.enums import DegradationReason, RunStatus, VisibilityStatus
from sightline.infrastructure.dataforseo.mock.transport import MockFault, MockTransport
from sightline.infrastructure.llm.fake import FakeLLMProvider
from sightline.observability.correlation import new_correlation_id

QUESTION = "How does Surfer SEO show up in AI answers and search results for SEO content tools?"


def request_for(profile: Profile) -> PipelineRequest:
    return PipelineRequest(
        profile=profile,
        question=QUESTION,
        run_uuid=uuid.uuid4(),
        correlation_id=new_correlation_id(),
    )


class UnavailableLLM(FakeLLMProvider):
    """A provider whose structured completions always fail - a total LLM outage."""

    async def complete_structured(self, *, system_prompt, user_prompt, schema):  # type: ignore[no-untyped-def, override]
        raise LLMError("fake", "simulated provider outage")


class TestHappyPath:
    """Spec S5 mandated test #1."""

    async def test_a_full_run_completes_and_produces_a_report(
        self, pipeline_engine: LangGraphPipelineEngine, profile: Profile
    ) -> None:
        outcome = await pipeline_engine.run(request_for(profile))

        assert outcome.status is RunStatus.COMPLETED
        assert outcome.degradation_reason is None
        assert outcome.planned_retrieval_count > 0
        assert outcome.successful_retrieval_count == outcome.planned_retrieval_count
        assert outcome.normalized_record_count > 0
        assert outcome.queries and outcome.insights
        assert outcome.report.headline and outcome.report.summary
        assert outcome.total_tokens > 0

    async def test_every_pipeline_node_executed(
        self, pipeline_engine: LangGraphPipelineEngine, profile: Profile
    ) -> None:
        outcome = await pipeline_engine.run(request_for(profile))
        executed = set(outcome.metrics_summary["node_latency_ms"])

        assert executed == {
            NodeName.INGEST,
            NodeName.QUERY_PLANNER,
            NodeName.RETRIEVAL,
            NodeName.NORMALIZATION,
            NodeName.ANALYSIS,
            NodeName.REPORT,
        }
        assert NodeName.DEGRADATION not in executed
        assert NodeName.PLANNER_FALLBACK not in executed

    async def test_retrieval_fans_out_concurrently_over_the_plan(
        self,
        pipeline_engine: LangGraphPipelineEngine,
        profile: Profile,
        mock_transport: MockTransport,
    ) -> None:
        """One branch per planned sub-query, each making several tool calls."""
        outcome = await pipeline_engine.run(request_for(profile))
        assert mock_transport.call_count > outcome.planned_retrieval_count

    async def test_measured_queries_are_scored_and_internally_consistent(
        self, pipeline_engine: LangGraphPipelineEngine, profile: Profile
    ) -> None:
        outcome = await pipeline_engine.run(request_for(profile))

        for query in outcome.queries:
            assert 0.0 <= query.opportunity_score.value <= 1.0
            assert 0 <= int(query.competitive_difficulty) <= 100
            assert query.domain_visible == (query.visibility_status is VisibilityStatus.VISIBLE)
            assert (query.visibility_position is not None) == query.domain_visible

    async def test_the_report_carries_both_required_forms(
        self, pipeline_engine: LangGraphPipelineEngine, profile: Profile
    ) -> None:
        """Spec S4.2 requires JSON *and* a human-readable summary."""
        outcome = await pipeline_engine.run(request_for(profile))

        assert "# " in outcome.report.summary
        for key in ("headline", "coverage", "top_opportunities", "insights"):
            assert key in outcome.report.structured
        assert outcome.report.structured["coverage"]["degraded"] is False

    async def test_recommendations_reference_queries_that_were_actually_measured(
        self, pipeline_engine: LangGraphPipelineEngine, profile: Profile
    ) -> None:
        """A hallucinated reference is dropped, never attached to the wrong query."""
        outcome = await pipeline_engine.run(request_for(profile))
        measured = {query.uuid for query in outcome.queries}

        for recommendation in outcome.recommendations:
            assert recommendation.target_query_uuid in measured
        for insight in outcome.insights:
            assert set(insight.supporting_query_uuids) <= measured

    async def test_runs_are_deterministic(
        self, pipeline_engine: LangGraphPipelineEngine, profile: Profile
    ) -> None:
        first = await pipeline_engine.run(request_for(profile))
        second = await pipeline_engine.run(request_for(profile))

        assert [q.query_text for q in first.queries] == [q.query_text for q in second.queries]
        assert [q.opportunity_score.value for q in first.queries] == [
            q.opportunity_score.value for q in second.queries
        ]


class TestSimulatedApiFailure:
    """Spec S5 mandated test #2: simulated API failure with successful retry / fallback."""

    @pytest.mark.parametrize(
        "faults",
        [
            pytest.param(
                (MockFault.rate_limited(retry_after_seconds=0.1), MockFault.server_error()),
                id="429-then-503",
            ),
            pytest.param(
                (
                    MockFault.transport_error(httpx.ConnectError("connection reset by peer")),
                    MockFault.transport_error(httpx.ReadTimeout("read timed out")),
                ),
                id="connection-reset-then-timeout",
            ),
        ],
    )
    async def test_a_transient_failure_is_retried_and_the_call_recovers(
        self,
        pipeline_engine: LangGraphPipelineEngine,
        profile: Profile,
        mock_transport: MockTransport,
        recorded_sleeps: list[float],
        faults: tuple[MockFault, ...],
    ) -> None:
        """Two transient failures on one call; the third attempt succeeds.

        The retry budget is three attempts, so injecting two faults proves recovery rather
        than exhaustion: no branch is lost, and every planned retrieval still succeeds.
        """
        mock_transport.queue_faults(*faults)

        outcome = await pipeline_engine.run(request_for(profile))

        assert outcome.status is RunStatus.COMPLETED
        assert outcome.successful_retrieval_count == outcome.planned_retrieval_count, (
            "the failing call recovered, so no branch should have been lost"
        )
        assert outcome.queries, "retries must have recovered real data"
        assert outcome.metrics_summary["total_retries"] == len(faults)
        assert len(recorded_sleeps) == len(faults), "backoff applied between each attempt"
        assert all(delay > 0 for delay in recorded_sleeps)

    async def test_a_rate_limit_honours_the_servers_retry_after(
        self,
        pipeline_engine: LangGraphPipelineEngine,
        profile: Profile,
        mock_transport: MockTransport,
        recorded_sleeps: list[float],
    ) -> None:
        mock_transport.queue_faults(MockFault.rate_limited(retry_after_seconds=2.0))

        outcome = await pipeline_engine.run(request_for(profile))

        assert outcome.status is RunStatus.COMPLETED
        assert recorded_sleeps[0] >= 2.0, "must never retry sooner than the server instructed"

    async def test_a_total_outage_degrades_to_partial_instead_of_crashing(
        self,
        pipeline_engine: LangGraphPipelineEngine,
        profile: Profile,
        mock_transport: MockTransport,
    ) -> None:
        """Spec S3.5: degrade gracefully rather than failing the whole pipeline."""
        mock_transport.queue_faults(*[MockFault.server_error()] * 200)

        outcome = await pipeline_engine.run(request_for(profile))

        assert outcome.status is RunStatus.PARTIAL
        assert outcome.degradation_reason in {
            DegradationReason.ALL_RETRIEVALS_FAILED,
            DegradationReason.UPSTREAM_CIRCUIT_OPEN,
        }
        assert outcome.successful_retrieval_count == 0
        assert outcome.report.headline, "a degraded run still returns a report"
        assert outcome.report.structured["coverage"]["degraded"] is True
        assert outcome.error_message

    async def test_the_circuit_breaker_stops_hammering_a_dead_dependency(
        self,
        pipeline_engine: LangGraphPipelineEngine,
        profile: Profile,
        mock_transport: MockTransport,
    ) -> None:
        mock_transport.queue_faults(*[MockFault.server_error()] * 500)

        await pipeline_engine.run(request_for(profile))

        # Without the breaker every branch would spend its full retry budget on every tool.
        assert mock_transport.call_count < 40

    async def test_an_auth_failure_is_not_retried(
        self,
        pipeline_engine: LangGraphPipelineEngine,
        profile: Profile,
        mock_transport: MockTransport,
        recorded_sleeps: list[float],
    ) -> None:
        """Retrying a bad credential reproduces it exactly - it must fail fast."""
        mock_transport.queue_faults(*[MockFault.auth_failure()] * 200)

        outcome = await pipeline_engine.run(request_for(profile))

        assert outcome.status is RunStatus.PARTIAL
        assert recorded_sleeps == []

    async def test_an_llm_outage_routes_to_the_deterministic_fallback_planner(
        self, node_dependencies, profile: Profile
    ) -> None:
        """Spec S3.1: route to a fallback path when a node cannot produce usable output."""
        import dataclasses

        engine = LangGraphPipelineEngine(
            dataclasses.replace(node_dependencies, llm=UnavailableLLM())
        )

        outcome = await engine.run(request_for(profile))
        executed = set(outcome.metrics_summary["node_latency_ms"])

        assert NodeName.PLANNER_FALLBACK in executed
        assert outcome.planned_retrieval_count > 0, "the fallback plan is a real plan"
        assert outcome.queries, "data was still retrieved without any LLM"
        assert outcome.report.headline, "the report rendered without an LLM"
        assert outcome.status is RunStatus.PARTIAL


class TestRecheckSubgraph:
    async def test_recheck_reuses_the_pipeline_nodes_and_preserves_identity(
        self, pipeline_engine: LangGraphPipelineEngine, profile: Profile
    ) -> None:
        run = await pipeline_engine.run(request_for(profile))
        original = run.queries[0]

        result = await pipeline_engine.recheck(
            RecheckRequest(
                profile=profile,
                query=original,
                run_uuid=original.run_uuid,
                correlation_id=new_correlation_id(),
            )
        )

        assert result.status is RunStatus.COMPLETED
        assert result.query.uuid == original.uuid
        assert result.query.run_uuid == original.run_uuid
        assert result.query.query_text == original.query_text
        assert set(result.metrics_summary["node_latency_ms"]) == {
            NodeName.RETRIEVAL,
            NodeName.NORMALIZATION,
            NodeName.ANALYSIS,
        }

    async def test_a_failed_recheck_reports_partial_and_keeps_the_original(
        self,
        pipeline_engine: LangGraphPipelineEngine,
        profile: Profile,
        mock_transport: MockTransport,
    ) -> None:
        run = await pipeline_engine.run(request_for(profile))
        original = run.queries[0]
        mock_transport.queue_faults(*[MockFault.server_error()] * 200)

        result = await pipeline_engine.recheck(
            RecheckRequest(
                profile=profile,
                query=original,
                run_uuid=original.run_uuid,
                correlation_id=new_correlation_id(),
            )
        )

        assert result.status is RunStatus.PARTIAL
        assert result.query is original, "a failed recheck must not overwrite good data"


class TestGraphTopology:
    def test_the_graph_is_non_linear(self, node_dependencies) -> None:
        """Spec S6 grades a non-linear structure: branches, not a straight chain."""
        mermaid = build_pipeline_graph(node_dependencies).get_graph().draw_mermaid()
        conditional_edges = [line for line in mermaid.splitlines() if "-.->" in line]

        assert len(conditional_edges) >= 8, "expected multiple conditional branch points"
        assert "degradation --> report" in mermaid

    def test_both_graphs_compile(self, node_dependencies) -> None:
        assert build_pipeline_graph(node_dependencies) is not None
        assert build_recheck_graph(node_dependencies) is not None

    @pytest.mark.parametrize("node", list(NodeName))
    def test_every_named_node_is_present_in_the_compiled_graph(
        self, node_dependencies, node: NodeName
    ) -> None:
        mermaid = build_pipeline_graph(node_dependencies).get_graph().draw_mermaid()
        assert node.value in mermaid
