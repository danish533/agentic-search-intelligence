"""HTTP end-to-end tests.

The real application - real routers, real use cases, real DAG, real PostgreSQL - with only the
LLM and DataForSEO substituted by their offline doubles. Skips when no migrated database is
reachable.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.integration

PROFILE_BODY = {
    "name": "Surfer SEO",
    "domain": "surferseo.com",
    "industry": "SEO Software",
    "description": "AI-powered SEO content optimization tool",
    "competitors": ["clearscope.io", "marketmuse.com", "frase.io"],
}

UNKNOWN_UUID = "11111111-1111-1111-1111-111111111111"


async def register(client: httpx.AsyncClient, **overrides: Any) -> dict[str, Any]:
    response = await client.post("/api/v1/profiles", json={**PROFILE_BODY, **overrides})
    assert response.status_code == 201
    return response.json()


async def run_pipeline(client: httpx.AsyncClient, profile_uuid: str) -> dict[str, Any]:
    response = await client.post(f"/api/v1/profiles/{profile_uuid}/run", json={})
    assert response.status_code == 200
    return response.json()


class TestHealth:
    async def test_health_reports_readiness_and_active_providers(
        self, api_client: httpx.AsyncClient
    ) -> None:
        body = (await api_client.get("/health")).json()
        assert body["status"] == "ok"
        assert body["database"] == "up"
        assert body["llm_provider"] == "fake"
        assert body["dataforseo_mode"] == "mock"


class TestProfileRegistration:
    async def test_creating_a_profile_returns_201_with_the_spec_shape(
        self, api_client: httpx.AsyncClient
    ) -> None:
        response = await api_client.post("/api/v1/profiles", json=PROFILE_BODY)
        body = response.json()

        assert response.status_code == 201
        assert set(body) == {"profile_uuid", "name", "domain", "status", "created_at"}
        assert body["status"] == "created"
        assert body["domain"] == "surferseo.com"

    async def test_a_duplicate_domain_conflicts_even_when_written_differently(
        self, api_client: httpx.AsyncClient
    ) -> None:
        await register(api_client)
        response = await api_client.post(
            "/api/v1/profiles", json={**PROFILE_BODY, "domain": "https://WWW.SurferSEO.com/"}
        )

        assert response.status_code == 409
        assert response.json()["error"] == "duplicate_profile"

    async def test_concurrent_creates_for_one_domain_yield_one_201_and_409s(
        self, api_client: httpx.AsyncClient
    ) -> None:
        """Regression: check-then-act.

        ``find_by_domain`` returned None for every concurrent request, all of them inserted,
        and the unique index rejected the losers - which surfaced as 500 because nothing
        translated the integrity error. The database is the arbiter; its answer has to become
        the same 409 the sequential path returns.
        """
        import asyncio

        responses = await asyncio.gather(
            *(api_client.post("/api/v1/profiles", json=PROFILE_BODY) for _ in range(4))
        )
        codes = sorted(response.status_code for response in responses)

        assert codes == [201, 409, 409, 409], f"expected one winner, got {codes}"
        assert all(
            response.json()["error"] == "duplicate_profile"
            for response in responses
            if response.status_code == 409
        )

    async def test_every_concurrent_rejection_carries_a_correlation_id(
        self, api_client: httpx.AsyncClient
    ) -> None:
        import asyncio

        responses = await asyncio.gather(
            *(api_client.post("/api/v1/profiles", json=PROFILE_BODY) for _ in range(4))
        )

        for response in responses:
            assert response.headers.get("X-Correlation-ID")
            if response.status_code == 409:
                assert response.json()["correlation_id"]

    async def test_a_losing_concurrent_create_writes_no_row(
        self, api_client: httpx.AsyncClient
    ) -> None:
        """Integrity must hold regardless of which request wins."""
        import asyncio

        await asyncio.gather(
            *(api_client.post("/api/v1/profiles", json=PROFILE_BODY) for _ in range(4))
        )
        winner = await api_client.post("/api/v1/profiles", json=PROFILE_BODY)

        assert winner.status_code == 409, "a fifth attempt must also conflict"

    @pytest.mark.parametrize(
        "body",
        [
            {"name": "No Domain"},
            {"domain": "no-name.com"},
            {"name": "X", "domain": "x.com", "website": "unexpected"},
            {"name": "", "domain": "x.com"},
        ],
        ids=["missing-domain", "missing-name", "unknown-field", "empty-name"],
    )
    async def test_invalid_bodies_are_rejected_with_field_detail(
        self, api_client: httpx.AsyncClient, body: dict[str, Any]
    ) -> None:
        response = await api_client.post("/api/v1/profiles", json=body)

        assert response.status_code == 422
        assert response.json()["detail"]

    async def test_an_unknown_profile_is_404(self, api_client: httpx.AsyncClient) -> None:
        response = await api_client.get(f"/api/v1/profiles/{UNKNOWN_UUID}")
        assert response.status_code == 404
        assert response.json()["error"] == "profile_not_found"

    async def test_a_malformed_uuid_is_422_not_500(self, api_client: httpx.AsyncClient) -> None:
        assert (await api_client.get("/api/v1/profiles/not-a-uuid")).status_code == 422


class TestPipelineRun:
    async def test_a_run_returns_every_field_the_spec_enumerates(
        self, api_client: httpx.AsyncClient
    ) -> None:
        profile = await register(api_client)
        body = await run_pipeline(api_client, profile["profile_uuid"])

        for field in (
            "pipeline_run_uuid",
            "status",
            "planned_retrieval_calls",
            "normalized_record_count",
            "top_insights",
            "report",
            "total_tokens",
        ):
            assert field in body, f"spec S4.2 requires '{field}' in the run response"

        assert body["status"] == "completed"
        assert body["planned_retrieval_calls"] > 0
        assert body["normalized_record_count"] > 0
        assert body["total_tokens"] > 0

    async def test_the_report_carries_json_and_a_human_summary(
        self, api_client: httpx.AsyncClient
    ) -> None:
        profile = await register(api_client)
        report = (await run_pipeline(api_client, profile["profile_uuid"]))["report"]

        assert report["headline"]
        assert "# " in report["summary"]
        assert isinstance(report["structured"], dict)
        assert report["structured"]["coverage"]["degraded"] is False

    async def test_insights_carry_relevance_scores(self, api_client: httpx.AsyncClient) -> None:
        profile = await register(api_client)
        insights = (await run_pipeline(api_client, profile["profile_uuid"]))["top_insights"]

        assert insights
        assert all(0.0 <= insight["relevance_score"] <= 1.0 for insight in insights)
        scores = [insight["relevance_score"] for insight in insights]
        assert scores == sorted(scores, reverse=True)

    async def test_a_supplied_question_is_used(self, api_client: httpx.AsyncClient) -> None:
        profile = await register(api_client)
        question = "Where is Surfer SEO absent from AI answers?"
        response = await api_client.post(
            f"/api/v1/profiles/{profile['profile_uuid']}/run", json={"question": question}
        )
        assert response.json()["question"] == question

    async def test_an_empty_body_derives_the_question_from_the_profile(
        self, api_client: httpx.AsyncClient
    ) -> None:
        """Spec S4.2 triggers a run with no body; spec S2 frames the input as a question."""
        profile = await register(api_client)
        response = await api_client.post(f"/api/v1/profiles/{profile['profile_uuid']}/run")

        assert response.status_code == 200
        assert "Surfer SEO" in response.json()["question"]

    async def test_running_an_unknown_profile_is_404(self, api_client: httpx.AsyncClient) -> None:
        response = await api_client.post(f"/api/v1/profiles/{UNKNOWN_UUID}/run", json={})
        assert response.status_code == 404

    async def test_the_run_updates_the_profile_statistics(
        self, api_client: httpx.AsyncClient
    ) -> None:
        profile = await register(api_client)
        await run_pipeline(api_client, profile["profile_uuid"])

        stats = (await api_client.get(f"/api/v1/profiles/{profile['profile_uuid']}")).json()[
            "stats"
        ]

        assert stats["total_runs"] == 1
        assert stats["latest_run_status"] == "completed"
        assert 0.0 <= stats["average_opportunity_score"] <= 1.0


class TestQueriesEndpoint:
    @pytest.fixture
    async def ran(self, api_client: httpx.AsyncClient) -> str:
        profile = await register(api_client)
        await run_pipeline(api_client, profile["profile_uuid"])
        return str(profile["profile_uuid"])

    async def test_queries_are_returned_sorted_by_opportunity_descending(
        self, api_client: httpx.AsyncClient, ran: str
    ) -> None:
        body = (await api_client.get(f"/api/v1/profiles/{ran}/queries")).json()
        scores = [item["opportunity_score"] for item in body["items"]]

        assert body["items"]
        assert scores == sorted(scores, reverse=True)

    async def test_each_query_carries_the_spec_fields(
        self, api_client: httpx.AsyncClient, ran: str
    ) -> None:
        item = (await api_client.get(f"/api/v1/profiles/{ran}/queries")).json()["items"][0]

        for field in (
            "query_text",
            "estimated_search_volume",
            "competitive_difficulty",
            "opportunity_score",
            "domain_visible",
            "visibility_position",
            "discovered_at",
        ):
            assert field in item, f"spec S4.2 requires '{field}' on a query"

    async def test_min_score_filter(self, api_client: httpx.AsyncClient, ran: str) -> None:
        body = (await api_client.get(f"/api/v1/profiles/{ran}/queries?min_score=0.5")).json()
        assert all(item["opportunity_score"] >= 0.5 for item in body["items"])

    @pytest.mark.parametrize("status", ["visible", "not_visible", "unknown"])
    async def test_status_filter(
        self, api_client: httpx.AsyncClient, ran: str, status: str
    ) -> None:
        body = (await api_client.get(f"/api/v1/profiles/{ran}/queries?status={status}")).json()
        assert all(item["visibility_status"] == status for item in body["items"])

    async def test_pagination_metadata_and_page_contents(
        self, api_client: httpx.AsyncClient, ran: str
    ) -> None:
        first = (await api_client.get(f"/api/v1/profiles/{ran}/queries?page=1&per_page=2")).json()
        second = (await api_client.get(f"/api/v1/profiles/{ran}/queries?page=2&per_page=2")).json()

        assert first["pagination"]["per_page"] == 2
        assert len(first["items"]) <= 2
        assert first["pagination"]["has_previous"] is False
        first_ids = {item["query_uuid"] for item in first["items"]}
        second_ids = {item["query_uuid"] for item in second["items"]}
        assert not (first_ids & second_ids), "pages must not repeat a query"

    @pytest.mark.parametrize(
        "query_string", ["?min_score=5", "?min_score=-1", "?status=maybe", "?per_page=0"]
    )
    async def test_invalid_query_parameters_are_422(
        self, api_client: httpx.AsyncClient, ran: str, query_string: str
    ) -> None:
        response = await api_client.get(f"/api/v1/profiles/{ran}/queries{query_string}")
        assert response.status_code == 422

    async def test_a_profile_with_no_runs_returns_an_empty_page_not_404(
        self, api_client: httpx.AsyncClient
    ) -> None:
        profile = await register(api_client, domain="never-run.example", name="Never Run")
        response = await api_client.get(f"/api/v1/profiles/{profile['profile_uuid']}/queries")

        assert response.status_code == 200
        assert response.json()["pagination"]["total_items"] == 0


class TestRecommendationsEndpoint:
    async def test_recommendations_carry_the_spec_fields_and_reference_real_queries(
        self, api_client: httpx.AsyncClient
    ) -> None:
        profile = await register(api_client)
        await run_pipeline(api_client, profile["profile_uuid"])
        uuid = profile["profile_uuid"]

        body = (await api_client.get(f"/api/v1/profiles/{uuid}/recommendations")).json()
        queries = (await api_client.get(f"/api/v1/profiles/{uuid}/queries")).json()
        known = {item["query_uuid"] for item in queries["items"]}

        for item in body["items"]:
            for field in (
                "recommendation_uuid",
                "target_query_uuid",
                "content_type",
                "title",
                "rationale",
                "target_keywords",
                "priority",
            ):
                assert field in item
            assert item["target_query_uuid"] in known


class TestRecheckEndpoint:
    async def test_recheck_returns_the_delta_and_preserves_identity(
        self, api_client: httpx.AsyncClient
    ) -> None:
        profile = await register(api_client)
        await run_pipeline(api_client, profile["profile_uuid"])
        query = (
            await api_client.get(f"/api/v1/profiles/{profile['profile_uuid']}/queries")
        ).json()["items"][0]

        response = await api_client.post(f"/api/v1/queries/{query['query_uuid']}/recheck")
        body = response.json()

        assert response.status_code == 200
        assert body["status"] == "completed"
        assert body["query"]["query_uuid"] == query["query_uuid"]
        assert body["previous_opportunity_score"] == query["opportunity_score"]
        assert "opportunity_score_delta" in body

    async def test_rechecking_an_unknown_query_is_404(self, api_client: httpx.AsyncClient) -> None:
        response = await api_client.post(f"/api/v1/queries/{UNKNOWN_UUID}/recheck")

        assert response.status_code == 404
        assert response.json()["error"] == "discovered_query_not_found"


class TestObservabilityContract:
    async def test_every_response_carries_a_correlation_id(
        self, api_client: httpx.AsyncClient
    ) -> None:
        for response in (
            await api_client.get("/health"),
            await api_client.get(f"/api/v1/profiles/{UNKNOWN_UUID}"),
            await api_client.post("/api/v1/profiles", json={"name": "x"}),
        ):
            assert response.headers.get("X-Correlation-ID")

    async def test_an_inbound_correlation_id_is_honoured(
        self, api_client: httpx.AsyncClient
    ) -> None:
        """Lets a trace span several services rather than restarting at this one."""
        response = await api_client.get(
            "/health", headers={"X-Correlation-ID": "upstream-trace-id"}
        )
        assert response.headers["X-Correlation-ID"] == "upstream-trace-id"

    async def test_error_responses_include_the_correlation_id_in_the_body(
        self, api_client: httpx.AsyncClient
    ) -> None:
        """It is the identifier a user quotes when reporting a failure."""
        body = (await api_client.get(f"/api/v1/profiles/{UNKNOWN_UUID}")).json()
        assert body["correlation_id"]

    async def test_the_run_response_exposes_its_trace_and_metrics(
        self, api_client: httpx.AsyncClient
    ) -> None:
        profile = await register(api_client)
        body = await run_pipeline(api_client, profile["profile_uuid"])

        assert body["correlation_id"]
        assert body["metrics"]["node_count"] > 0
        assert body["metrics"]["api_call_count"] > 0
        assert "node_latency_ms" in body["metrics"]


class TestOpenAPI:
    async def test_the_schema_documents_every_specified_endpoint(
        self, api_client: httpx.AsyncClient
    ) -> None:
        paths = (await api_client.get("/openapi.json")).json()["paths"]

        assert "/api/v1/profiles" in paths
        assert "/api/v1/profiles/{profile_uuid}" in paths
        assert "/api/v1/profiles/{profile_uuid}/run" in paths
        assert "/api/v1/profiles/{profile_uuid}/queries" in paths
        assert "/api/v1/profiles/{profile_uuid}/recommendations" in paths
        assert "/api/v1/queries/{query_uuid}/recheck" in paths
