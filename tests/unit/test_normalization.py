"""Payload parsing and DataForSEO status classification."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from sightline.agents.contracts.normalized import NormalizedObservation, NormalizedQuery
from sightline.agents.nodes.normalization import (
    build_discovered_query,
    parse_ai_overview,
    parse_keyword_metrics,
    parse_llm_visibility,
    parse_organic_serp,
)
from sightline.application.ports.errors import RetryableError
from sightline.application.ports.search_data_provider import RawApiResult
from sightline.domain.entities.profile import Profile
from sightline.domain.value_objects.enums import RetrievalKind, VisibilityStatus
from sightline.infrastructure.dataforseo.mock import fixtures
from sightline.infrastructure.dataforseo.status import classify_payload


def raw(payload: dict[str, Any], endpoint: str = "/v3/serp") -> RawApiResult:
    return RawApiResult(endpoint=endpoint, payload=payload, from_mock=True)


class TestDataForSEOStatusClassification:
    """DataForSEO answers HTTP 200 for application-level failures."""

    def test_a_healthy_payload_is_usable(self) -> None:
        payload = fixtures.organic_serp(
            keyword="seo", location_code=2840, language_code="en", depth=5
        )
        assert classify_payload(payload, endpoint="/x") is None

    @pytest.mark.parametrize(
        ("status_code", "message", "retryable"),
        [
            (40100, "Unauthorized.", False),
            (40200, "Payment Required.", False),
            (40202, "Rate limit exceeded.", True),
            (50000, "Internal Error.", True),
            (30000, "Task In Queue.", True),
        ],
    )
    def test_envelope_status_codes_are_classified(
        self, status_code: int, message: str, retryable: bool
    ) -> None:
        error = classify_payload(
            {"status_code": status_code, "status_message": message, "tasks": []},
            endpoint="/x",
        )
        assert error is not None
        assert isinstance(error, RetryableError) is retryable

    def test_a_task_level_failure_inside_an_http_200_is_caught(self) -> None:
        error = classify_payload(
            {
                "status_code": 20000,
                "status_message": "Ok.",
                "tasks": [{"status_code": 40501, "status_message": "Invalid Field."}],
            },
            endpoint="/x",
        )
        assert error is not None

    @pytest.mark.parametrize(
        "payload",
        [
            {"status_code": 20000, "status_message": "Ok.", "tasks": []},
            {"tasks": []},
            {"status_code": "not-an-int", "tasks": []},
        ],
        ids=["no-tasks", "missing-status", "wrong-status-type"],
    )
    def test_contract_violations_are_rejected(self, payload: dict[str, Any]) -> None:
        assert classify_payload(payload, endpoint="/x") is not None


class TestOrganicSerpParsing:
    def test_the_profile_domain_is_found_at_its_rank(self, profile: Profile) -> None:
        payload = {
            "status_code": 20000,
            "tasks": [
                {
                    "status_code": 20000,
                    "result": [
                        {
                            "items": [
                                {"type": "organic", "rank_absolute": 1, "domain": "ahrefs.com"},
                                {"type": "organic", "rank_absolute": 2, "domain": "surferseo.com"},
                                {"type": "organic", "rank_absolute": 3, "domain": "moz.com"},
                            ]
                        }
                    ],
                }
            ],
        }
        observation = parse_organic_serp(raw(payload), query_text="seo", profile=profile)

        assert observation.visibility_status is VisibilityStatus.VISIBLE
        assert observation.position == 2
        assert observation.cited_domains == ("ahrefs.com", "surferseo.com", "moz.com")

    def test_absence_is_recorded_as_measured_not_unknown(self, profile: Profile) -> None:
        payload = {
            "status_code": 20000,
            "tasks": [
                {
                    "status_code": 20000,
                    "result": [
                        {"items": [{"type": "organic", "rank_absolute": 1, "domain": "moz.com"}]}
                    ],
                }
            ],
        }
        observation = parse_organic_serp(raw(payload), query_text="seo", profile=profile)
        assert observation.visibility_status is VisibilityStatus.NOT_VISIBLE

    def test_a_subdomain_counts_as_the_brand_but_a_lookalike_does_not(
        self, profile: Profile
    ) -> None:
        def serp(domain: str) -> dict[str, Any]:
            return {
                "status_code": 20000,
                "tasks": [
                    {
                        "status_code": 20000,
                        "result": [
                            {"items": [{"type": "organic", "rank_absolute": 1, "domain": domain}]}
                        ],
                    }
                ],
            }

        assert (
            parse_organic_serp(
                raw(serp("blog.surferseo.com")), query_text="x", profile=profile
            ).visibility_status
            is VisibilityStatus.VISIBLE
        )
        assert (
            parse_organic_serp(
                raw(serp("notsurferseo.com")), query_text="x", profile=profile
            ).visibility_status
            is VisibilityStatus.NOT_VISIBLE
        )

    def test_competitor_positions_are_recorded(self, profile: Profile) -> None:
        payload = {
            "status_code": 20000,
            "tasks": [
                {
                    "status_code": 20000,
                    "result": [
                        {
                            "items": [
                                {"type": "organic", "rank_absolute": 1, "domain": "clearscope.io"},
                                {"type": "organic", "rank_absolute": 4, "domain": "frase.io"},
                            ]
                        }
                    ],
                }
            ],
        }
        observation = parse_organic_serp(raw(payload), query_text="x", profile=profile)
        assert observation.competitor_positions == {"clearscope.io": 1, "frase.io": 4}

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"tasks": None},
            {"tasks": [{"result": None}]},
            {"tasks": [{"result": [{"items": None}]}]},
            {"tasks": [{"result": [{"items": [{"type": "organic"}]}]}]},
            {"tasks": ["not-a-mapping"]},
        ],
        ids=["empty", "null-tasks", "null-result", "null-items", "item-missing-keys", "bad-task"],
    )
    def test_malformed_payloads_degrade_instead_of_raising(
        self, profile: Profile, payload: dict[str, Any]
    ) -> None:
        """One unexpected field must not abort a run."""
        observation = parse_organic_serp(raw(payload), query_text="x", profile=profile)
        assert isinstance(observation, NormalizedObservation)


class TestOtherParsers:
    def test_ai_overview_citations_are_extracted(self, profile: Profile) -> None:
        payload = fixtures.ai_overview(
            keyword="best seo tools", location_code=2840, language_code="en"
        )
        observation = parse_ai_overview(raw(payload), query_text="best seo tools", profile=profile)
        assert observation.kind is RetrievalKind.AI_OVERVIEW

    def test_an_absent_ai_overview_is_unknown_not_absent(self, profile: Profile) -> None:
        """No AI Overview generated is missing data, not proof the brand is not cited."""
        payload = {
            "status_code": 20000,
            "tasks": [{"status_code": 20000, "result": [{"items": []}]}],
        }
        observation = parse_ai_overview(raw(payload), query_text="x", profile=profile)
        assert observation.visibility_status is VisibilityStatus.UNKNOWN

    def test_llm_citations_are_extracted(self, profile: Profile) -> None:
        payload = fixtures.llm_visibility(prompt="best seo tool?", llm_name="chat_gpt")
        observation = parse_llm_visibility(
            raw(payload), query_text="best seo tool?", profile=profile
        )
        assert observation.cited_domains

    def test_keyword_metrics_reports_size_but_never_visibility(self, profile: Profile) -> None:
        payload = fixtures.keyword_metrics(
            keywords=["best seo tools"], location_code=2840, language_code="en"
        )
        observation = parse_keyword_metrics(
            raw(payload), query_text="best seo tools", profile=profile
        )

        assert observation.search_volume is not None
        assert observation.competition_index is not None
        assert observation.visibility_status is VisibilityStatus.UNKNOWN


class TestVisibilityMerging:
    def _observation(self, **kwargs: Any) -> NormalizedObservation:
        defaults: dict[str, Any] = {
            "query_text": "q",
            "kind": RetrievalKind.ORGANIC_SERP,
            "source_endpoint": "/x",
            "visibility_status": VisibilityStatus.UNKNOWN,
        }
        return NormalizedObservation(**{**defaults, **kwargs})

    def test_the_best_position_across_capabilities_wins(self) -> None:
        merged = NormalizedQuery(
            query_text="q",
            observations=(
                self._observation(visibility_status=VisibilityStatus.VISIBLE, position=7),
                self._observation(visibility_status=VisibilityStatus.VISIBLE, position=2),
            ),
        )
        assert merged.best_visibility() == (VisibilityStatus.VISIBLE, 2)

    def test_visible_anywhere_beats_absent_elsewhere(self) -> None:
        merged = NormalizedQuery(
            query_text="q",
            observations=(
                self._observation(visibility_status=VisibilityStatus.NOT_VISIBLE),
                self._observation(visibility_status=VisibilityStatus.VISIBLE, position=3),
            ),
        )
        assert merged.best_visibility() == (VisibilityStatus.VISIBLE, 3)

    def test_only_unknown_observations_yield_unknown(self) -> None:
        """No capability actually looked, so absence must not be inferred."""
        merged = NormalizedQuery(
            query_text="q", observations=(self._observation(), self._observation())
        )
        assert merged.best_visibility() == (VisibilityStatus.UNKNOWN, None)

    def test_a_citation_without_a_rank_is_still_visible(self) -> None:
        merged = NormalizedQuery(
            query_text="q",
            observations=(
                self._observation(
                    kind=RetrievalKind.AI_OVERVIEW,
                    visibility_status=VisibilityStatus.VISIBLE,
                ),
            ),
        )
        status, position = merged.best_visibility()
        assert status is VisibilityStatus.VISIBLE
        assert position == 1


class TestEntityAssembly:
    def test_missing_metrics_fall_back_to_neutral_not_favourable(self, profile: Profile) -> None:
        """An unmeasured query must not float to the top of the opportunity ranking."""
        normalized = NormalizedQuery(
            query_text="unmeasured query",
            observations=(
                NormalizedObservation(
                    query_text="unmeasured query",
                    kind=RetrievalKind.ORGANIC_SERP,
                    source_endpoint="/x",
                    visibility_status=VisibilityStatus.NOT_VISIBLE,
                ),
            ),
        )
        query = build_discovered_query(normalized, profile=profile, run_uuid=uuid4())

        assert int(query.estimated_search_volume) == 0
        assert int(query.competitive_difficulty) == 50
