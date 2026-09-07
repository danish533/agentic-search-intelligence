"""Domain entities, value objects and the opportunity-scoring service."""

from __future__ import annotations

from uuid import uuid4

import pytest

from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.entities.pipeline_run import PipelineRun
from sightline.domain.entities.profile import Profile, normalize_domain
from sightline.domain.entities.recommendation import Recommendation
from sightline.domain.errors import DomainValidationError, InvalidRunTransitionError
from sightline.domain.services.opportunity_scoring import (
    WEIGHT_DEMAND,
    WEIGHT_VISIBILITY_GAP,
    calculate_opportunity_score,
)
from sightline.domain.value_objects.enums import (
    ContentType,
    DegradationReason,
    Priority,
    RunStatus,
    VisibilityStatus,
)
from sightline.domain.value_objects.scores import (
    CompetitiveDifficulty,
    OpportunityScore,
    SearchVolume,
)


class TestValueObjects:
    @pytest.mark.parametrize("value", [-0.1, 1.5, True, "0.5", None])
    def test_out_of_range_or_non_numeric_scores_are_rejected(self, value: object) -> None:
        with pytest.raises(DomainValidationError):
            OpportunityScore(value)  # type: ignore[arg-type]

    def test_clamped_accepts_overshoot_from_a_weighted_sum(self) -> None:
        assert OpportunityScore.clamped(1.0000002).value == 1.0
        assert OpportunityScore.clamped(-0.0000001).value == 0.0

    def test_bounds_are_class_constants_not_fields(self) -> None:
        """Regression for B1: annotating bounds as fields put them in __slots__ and __eq__."""
        assert OpportunityScore(0.5) == OpportunityScore(0.5)
        assert OpportunityScore(0.2) < OpportunityScore(0.8)
        assert OpportunityScore.MAXIMUM == 1.0

    def test_difficulty_normalises_to_a_ratio(self) -> None:
        assert CompetitiveDifficulty(42).as_ratio == 0.42

    def test_search_volume_rejects_negatives(self) -> None:
        with pytest.raises(DomainValidationError):
            SearchVolume(-1)


class TestOpportunityScoring:
    def test_score_spans_the_full_range(self) -> None:
        floor = calculate_opportunity_score(
            search_volume=SearchVolume(0),
            competitive_difficulty=CompetitiveDifficulty(100),
            visibility_status=VisibilityStatus.VISIBLE,
            visibility_position=1,
        )
        ceiling = calculate_opportunity_score(
            search_volume=SearchVolume(100_000),
            competitive_difficulty=CompetitiveDifficulty(0),
            visibility_status=VisibilityStatus.NOT_VISIBLE,
        )
        assert floor.value == 0.0
        assert ceiling.value == 1.0

    def test_an_unmeasured_query_scores_neutrally_not_maximally(self) -> None:
        """A retrieval failure must not be laundered into a maximum-opportunity signal."""
        common = {
            "search_volume": SearchVolume(8100),
            "competitive_difficulty": CompetitiveDifficulty(45),
        }
        unknown = calculate_opportunity_score(**common, visibility_status=VisibilityStatus.UNKNOWN)
        absent = calculate_opportunity_score(
            **common, visibility_status=VisibilityStatus.NOT_VISIBLE
        )
        ranked_first = calculate_opportunity_score(
            **common, visibility_status=VisibilityStatus.VISIBLE, visibility_position=1
        )

        assert ranked_first.value < unknown.value < absent.value

    def test_a_better_position_scores_lower_than_a_worse_one(self) -> None:
        common = {
            "search_volume": SearchVolume(5000),
            "competitive_difficulty": CompetitiveDifficulty(50),
            "visibility_status": VisibilityStatus.VISIBLE,
        }
        assert (
            calculate_opportunity_score(**common, visibility_position=1).value
            < calculate_opportunity_score(**common, visibility_position=5).value
            < calculate_opportunity_score(**common, visibility_position=11).value
        )

    def test_the_visibility_gap_outweighs_raw_demand(self) -> None:
        """The product is about closing gaps, so the gap term must dominate."""
        assert WEIGHT_VISIBILITY_GAP > WEIGHT_DEMAND

    def test_volume_is_log_scaled_so_small_queries_are_not_flattened(self) -> None:
        common = {
            "competitive_difficulty": CompetitiveDifficulty(50),
            "visibility_status": VisibilityStatus.NOT_VISIBLE,
        }
        low_to_mid = (
            calculate_opportunity_score(search_volume=SearchVolume(1000), **common).value
            - calculate_opportunity_score(search_volume=SearchVolume(100), **common).value
        )
        high_to_higher = (
            calculate_opportunity_score(search_volume=SearchVolume(100_000), **common).value
            - calculate_opportunity_score(search_volume=SearchVolume(99_000), **common).value
        )
        assert low_to_mid > high_to_higher


class TestProfile:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://WWW.SurferSEO.com/pricing/", "surferseo.com"),
            ("http://example.com", "example.com"),
            ("Example.COM/", "example.com"),
            ("www.example.com", "example.com"),
        ],
    )
    def test_domain_is_normalised(self, raw: str, expected: str) -> None:
        assert normalize_domain(raw) == expected

    def test_competitors_are_normalised_deduplicated_and_self_reference_dropped(self) -> None:
        profile = Profile(
            name="Surfer SEO",
            domain="surferseo.com",
            industry="SEO",
            description="",
            competitors=(
                "https://clearscope.io",
                "CLEARSCOPE.io",
                "surferseo.com",
                "frase.io",
            ),
        )
        assert profile.competitors == ("clearscope.io", "frase.io")

    @pytest.mark.parametrize("domain", ["", "   ", "not-a-domain", "http://"])
    def test_invalid_domains_are_rejected(self, domain: str) -> None:
        with pytest.raises(DomainValidationError):
            Profile(name="X", domain=domain, industry="", description="")

    def test_a_research_question_can_be_derived_from_the_profile_alone(self) -> None:
        profile = Profile(
            name="Surfer SEO", domain="surferseo.com", industry="SEO Software", description=""
        )
        question = profile.default_research_question()
        assert "Surfer SEO" in question
        assert "SEO Software" in question


class TestDiscoveredQuery:
    def _query(self, status: VisibilityStatus, position: int | None) -> DiscoveredQuery:
        return DiscoveredQuery(
            profile_uuid=uuid4(),
            run_uuid=uuid4(),
            query_text="best seo tools",
            estimated_search_volume=SearchVolume(100),
            competitive_difficulty=CompetitiveDifficulty(50),
            opportunity_score=OpportunityScore(0.5),
            visibility_status=status,
            visibility_position=position,
        )

    def test_a_visible_query_requires_a_position(self) -> None:
        with pytest.raises(DomainValidationError):
            self._query(VisibilityStatus.VISIBLE, None)

    @pytest.mark.parametrize("status", [VisibilityStatus.NOT_VISIBLE, VisibilityStatus.UNKNOWN])
    def test_a_non_visible_query_must_not_carry_a_position(self, status: VisibilityStatus) -> None:
        with pytest.raises(DomainValidationError):
            self._query(status, 3)

    def test_domain_visible_is_false_for_unknown(self) -> None:
        """'Unknown' must not be reported as visible, nor confused with measured absence."""
        assert self._query(VisibilityStatus.UNKNOWN, None).domain_visible is False
        assert self._query(VisibilityStatus.NOT_VISIBLE, None).domain_visible is False
        assert self._query(VisibilityStatus.VISIBLE, 1).domain_visible is True


class TestPipelineRun:
    def _run(self) -> PipelineRun:
        return PipelineRun(profile_uuid=uuid4(), correlation_id="abc", question="q?")

    def test_a_new_run_is_running_and_non_terminal(self) -> None:
        run = self._run()
        assert run.status is RunStatus.RUNNING
        assert not run.is_terminal
        assert run.duration_ms is None

    def test_completing_a_run_records_a_duration(self) -> None:
        run = self._run()
        run.mark_completed()
        assert run.status is RunStatus.COMPLETED
        assert run.is_terminal
        assert run.duration_ms is not None

    def test_a_terminal_run_cannot_transition_again(self) -> None:
        run = self._run()
        run.mark_completed()
        with pytest.raises(InvalidRunTransitionError):
            run.mark_failed("too late")

    def test_partial_records_its_degradation_reason(self) -> None:
        run = self._run()
        run.mark_partial(DegradationReason.ALL_RETRIEVALS_FAILED, "everything failed")
        assert run.status is RunStatus.PARTIAL
        assert run.degradation_reason is DegradationReason.ALL_RETRIEVALS_FAILED

    def test_counters_cannot_be_recorded_after_a_run_ends(self) -> None:
        run = self._run()
        run.mark_completed()
        with pytest.raises(InvalidRunTransitionError):
            run.record_plan(5)


class TestRecommendation:
    def test_a_recommendation_without_a_rationale_is_rejected(self) -> None:
        with pytest.raises(DomainValidationError):
            Recommendation(
                run_uuid=uuid4(),
                target_query_uuid=uuid4(),
                content_type=ContentType.BLOG_POST,
                title="A Title",
                rationale="   ",
                priority=Priority.HIGH,
            )

    def test_keywords_are_normalised_and_deduplicated(self) -> None:
        recommendation = Recommendation(
            run_uuid=uuid4(),
            target_query_uuid=uuid4(),
            content_type=ContentType.BLOG_POST,
            title="A Title",
            rationale="Closes a measured gap.",
            target_keywords=("SEO Tools", "seo tools", " ai seo "),
        )
        assert recommendation.target_keywords == ("seo tools", "ai seo")
