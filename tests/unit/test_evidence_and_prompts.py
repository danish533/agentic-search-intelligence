"""Evidence capture, and the date the prompts supply.

Both cover defects that every existing check passed over. The evidence column was declared,
documented as used, and never written - `nullable=True` meant mypy was satisfied and no test
asserted on it. The missing date produced a recommendation titled "...in 2023" during a 2026
run, which is wrong in the product's primary deliverable.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from sightline.agents.contracts.normalized import NormalizedObservation, NormalizedQuery
from sightline.agents.nodes.normalization import build_discovered_query
from sightline.agents.prompts import analysis, query_planner, report, today_line
from sightline.domain.entities.profile import Profile
from sightline.domain.value_objects.enums import RetrievalKind, VisibilityStatus


def observation(**overrides: object) -> NormalizedObservation:
    defaults: dict[str, object] = {
        "query_text": "best seo tools",
        "kind": RetrievalKind.ORGANIC_SERP,
        "source_endpoint": "/v3/serp/google/organic/live/advanced",
        "visibility_status": VisibilityStatus.VISIBLE,
        "position": 2,
        "cited_domains": ("ahrefs.com", "surferseo.com", "moz.com"),
        "competitor_positions": {"clearscope.io": 4},
        "search_volume": 8100,
        "competition_index": 45,
    }
    return NormalizedObservation(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestEvidenceCapture:
    def test_evidence_is_json_serialisable_primitives_only(self) -> None:
        """It has to survive a round trip through a JSONB column."""
        import json

        normalized = NormalizedQuery(query_text="best seo tools", observations=(observation(),))
        evidence = normalized.to_evidence()

        assert json.loads(json.dumps(evidence)) == evidence

    def test_evidence_records_what_the_verdict_was_derived_from(self) -> None:
        normalized = NormalizedQuery(query_text="best seo tools", observations=(observation(),))
        evidence = normalized.to_evidence()

        assert evidence["cited_domains"] == ["ahrefs.com", "surferseo.com", "moz.com"]
        assert len(evidence["observations"]) == 1
        recorded = evidence["observations"][0]
        assert recorded["kind"] == "organic_serp"
        assert recorded["position"] == 2
        assert recorded["competitor_positions"] == {"clearscope.io": 4}
        assert recorded["source_endpoint"].startswith("/v3/")

    def test_a_built_query_carries_its_evidence(self, profile: Profile) -> None:
        """Regression: the column existed and was documented, but nothing ever wrote to it."""
        normalized = NormalizedQuery(query_text="best seo tools", observations=(observation(),))
        query = build_discovered_query(normalized, profile=profile, run_uuid=uuid4())

        assert query.evidence is not None
        assert query.cited_domains() == ("ahrefs.com", "surferseo.com", "moz.com")

    def test_a_query_without_evidence_reports_no_domains_rather_than_failing(
        self, profile: Profile
    ) -> None:
        from sightline.domain.entities.discovered_query import DiscoveredQuery
        from sightline.domain.value_objects.scores import (
            CompetitiveDifficulty,
            OpportunityScore,
            SearchVolume,
        )

        query = DiscoveredQuery(
            profile_uuid=profile.uuid,
            run_uuid=uuid4(),
            query_text="q",
            estimated_search_volume=SearchVolume(1),
            competitive_difficulty=CompetitiveDifficulty(1),
            opportunity_score=OpportunityScore(0.5),
            visibility_status=VisibilityStatus.NOT_VISIBLE,
        )

        assert query.evidence is None
        assert query.cited_domains() == ()

    def test_evidence_supports_a_before_and_after_comparison(self, profile: Profile) -> None:
        """The whole reason the column exists: say what moved, not just that a score moved."""
        before = build_discovered_query(
            NormalizedQuery(
                query_text="q",
                observations=(observation(cited_domains=("a.com", "b.com")),),
            ),
            profile=profile,
            run_uuid=uuid4(),
        )
        after = build_discovered_query(
            NormalizedQuery(
                query_text="q",
                observations=(observation(cited_domains=("b.com", "c.com")),),
            ),
            profile=profile,
            run_uuid=uuid4(),
        )

        gained = set(after.cited_domains()) - set(before.cited_domains())
        lost = set(before.cited_domains()) - set(after.cited_domains())

        assert gained == {"c.com"}
        assert lost == {"a.com"}


class TestPromptsSupplyTheDate:
    def test_the_date_line_states_the_current_year(self) -> None:
        assert str(datetime.now(UTC).year) in today_line()

    @pytest.fixture
    def profile(self) -> Profile:
        return Profile(
            name="Surfer SEO",
            domain="surferseo.com",
            industry="SEO Software",
            description="AI-powered SEO content optimization tool",
        )

    def test_the_planner_prompt_carries_the_date(self, profile: Profile) -> None:
        prompt = query_planner.build_user_prompt(profile, "How visible is it?", 6)
        assert str(datetime.now(UTC).year) in prompt

    def test_the_analysis_prompt_carries_the_date(self, profile: Profile) -> None:
        prompt = analysis.build_user_prompt(profile, "How visible?", [])
        assert str(datetime.now(UTC).year) in prompt

    def test_the_report_prompt_carries_the_date(self, profile: Profile) -> None:
        prompt = report.build_user_prompt(profile, "How visible?", None, [], None)
        assert str(datetime.now(UTC).year) in prompt

    def test_no_prompt_leaves_the_model_to_guess_the_year(self, profile: Profile) -> None:
        """Without a date, the model falls back to its training distribution and writes a
        stale year into content titles - which is exactly what happened."""
        prompts = [
            query_planner.build_user_prompt(profile, "q", 6),
            analysis.build_user_prompt(profile, "q", []),
            report.build_user_prompt(profile, "q", None, [], None),
        ]
        for prompt in prompts:
            assert re.search(r"Today's date is \d{2} \w+ \d{4}", prompt)

    def test_the_analysis_prompt_forbids_stale_years_in_titles(self) -> None:
        assert "year in a content title" in analysis.SYSTEM_PROMPT
