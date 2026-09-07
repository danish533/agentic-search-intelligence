"""Brands whose names are shorter than a usable search query.

The fallback planner emitted a bare ``query_text = brand`` template, which fails the contract's
three-character minimum. HP, BP, GE, LG and 3M therefore crashed the planner - and it is the
*fallback* planner, the safety net that exists for when the LLM is unavailable, so those brands
had no degradation protection at all.

344 tests missed it because every fixture used a long brand name. These are parametrised over
real short names so the gap cannot reopen.
"""

from __future__ import annotations

import pytest

from sightline.agents.contracts.plan import MIN_QUERY_LENGTH
from sightline.agents.nodes.planner_fallback import build_fallback_plan
from sightline.domain.entities.profile import Profile
from sightline.domain.value_objects.enums import QueryIntent

#: Real brands, all among the largest advertisers there are.
SHORT_BRANDS = ["3M", "HP", "BP", "GE", "LG", "X", "Ai"]
NORMAL_BRANDS = ["IBM", "Nike", "Surfer SEO", "MarketMuse"]


def profile_for(name: str, industry: str = "SEO Software") -> Profile:
    slug = "".join(character for character in name.lower() if character.isalnum()) or "brand"
    return Profile(name=name, domain=f"{slug}.com", industry=industry, description="")


class TestShortBrandNames:
    @pytest.mark.parametrize("brand", SHORT_BRANDS + NORMAL_BRANDS)
    def test_the_fallback_planner_produces_a_usable_plan(self, brand: str) -> None:
        """Regression: a two-character brand raised ValidationError out of the node."""
        plan = build_fallback_plan(profile_for(brand), limit=6)

        assert plan.is_usable
        assert len(plan.sub_queries) >= 1

    @pytest.mark.parametrize("brand", SHORT_BRANDS + NORMAL_BRANDS)
    def test_every_generated_query_satisfies_the_contract(self, brand: str) -> None:
        for sub_query in build_fallback_plan(profile_for(brand), limit=6).sub_queries:
            assert len(sub_query.query_text) >= MIN_QUERY_LENGTH
            assert sub_query.retrieval_kinds

    @pytest.mark.parametrize("brand", SHORT_BRANDS)
    def test_a_short_brand_gets_a_qualified_defence_query(self, brand: str) -> None:
        """'HP' alone matches almost anything; 'HP SEO Software' is a real search."""
        plan = build_fallback_plan(profile_for(brand), limit=6)
        navigational = [
            sub_query.query_text
            for sub_query in plan.sub_queries
            if sub_query.intent is QueryIntent.NAVIGATIONAL
        ]

        assert navigational
        assert navigational[0] != brand
        assert brand in navigational[0]

    @pytest.mark.parametrize("brand", NORMAL_BRANDS)
    def test_a_normal_brand_keeps_its_bare_defence_query(self, brand: str) -> None:
        """Qualification applies only where it is needed."""
        plan = build_fallback_plan(profile_for(brand), limit=6)
        navigational = [
            sub_query.query_text
            for sub_query in plan.sub_queries
            if sub_query.intent is QueryIntent.NAVIGATIONAL
        ]

        assert navigational == [brand]

    @pytest.mark.parametrize("industry", ["", "   ", "SEO Software", "3D"])
    def test_a_missing_or_short_industry_still_yields_a_plan(self, industry: str) -> None:
        plan = build_fallback_plan(profile_for("HP", industry), limit=6)

        assert plan.is_usable
        for sub_query in plan.sub_queries:
            assert len(sub_query.query_text) >= MIN_QUERY_LENGTH

    def test_the_fallback_planner_never_raises(self) -> None:
        """It is the safety net. If it can fail, there is no safety net (CLAUDE.md A8, R5)."""
        awkward = ["X", "3M", "A B", "Ω", "a", "L&G", "AT&T", "23andMe"]

        for brand in awkward:
            plan = build_fallback_plan(profile_for(brand), limit=6)
            assert plan.is_usable, f"no usable plan for brand {brand!r}"

    @pytest.mark.parametrize("brand", SHORT_BRANDS)
    def test_competitor_comparisons_are_generated_for_short_brands_too(self, brand: str) -> None:
        profile = Profile(
            name=brand,
            domain=f"{brand.lower()}.com",
            industry="SEO Software",
            description="",
            competitors=("clearscope.io", "frase.io"),
        )
        queries = [
            sub_query.query_text for sub_query in build_fallback_plan(profile, limit=8).sub_queries
        ]

        assert any("vs" in query for query in queries)
