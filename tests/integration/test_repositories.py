"""Persistence against a real PostgreSQL instance.

These run only when a migrated database is reachable (``make db && make migrate``); they skip
otherwise so a fresh clone can still run ``make test``.

They target PostgreSQL specifically, not a stand-in. The flush-ordering defect that broke every
write path was invisible to every non-database test, and the check constraints asserted below
exist only in PostgreSQL.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from sightline.application.dto.pagination import PageRequest
from sightline.application.dto.query_filters import QueryFilters
from sightline.application.ports.unit_of_work import UnitOfWorkFactory
from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.entities.insight import Insight
from sightline.domain.entities.pipeline_run import PipelineRun
from sightline.domain.entities.profile import Profile
from sightline.domain.entities.recommendation import Recommendation
from sightline.domain.errors import DiscoveredQueryNotFoundError, ProfileNotFoundError
from sightline.domain.services.opportunity_scoring import calculate_opportunity_score
from sightline.domain.value_objects.enums import (
    ContentType,
    Priority,
    RunStatus,
    VisibilityStatus,
)
from sightline.domain.value_objects.scores import (
    CompetitiveDifficulty,
    OpportunityScore,
    RelevanceScore,
    SearchVolume,
)

pytestmark = pytest.mark.integration

#: Parameterised throughout - no value is ever interpolated into the statement text.
INSERT_QUERY_SQL = text(
    "INSERT INTO discovered_queries (uuid, profile_uuid, run_uuid, query_text, "
    "estimated_search_volume, competitive_difficulty, opportunity_score, "
    "visibility_status, discovered_at) VALUES (gen_random_uuid(), :p, :r, 'x', "
    ":estimated_search_volume, :competitive_difficulty, :opportunity_score, "
    "'not_visible', now())"
)

CHILD_ROW_COUNTS_SQL = text(
    "SELECT (SELECT count(*) FROM pipeline_runs), "
    "(SELECT count(*) FROM discovered_queries), "
    "(SELECT count(*) FROM insights), "
    "(SELECT count(*) FROM recommendations)"
)

SPECS = [
    ("best seo content tool", 8100, 45, VisibilityStatus.VISIBLE, 2),
    ("ai content optimization software", 2400, 62, VisibilityStatus.NOT_VISIBLE, None),
    ("surfer seo alternatives", 5400, 38, VisibilityStatus.VISIBLE, 1),
    ("content brief generator", 880, 25, VisibilityStatus.NOT_VISIBLE, None),
    ("serp analysis tool", 1600, 71, VisibilityStatus.UNKNOWN, None),
]


def build_queries(profile: Profile, run: PipelineRun) -> list[DiscoveredQuery]:
    queries = []
    for text_, volume, difficulty, status, position in SPECS:
        search_volume = SearchVolume(volume)
        competitive = CompetitiveDifficulty(difficulty)
        queries.append(
            DiscoveredQuery(
                profile_uuid=profile.uuid,
                run_uuid=run.uuid,
                query_text=text_,
                estimated_search_volume=search_volume,
                competitive_difficulty=competitive,
                opportunity_score=calculate_opportunity_score(
                    search_volume=search_volume,
                    competitive_difficulty=competitive,
                    visibility_status=status,
                    visibility_position=position,
                ),
                visibility_status=status,
                visibility_position=position,
            )
        )
    return queries


@pytest.fixture
async def seeded(
    unit_of_work_factory: UnitOfWorkFactory, profile: Profile
) -> tuple[Profile, PipelineRun, list[DiscoveredQuery]]:
    run = PipelineRun(
        profile_uuid=profile.uuid, correlation_id=uuid4().hex, question="How visible?"
    )
    queries = build_queries(profile, run)
    insights = [
        Insight(
            run_uuid=run.uuid,
            headline="Absent from AI answers",
            detail="Not cited.",
            relevance_score=RelevanceScore(0.92),
            supporting_query_uuids=(queries[1].uuid,),
        ),
        Insight(
            run_uuid=run.uuid,
            headline="Ranked but not cited",
            detail="Rank does not imply citation.",
            relevance_score=RelevanceScore(0.61),
        ),
    ]
    recommendations = [
        Recommendation(
            run_uuid=run.uuid,
            target_query_uuid=queries[1].uuid,
            content_type=ContentType.BLOG_POST,
            title="Guide",
            rationale="Closes the gap.",
            priority=Priority.LOW,
        ),
        Recommendation(
            run_uuid=run.uuid,
            target_query_uuid=queries[3].uuid,
            content_type=ContentType.LANDING_PAGE,
            title="Tool page",
            rationale="Unowned query.",
            priority=Priority.HIGH,
        ),
    ]

    run.record_plan(5)
    run.record_retrieval_outcome(5)
    run.record_normalized(5)
    run.add_tokens(4820)
    run.attach_output(report={"headline": "3 gaps"}, metrics={"node_count": 8})
    run.mark_completed()

    async with unit_of_work_factory() as uow:
        await uow.profiles.add(profile)
        await uow.runs.add(run)
        await uow.queries.add_many(queries)
        await uow.insights.add_many(insights)
        await uow.recommendations.add_many(recommendations)
        await uow.commit()

    return profile, run, queries


class TestTransactionalWrites:
    async def test_a_whole_run_persists_in_one_transaction(
        self, unit_of_work_factory: UnitOfWorkFactory, seeded: tuple
    ) -> None:
        """Regression: without mapper relationships, children flushed before their parent."""
        profile, run, queries = seeded
        async with unit_of_work_factory() as uow:
            assert (await uow.profiles.get(profile.uuid)).domain == "surferseo.com"
            assert (await uow.runs.get(run.uuid)).status is RunStatus.COMPLETED
            assert (await uow.queries.get(queries[0].uuid)).query_text == queries[0].query_text

    async def test_uncommitted_work_is_discarded_on_an_exception(
        self, unit_of_work_factory: UnitOfWorkFactory
    ) -> None:
        ghost = Profile(name="Ghost", domain="ghost.example", industry="x", description="")

        with pytest.raises(RuntimeError):
            async with unit_of_work_factory() as uow:
                await uow.profiles.add(ghost)
                raise RuntimeError("boom, halfway through")

        async with unit_of_work_factory() as uow:
            with pytest.raises(ProfileNotFoundError):
                await uow.profiles.get(ghost.uuid)

    async def test_forgetting_to_commit_is_not_the_same_as_committing(
        self, unit_of_work_factory: UnitOfWorkFactory
    ) -> None:
        forgotten = Profile(name="Forgotten", domain="forgot.example", industry="", description="")
        async with unit_of_work_factory() as uow:
            await uow.profiles.add(forgotten)

        async with unit_of_work_factory() as uow:
            with pytest.raises(ProfileNotFoundError):
                await uow.profiles.get(forgotten.uuid)


class TestReads:
    async def test_domain_lookup_normalises_its_input(
        self, unit_of_work_factory: UnitOfWorkFactory, seeded: tuple
    ) -> None:
        async with unit_of_work_factory() as uow:
            found = await uow.profiles.find_by_domain("HTTPS://WWW.SurferSEO.com/pricing/")
        assert found is not None
        assert found.domain == "surferseo.com"

    async def test_run_statistics(
        self, unit_of_work_factory: UnitOfWorkFactory, seeded: tuple
    ) -> None:
        profile, run, _ = seeded
        async with unit_of_work_factory() as uow:
            assert await uow.runs.count_for_profile(profile.uuid) == 1
            latest = await uow.runs.find_latest_for_profile(profile.uuid)
            average = await uow.queries.average_opportunity_score(profile.uuid)

        assert latest is not None
        assert latest.uuid == run.uuid
        assert latest.report == {"headline": "3 gaps"}
        assert average is not None and 0.0 <= average <= 1.0

    async def test_average_is_none_when_nothing_has_been_measured(
        self, unit_of_work_factory: UnitOfWorkFactory, profile: Profile
    ) -> None:
        """None means 'no data', which must not be flattened into a mean of 0.0."""
        async with unit_of_work_factory() as uow:
            await uow.profiles.add(profile)
            await uow.commit()
            assert await uow.queries.average_opportunity_score(profile.uuid) is None

    async def test_queries_are_ordered_by_opportunity_descending(
        self, unit_of_work_factory: UnitOfWorkFactory, seeded: tuple
    ) -> None:
        _, run, _ = seeded
        async with unit_of_work_factory() as uow:
            page = await uow.queries.list_for_run(
                run.uuid, filters=QueryFilters(), page_request=PageRequest(per_page=20)
            )

        scores = [query.opportunity_score.value for query in page.items]
        assert scores == sorted(scores, reverse=True)
        assert page.total_items == len(SPECS)

    @pytest.mark.parametrize(
        ("filters", "expected"),
        [
            (QueryFilters(), 5),
            (QueryFilters(min_score=0.6), 2),
            (QueryFilters(visibility_status=VisibilityStatus.NOT_VISIBLE), 2),
            (QueryFilters(visibility_status=VisibilityStatus.VISIBLE), 2),
            (QueryFilters(visibility_status=VisibilityStatus.UNKNOWN), 1),
            (
                QueryFilters(min_score=0.6, visibility_status=VisibilityStatus.NOT_VISIBLE),
                2,
            ),
        ],
    )
    async def test_filters(
        self,
        unit_of_work_factory: UnitOfWorkFactory,
        seeded: tuple,
        filters: QueryFilters,
        expected: int,
    ) -> None:
        _, run, _ = seeded
        async with unit_of_work_factory() as uow:
            page = await uow.queries.list_for_run(
                run.uuid, filters=filters, page_request=PageRequest(per_page=20)
            )
        assert page.total_items == expected

    async def test_pagination_has_no_gaps_or_duplicates(
        self, unit_of_work_factory: UnitOfWorkFactory, seeded: tuple
    ) -> None:
        """A stable tiebreak is what prevents equal scores reshuffling between pages."""
        _, run, _ = seeded
        seen: list[str] = []
        async with unit_of_work_factory() as uow:
            for page_number in (1, 2, 3):
                page = await uow.queries.list_for_run(
                    run.uuid,
                    filters=QueryFilters(),
                    page_request=PageRequest(page=page_number, per_page=2),
                )
                seen.extend(query.query_text for query in page.items)

        assert len(seen) == len(set(seen)) == len(SPECS)

    async def test_recommendations_come_back_high_priority_first(
        self, unit_of_work_factory: UnitOfWorkFactory, seeded: tuple
    ) -> None:
        _, run, _ = seeded
        async with unit_of_work_factory() as uow:
            recommendations = await uow.recommendations.list_for_run(run.uuid)
        assert [r.priority for r in recommendations] == [Priority.HIGH, Priority.LOW]

    async def test_insights_come_back_by_relevance_descending(
        self, unit_of_work_factory: UnitOfWorkFactory, seeded: tuple
    ) -> None:
        _, run, _ = seeded
        async with unit_of_work_factory() as uow:
            insights = await uow.insights.list_for_run(run.uuid)
        assert [i.relevance_score.value for i in insights] == [0.92, 0.61]


class TestDatabaseInvariants:
    async def test_the_database_rejects_a_position_without_visibility(
        self, database_engine: AsyncEngine, seeded: tuple
    ) -> None:
        """Enforced in the schema too, so a backfill or manual fix cannot violate it."""
        profile, run, _ = seeded
        with pytest.raises(IntegrityError):
            async with database_engine.begin() as connection:
                await connection.execute(
                    text(
                        "INSERT INTO discovered_queries (uuid, profile_uuid, run_uuid, "
                        "query_text, estimated_search_volume, competitive_difficulty, "
                        "opportunity_score, visibility_status, visibility_position, "
                        "discovered_at) VALUES (gen_random_uuid(), :p, :r, 'smuggled', 100, "
                        "50, 0.5, 'not_visible', 3, now())"
                    ),
                    {"p": profile.uuid, "r": run.uuid},
                )

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("opportunity_score", 2.0),
            ("opportunity_score", -0.5),
            ("competitive_difficulty", 150),
            ("estimated_search_volume", -1),
        ],
    )
    async def test_the_database_rejects_out_of_range_measurements(
        self, database_engine: AsyncEngine, seeded: tuple, column: str, value: float
    ) -> None:
        profile, run, _ = seeded
        values: dict[str, object] = {
            "estimated_search_volume": 100,
            "competitive_difficulty": 50,
            "opportunity_score": 0.5,
            column: value,
            "p": profile.uuid,
            "r": run.uuid,
        }
        with pytest.raises(IntegrityError):
            async with database_engine.begin() as connection:
                await connection.execute(INSERT_QUERY_SQL, values)

    async def test_deleting_a_profile_cascades(
        self, database_engine: AsyncEngine, seeded: tuple
    ) -> None:
        profile, _, _ = seeded
        async with database_engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM profiles WHERE uuid = :u"), {"u": profile.uuid}
            )
            remaining = (await connection.execute(CHILD_ROW_COUNTS_SQL)).one()

        assert tuple(remaining) == (0, 0, 0, 0)


class TestRecheckWrite:
    async def test_saving_a_query_overwrites_its_measurements_in_place(
        self, unit_of_work_factory: UnitOfWorkFactory, seeded: tuple
    ) -> None:
        import dataclasses

        _, _, queries = seeded
        original = queries[0]
        updated = dataclasses.replace(
            original,
            opportunity_score=OpportunityScore(0.1234),
            visibility_status=VisibilityStatus.VISIBLE,
            visibility_position=1,
        )

        async with unit_of_work_factory() as uow:
            await uow.queries.save(updated)
            await uow.commit()

        async with unit_of_work_factory() as uow:
            reloaded = await uow.queries.get(original.uuid)

        assert reloaded.uuid == original.uuid
        assert reloaded.opportunity_score.value == 0.1234
        assert reloaded.visibility_position == 1

    async def test_saving_an_unknown_query_raises(
        self, unit_of_work_factory: UnitOfWorkFactory, seeded: tuple
    ) -> None:
        async with unit_of_work_factory() as uow:
            with pytest.raises(DiscoveredQueryNotFoundError):
                await uow.queries.get(uuid4())
