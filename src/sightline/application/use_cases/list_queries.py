"""List the sub-queries from a profile's most recent run (spec S4.2)."""

from __future__ import annotations

from uuid import UUID

from sightline.application.dto.pagination import Page, PageRequest
from sightline.application.dto.query_filters import QueryFilters
from sightline.application.ports.unit_of_work import UnitOfWorkFactory
from sightline.domain.entities.discovered_query import DiscoveredQuery


class ListQueriesUseCase:
    """Filtered, paginated queries for a profile's latest run."""

    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def execute(
        self,
        profile_uuid: UUID,
        *,
        filters: QueryFilters,
        page_request: PageRequest,
    ) -> Page[DiscoveredQuery]:
        """Raise :class:`ProfileNotFoundError` when the profile does not exist.

        A profile that exists but has never been run returns an empty page rather than a 404:
        the resource is real, it simply has no queries yet.
        """
        async with self._unit_of_work() as uow:
            await uow.profiles.get(profile_uuid)
            latest = await uow.runs.find_latest_for_profile(profile_uuid)
            if latest is None:
                return Page(
                    items=[],
                    total_items=0,
                    page=page_request.page,
                    per_page=page_request.per_page,
                )
            return await uow.queries.list_for_run(
                latest.uuid, filters=filters, page_request=page_request
            )
