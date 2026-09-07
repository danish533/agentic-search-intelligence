"""Tool: live organic Google SERP."""

from __future__ import annotations

from sightline.application.ports.search_data_provider import RawApiResult, SearchDataProvider
from sightline.domain.value_objects.enums import RetrievalKind
from sightline.tools.definition import Tool
from sightline.tools.schemas.dataforseo import FetchOrganicSerpArgs


async def _execute(provider: SearchDataProvider, args: FetchOrganicSerpArgs) -> RawApiResult:
    return await provider.fetch_organic_serp(
        keyword=args.keyword,
        location_code=args.location_code,
        language_code=args.language_code,
        depth=args.depth,
    )


FETCH_ORGANIC_SERP = Tool(
    name="fetch_organic_serp",
    description=(
        "Retrieve the live organic Google search results for one keyword, as a ranked list "
        "of domains with titles, URLs and descriptions. Use this to determine whether a "
        "domain ranks for a keyword and at what position. This returns classic blue-link "
        "results only - it does not include AI Overview content."
    ),
    kind=RetrievalKind.ORGANIC_SERP,
    args_schema=FetchOrganicSerpArgs,
    executor=_execute,
)
