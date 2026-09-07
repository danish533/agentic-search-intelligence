"""Tool: Google AI Overview block."""

from __future__ import annotations

from sightline.application.ports.search_data_provider import RawApiResult, SearchDataProvider
from sightline.domain.value_objects.enums import RetrievalKind
from sightline.tools.definition import Tool
from sightline.tools.schemas.dataforseo import FetchAiOverviewArgs


async def _execute(provider: SearchDataProvider, args: FetchAiOverviewArgs) -> RawApiResult:
    return await provider.fetch_ai_overview(
        keyword=args.keyword,
        location_code=args.location_code,
        language_code=args.language_code,
    )


FETCH_AI_OVERVIEW = Tool(
    name="fetch_ai_overview",
    description=(
        "Retrieve Google's AI Overview for one keyword: the AI-generated answer shown above "
        "the organic results, together with the sources it cites. Use this to determine "
        "whether a domain is cited by Google's own AI answer, which is a different and often "
        "more valuable form of visibility than ranking organically. Not every keyword "
        "triggers an AI Overview; an empty result is a meaningful finding, not an error."
    ),
    kind=RetrievalKind.AI_OVERVIEW,
    args_schema=FetchAiOverviewArgs,
    executor=_execute,
)
