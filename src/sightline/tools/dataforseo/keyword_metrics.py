"""Tool: search volume and competition metrics."""

from __future__ import annotations

from sightline.application.ports.search_data_provider import RawApiResult, SearchDataProvider
from sightline.domain.value_objects.enums import RetrievalKind
from sightline.tools.definition import Tool
from sightline.tools.schemas.dataforseo import FetchKeywordMetricsArgs


async def _execute(provider: SearchDataProvider, args: FetchKeywordMetricsArgs) -> RawApiResult:
    return await provider.fetch_keyword_metrics(
        keywords=args.keywords,
        location_code=args.location_code,
        language_code=args.language_code,
    )


FETCH_KEYWORD_METRICS = Tool(
    name="fetch_keyword_metrics",
    description=(
        "Retrieve monthly search volume and a 0-100 competition index for a batch of "
        "keywords. Use this to size how much traffic a keyword is worth and how hard it is "
        "to compete for, which together determine whether a visibility gap is worth closing. "
        "Batch all keywords of interest into one call rather than calling per keyword."
    ),
    kind=RetrievalKind.KEYWORD_METRICS,
    args_schema=FetchKeywordMetricsArgs,
    executor=_execute,
)
