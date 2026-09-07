"""Tool: LLM answer and its citations."""

from __future__ import annotations

from sightline.application.ports.search_data_provider import RawApiResult, SearchDataProvider
from sightline.domain.value_objects.enums import RetrievalKind
from sightline.tools.definition import Tool
from sightline.tools.schemas.dataforseo import FetchLlmVisibilityArgs


async def _execute(provider: SearchDataProvider, args: FetchLlmVisibilityArgs) -> RawApiResult:
    return await provider.fetch_llm_visibility(prompt=args.prompt, llm_name=args.llm_name)


FETCH_LLM_VISIBILITY = Tool(
    name="fetch_llm_visibility",
    description=(
        "Ask a named AI assistant (ChatGPT, Gemini or Perplexity) a natural-language question "
        "and capture both its answer and the domains it cites. Use this to measure whether a "
        "brand is recommended in AI-generated answers, which is where buyer research "
        "increasingly happens. Pass a full question a real buyer would ask, not a keyword."
    ),
    kind=RetrievalKind.LLM_VISIBILITY,
    args_schema=FetchLlmVisibilityArgs,
    executor=_execute,
)
