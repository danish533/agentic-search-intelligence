"""DataForSEO implementation of the ``SearchDataProvider`` port.

Thin by design. Its whole job is to translate a typed capability call into the task dictionary
DataForSEO expects and hand it to the client - which is the layer that owns classification,
retries and metrics. Keeping request shaping here and resilience there means neither can be
changed by accident while editing the other.

The returned payload is deliberately **unparsed**. Interpreting it belongs to the
Normalization agent (CLAUDE.md R1); a provider that also parsed would put retrieval and
extraction in the same component, which is the single-responsibility violation this assessment
grades first.
"""

from __future__ import annotations

from collections.abc import Sequence

from sightline.application.ports.search_data_provider import RawApiResult
from sightline.config.settings import DataForSEOSettings
from sightline.infrastructure.dataforseo.client import DataForSEOClient
from sightline.infrastructure.dataforseo.endpoints import (
    AI_OVERVIEW,
    KEYWORD_METRICS,
    ORGANIC_SERP,
    llm_visibility_endpoint,
)


class DataForSEOProvider:
    """Search and AI-visibility retrieval backed by DataForSEO."""

    def __init__(self, *, client: DataForSEOClient, settings: DataForSEOSettings) -> None:
        self._client = client
        self._settings = settings

    @property
    def mode_label(self) -> str:
        """Transport mode in use - reported in the run summary and README-visible."""
        return self._client.mode_label

    async def fetch_organic_serp(
        self,
        *,
        keyword: str,
        location_code: int,
        language_code: str,
        depth: int,
    ) -> RawApiResult:
        return await self._client.execute(
            ORGANIC_SERP,
            {
                "keyword": keyword,
                "location_code": location_code,
                "language_code": language_code,
                "depth": depth,
                "device": "desktop",
                "os": "windows",
            },
        )

    async def fetch_ai_overview(
        self,
        *,
        keyword: str,
        location_code: int,
        language_code: str,
    ) -> RawApiResult:
        return await self._client.execute(
            AI_OVERVIEW,
            {
                "keyword": keyword,
                "location_code": location_code,
                "language_code": language_code,
                # Instructs DataForSEO to resolve the AI Overview block, which is populated
                # asynchronously and is absent from a default organic request.
                "load_async_ai_overview": True,
            },
        )

    async def fetch_llm_visibility(self, *, prompt: str, llm_name: str) -> RawApiResult:
        return await self._client.execute(
            llm_visibility_endpoint(llm_name),
            {
                "user_prompt": prompt,
                # Web search on: without it the model answers from parametric memory, which
                # measures training-data recall rather than present-day AI visibility.
                "web_search": True,
            },
        )

    async def fetch_keyword_metrics(
        self,
        *,
        keywords: Sequence[str],
        location_code: int,
        language_code: str,
    ) -> RawApiResult:
        return await self._client.execute(
            KEYWORD_METRICS,
            {
                "keywords": list(keywords),
                "location_code": location_code,
                "language_code": language_code,
                "search_partners": False,
            },
        )
