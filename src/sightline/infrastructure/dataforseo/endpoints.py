"""DataForSEO endpoint catalogue.

One :class:`Endpoint` per *logical* API call, as spec S3.4 requires. Each is paired with
exactly one method on the ``SearchDataProvider`` port and exactly one LLM-facing tool, so the
three lists stay in one-to-one correspondence and a reviewer can trace any tool the model
calls to the HTTP request it produces.

Note that ``ORGANIC_SERP`` and ``AI_OVERVIEW`` share a path. They remain distinct entries
because they are distinct logical operations with different request parameters and different
result shapes: "one tool per logical API call" is about operations, not about unique URLs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sightline.domain.value_objects.enums import RetrievalKind


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A single DataForSEO operation."""

    kind: RetrievalKind
    path: str
    summary: str

    def __str__(self) -> str:
        return self.path


ORGANIC_SERP: Final = Endpoint(
    kind=RetrievalKind.ORGANIC_SERP,
    path="/v3/serp/google/organic/live/advanced",
    summary="Live organic Google SERP for a keyword, with ranked result items.",
)

AI_OVERVIEW: Final = Endpoint(
    kind=RetrievalKind.AI_OVERVIEW,
    path="/v3/serp/google/organic/live/advanced",
    summary="Google AI Overview block for a keyword, including its cited references.",
)

LLM_VISIBILITY: Final = Endpoint(
    kind=RetrievalKind.LLM_VISIBILITY,
    path="/v3/ai_optimization/chat_gpt/llm_responses/live",
    summary="How an LLM answers a prompt, and which domains it cites.",
)

KEYWORD_METRICS: Final = Endpoint(
    kind=RetrievalKind.KEYWORD_METRICS,
    path="/v3/keywords_data/google_ads/search_volume/live",
    summary="Monthly search volume and competition index for a set of keywords.",
)

#: Path template per supported LLM, for the AI-visibility endpoint.
_LLM_PATHS: Final[dict[str, str]] = {
    "chat_gpt": "/v3/ai_optimization/chat_gpt/llm_responses/live",
    "gemini": "/v3/ai_optimization/gemini/llm_responses/live",
    "perplexity": "/v3/ai_optimization/perplexity/llm_responses/live",
}

DEFAULT_LLM_NAME: Final = "chat_gpt"

ALL_ENDPOINTS: Final[tuple[Endpoint, ...]] = (
    ORGANIC_SERP,
    AI_OVERVIEW,
    LLM_VISIBILITY,
    KEYWORD_METRICS,
)


def supported_llm_names() -> tuple[str, ...]:
    """LLM identifiers accepted by :func:`llm_visibility_endpoint`."""
    return tuple(_LLM_PATHS)


def llm_visibility_endpoint(llm_name: str) -> Endpoint:
    """Resolve the AI-visibility endpoint for a named LLM.

    Falls back to the default rather than raising: an unrecognised model name from the LLM's
    tool call is a validation concern handled by the tool layer, and this catalogue should not
    be a second place that can reject it.
    """
    path = _LLM_PATHS.get(llm_name.strip().lower(), _LLM_PATHS[DEFAULT_LLM_NAME])
    return Endpoint(
        kind=RetrievalKind.LLM_VISIBILITY,
        path=path,
        summary=LLM_VISIBILITY.summary,
    )
