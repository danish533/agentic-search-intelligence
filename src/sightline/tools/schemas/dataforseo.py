"""Tool argument schemas.

Spec S3.3 requires tools "defined using proper schemas ... with accurate types, descriptions,
and required fields - not just free-text prompts asking the model to call the API". These
Pydantic models are that definition, and they serve two roles at once:

1. They are rendered into the JSON Schema the model is shown, so ``Field(description=...)``
   is not documentation for humans - it is the instruction the model actually reads when
   deciding what to pass.
2. They are the validator the proposed arguments are checked against before any HTTP request
   is made. The schema the model sees and the schema its output is judged by are the same
   object, so the two cannot drift apart.

Every model sets ``extra="forbid"``. A hallucinated argument is a signal that the model has
misunderstood the tool, and silently dropping it would hide that; rejecting it surfaces the
problem and gives the retrieval node something concrete to correct.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: DataForSEO location code for the United States. Used as the default throughout because the
#: assessment's example questions are US-market, and requiring the model to supply a numeric
#: geo code it cannot know would guarantee malformed calls.
DEFAULT_LOCATION_CODE = 2840
DEFAULT_LANGUAGE_CODE = "en"

MIN_KEYWORD_LENGTH = 2
MAX_KEYWORD_LENGTH = 200
MAX_KEYWORD_BATCH = 20

Keyword = Annotated[
    str,
    Field(
        min_length=MIN_KEYWORD_LENGTH,
        max_length=MAX_KEYWORD_LENGTH,
        description="A single search query exactly as a user would type it into Google. "
        "Lower-case, no quotes, no boolean operators, no site: filters.",
    ),
]

LocationCode = Annotated[
    int,
    Field(
        default=DEFAULT_LOCATION_CODE,
        ge=1,
        description="DataForSEO numeric location code identifying the search market. "
        f"Use {DEFAULT_LOCATION_CODE} (United States) unless the research question names a "
        "different country.",
    ),
]

LanguageCode = Annotated[
    str,
    Field(
        default=DEFAULT_LANGUAGE_CODE,
        min_length=2,
        max_length=5,
        description="ISO 639-1 language code for the search, such as 'en', 'de' or 'es'.",
    ),
]


class _ToolArgs(BaseModel):
    """Shared strictness for every tool argument model."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True,
    )


class FetchOrganicSerpArgs(_ToolArgs):
    """Arguments for retrieving a live organic Google SERP."""

    keyword: Keyword
    location_code: LocationCode = DEFAULT_LOCATION_CODE
    language_code: LanguageCode = DEFAULT_LANGUAGE_CODE
    depth: int = Field(
        default=10,
        ge=1,
        le=100,
        description="How many ranked results to retrieve. 10 covers page one, which is what "
        "matters for visibility; go higher only when checking deep rankings.",
    )

    @field_validator("keyword")
    @classmethod
    def reject_search_operators(cls, value: str) -> str:
        """Reject operator syntax, which DataForSEO treats literally and which skews results."""
        if value.startswith(("site:", "inurl:", "intitle:")):
            raise ValueError(
                "keyword must be a plain search phrase, not a search operator expression"
            )
        return value


class FetchAiOverviewArgs(_ToolArgs):
    """Arguments for retrieving Google's AI Overview block for a keyword."""

    keyword: Keyword
    location_code: LocationCode = DEFAULT_LOCATION_CODE
    language_code: LanguageCode = DEFAULT_LANGUAGE_CODE


class FetchLlmVisibilityArgs(_ToolArgs):
    """Arguments for asking a named LLM a question and capturing which domains it cites."""

    prompt: str = Field(
        min_length=5,
        max_length=1000,
        description="The question to put to the LLM, phrased the way a real buyer would ask "
        "it - for example 'what is the best project management software for agencies?'. "
        "This is a natural-language question, not a keyword.",
    )
    llm_name: Literal["chat_gpt", "gemini", "perplexity"] = Field(
        default="chat_gpt",
        description="Which assistant to query. Use 'chat_gpt' unless the research question "
        "asks specifically about Gemini or Perplexity visibility.",
    )


class FetchKeywordMetricsArgs(_ToolArgs):
    """Arguments for retrieving search volume and competition metrics for a keyword set."""

    keywords: list[Keyword] = Field(
        min_length=1,
        max_length=MAX_KEYWORD_BATCH,
        description="The keywords to size, batched in one call. Prefer a single batched call "
        f"over many single-keyword calls; up to {MAX_KEYWORD_BATCH} per request.",
    )
    location_code: LocationCode = DEFAULT_LOCATION_CODE
    language_code: LanguageCode = DEFAULT_LANGUAGE_CODE

    @field_validator("keywords")
    @classmethod
    def deduplicate(cls, values: list[str]) -> list[str]:
        """Drop duplicates case-insensitively: they cost credits and add no information."""
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            folded = value.casefold()
            if folded not in seen:
                seen.add(folded)
                unique.append(value)
        return unique
