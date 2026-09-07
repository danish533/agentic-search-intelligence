"""Query Planner output contract.

The planner's sole responsibility is to decide *what to retrieve* (spec S3.2). This is the
typed shape it must produce, and it is also the schema the LLM is constrained to - so a plan
that reaches the retrieval stage is structurally valid by construction, and a planner that
cannot produce one routes to the fallback rather than emitting something half-formed.

Note what is absent: no URLs, no endpoints, no tool names. The planner decides *which
capabilities* a sub-query needs; choosing the concrete tool and its arguments belongs to the
retrieval agent. Keeping that line sharp is what stops the planner from quietly becoming a
do-everything agent.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sightline.domain.value_objects.enums import QueryIntent, RetrievalKind

MAX_SUB_QUERIES = 12


class PlannedRetrieval(BaseModel):
    """One sub-query the planner wants measured."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    query_text: str = Field(
        min_length=3,
        max_length=200,
        description="The search query to measure, phrased exactly as a real user would type "
        "or ask it. Lower-case, no quotes, no operators.",
    )
    intent: QueryIntent = Field(
        description="What the searcher is trying to accomplish with this query."
    )
    rationale: str = Field(
        min_length=5,
        max_length=500,
        description="Why measuring this query helps answer the research question. One sentence.",
    )
    retrieval_kinds: list[RetrievalKind] = Field(
        min_length=1,
        max_length=4,
        description="Which retrieval capabilities this sub-query needs. Include "
        "'keyword_metrics' whenever the query should be sized; include 'ai_overview' and "
        "'llm_visibility' when AI-answer presence matters, which it usually does for "
        "commercial and comparison intent.",
    )


class RetrievalPlan(BaseModel):
    """The complete plan for one run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    interpretation: str = Field(
        min_length=10,
        max_length=1000,
        description="A restatement of what the research question is actually asking, in one "
        "or two sentences. Used to check the plan against the question in the final report.",
    )
    sub_queries: list[PlannedRetrieval] = Field(
        min_length=1,
        max_length=MAX_SUB_QUERIES,
        description="The sub-queries to measure, most important first. Cover the brand's own "
        "category terms, comparison and alternatives queries, and the problems its buyers "
        "search for - not just the brand name.",
    )

    @property
    def is_usable(self) -> bool:
        return bool(self.sub_queries)
