"""Analysis / Synthesis output contract.

The Analysis agent reasons over normalized data to produce insights (spec S3.2). It does not
fetch, and it does not format the final document.

One deliberate design choice: drafts reference queries by **text**, never by UUID. Asking a
model to echo a UUID invites it to invent one, and a hallucinated identifier is a foreign-key
violation or - worse - a recommendation silently attached to the wrong query. Resolving text
back to an entity is the node's job, and a reference that does not resolve is dropped with a
warning rather than persisted.

Opportunity scores are likewise absent here. They are computed by the domain scoring service
from measured volume, difficulty and visibility - a business rule kept out of a prompt where
it could drift silently between runs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sightline.domain.value_objects.enums import ContentType, Priority

MAX_INSIGHTS = 10
MAX_RECOMMENDATIONS = 10


class InsightDraft(BaseModel):
    """One finding, as proposed by the model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    headline: str = Field(
        min_length=10,
        max_length=200,
        description="The finding in one sentence, stated as a conclusion rather than a "
        "description of the data. Prefer 'Absent from AI answers for comparison queries' "
        "over 'Five queries were measured'.",
    )
    detail: str = Field(
        max_length=1200,
        description="What the evidence shows and why it matters commercially. Cite the "
        "specific queries and competing domains involved.",
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How strongly the retrieved evidence supports this finding. Use 0.9+ only "
        "when several queries point the same way; use below 0.5 when it rests on one "
        "observation.",
    )
    supporting_queries: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="The exact query_text values this finding is drawn from. Copy them "
        "verbatim from the input; do not paraphrase and do not invent identifiers.",
    )


class RecommendationDraft(BaseModel):
    """One proposed content action."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    target_query: str = Field(
        min_length=3,
        max_length=200,
        description="The exact query_text this content would compete for. Copy it verbatim "
        "from the input.",
    )
    content_type: ContentType = Field(description="The kind of asset to produce.")
    title: str = Field(
        min_length=5,
        max_length=300,
        description="A specific, publishable working title - not a topic label.",
    )
    rationale: str = Field(
        min_length=10,
        max_length=800,
        description="Why this specific asset closes this specific gap. Reference the "
        "competitors currently occupying the query.",
    )
    target_keywords: list[str] = Field(
        default_factory=list,
        max_length=15,
        description="Keywords the asset should target, the primary one first.",
    )
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Reserve 'high' for gaps that are both commercially valuable and "
        "realistically winnable.",
    )


class AnalysisResult(BaseModel):
    """The Analysis agent's complete output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    insights: list[InsightDraft] = Field(
        min_length=1,
        max_length=MAX_INSIGHTS,
        description="Findings, most significant first.",
    )
    recommendations: list[RecommendationDraft] = Field(
        default_factory=list,
        max_length=MAX_RECOMMENDATIONS,
        description="Content actions, each tied to a query that was actually measured.",
    )
    competitive_summary: str = Field(
        default="",
        max_length=1000,
        description="Which competing domains dominate the measured surface, and where.",
    )
