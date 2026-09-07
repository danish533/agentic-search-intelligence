"""Report agent output contract.

The Report agent assembles - it does not reason (spec S3.2). Its schema is deliberately
narrow: prose fields that *render* findings the Analysis agent already produced, and nothing
that could constitute a new claim. There is no score field, no ranking field and no
recommendation field here, because every one of those would be a decision, and decisions were
made upstream.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ReportDraft(BaseModel):
    """The human-readable half of the final report."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    headline: str = Field(
        min_length=10,
        max_length=200,
        description="A one-line verdict a busy reader could act on alone.",
    )
    executive_summary: str = Field(
        min_length=50,
        max_length=2000,
        description="Two to four paragraphs covering where the brand is visible, where it is "
        "absent, and who occupies the gaps. State only what the provided findings support - "
        "introduce no facts, figures or competitors that do not appear in the input.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="The findings as standalone bullets, most significant first.",
    )
    recommended_next_steps: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Concrete next actions, drawn from the supplied recommendations only.",
    )
