"""Profile request and response schemas (spec S4.1)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from sightline.application.dto.results import ProfileSummary
from sightline.domain.entities.profile import Profile
from sightline.domain.value_objects.enums import RunStatus

MAX_COMPETITORS = 20


class ProfileCreateRequest(BaseModel):
    """Body of ``POST /api/v1/profiles``, matching the spec's printed shape exactly."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200, examples=["Surfer SEO"])
    domain: str = Field(min_length=3, max_length=255, examples=["surferseo.com"])
    industry: str = Field(default="", max_length=200, examples=["SEO Software"])
    description: str = Field(
        default="", max_length=2000, examples=["AI-powered SEO content optimization tool"]
    )
    competitors: list[str] = Field(
        default_factory=list,
        max_length=MAX_COMPETITORS,
        examples=[["clearscope.io", "marketmuse.com", "frase.io"]],
    )


class ProfileCreatedResponse(BaseModel):
    """201 response for a newly registered profile."""

    model_config = ConfigDict(frozen=True)

    profile_uuid: UUID
    name: str
    domain: str
    status: Literal["created"] = "created"
    created_at: datetime

    @classmethod
    def from_entity(cls, profile: Profile) -> ProfileCreatedResponse:
        return cls(
            profile_uuid=profile.uuid,
            name=profile.name,
            domain=profile.domain,
            created_at=profile.created_at,
        )


class ProfileStats(BaseModel):
    """The summary statistics spec S4.1 requires on retrieval."""

    model_config = ConfigDict(frozen=True)

    total_runs: int
    latest_run_uuid: UUID | None = None
    latest_run_status: RunStatus | None = None
    latest_run_at: str | None = None
    average_opportunity_score: float | None = Field(
        default=None,
        description="Mean opportunity score across every query discovered for this profile. "
        "Null when no queries exist yet, which is distinct from a mean of 0.0.",
    )


class ProfileDetailResponse(BaseModel):
    """200 response for ``GET /api/v1/profiles/{profile_uuid}``."""

    model_config = ConfigDict(frozen=True)

    profile_uuid: UUID
    name: str
    domain: str
    industry: str
    description: str
    competitors: list[str]
    created_at: datetime
    stats: ProfileStats

    @classmethod
    def from_summary(cls, summary: ProfileSummary) -> ProfileDetailResponse:
        profile = summary.profile
        return cls(
            profile_uuid=profile.uuid,
            name=profile.name,
            domain=profile.domain,
            industry=profile.industry,
            description=profile.description,
            competitors=list(profile.competitors),
            created_at=profile.created_at,
            stats=ProfileStats(
                total_runs=summary.total_runs,
                latest_run_uuid=summary.latest_run_uuid,
                latest_run_status=summary.latest_run_status,
                latest_run_at=summary.latest_run_at,
                average_opportunity_score=summary.average_opportunity_score,
            ),
        )
