"""``discovered_queries`` table."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, Enum, Float, ForeignKey, Index, Integer, SmallInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Text

from sightline.domain.value_objects.enums import VisibilityStatus
from sightline.infrastructure.persistence.base import Base, now_defaulted_timestamp
from sightline.infrastructure.persistence.models.pipeline_run import PipelineRunModel
from sightline.infrastructure.persistence.models.profile import ProfileModel

visibility_status_enum = Enum(
    VisibilityStatus,
    name="visibility_status",
    values_callable=lambda enum: [member.value for member in enum],
)


class DiscoveredQueryModel(Base):
    """A measured sub-query.

    The check constraints mirror the domain invariants exactly. That duplication is
    deliberate: the domain entity protects the application path, and the database protects
    against every other path - a migration backfill, a manual fix, a future service. An
    invariant enforced in only one of the two is an invariant that will eventually be violated
    in the other.
    """

    __tablename__ = "discovered_queries"
    __table_args__ = (
        # Serves GET /profiles/{uuid}/queries exactly as the spec specifies it: filtered by
        # run, ordered by opportunity score descending. A composite index in the query's own
        # order means the sort is satisfied by the index rather than by a sort node.
        Index(
            "ix_discovered_queries_run_opportunity",
            "run_uuid",
            "opportunity_score",
            postgresql_ops={"opportunity_score": "DESC"},
        ),
        Index("ix_discovered_queries_run_status", "run_uuid", "visibility_status"),
        Index("ix_discovered_queries_profile_discovered", "profile_uuid", "discovered_at"),
        CheckConstraint(
            "opportunity_score >= 0 AND opportunity_score <= 1",
            name="opportunity_score_in_range",
        ),
        CheckConstraint(
            "competitive_difficulty >= 0 AND competitive_difficulty <= 100",
            name="competitive_difficulty_in_range",
        ),
        CheckConstraint("estimated_search_volume >= 0", name="search_volume_non_negative"),
        # The pairing invariant: a position is meaningful only when the domain was found.
        # Without this the database can hold "not visible, at position 3".
        CheckConstraint(
            "(visibility_status = 'visible') = (visibility_position IS NOT NULL)",
            name="position_present_iff_visible",
        ),
        CheckConstraint(
            "visibility_position IS NULL OR visibility_position >= 1",
            name="visibility_position_positive",
        ),
    )

    uuid: Mapped[UUID] = mapped_column(primary_key=True)
    profile_uuid: Mapped[UUID] = mapped_column(
        ForeignKey("profiles.uuid", ondelete="CASCADE"), nullable=False
    )
    run_uuid: Mapped[UUID] = mapped_column(
        ForeignKey("pipeline_runs.uuid", ondelete="CASCADE"), nullable=False
    )

    # Declared purely so SQLAlchemy's unit of work knows this row depends on its parent and
    # orders INSERTs accordingly. Foreign-key *columns* alone do not establish that: the
    # flush sorts by mapper relationships, so without this a child can be inserted before
    # its parent whenever both are written in one transaction.
    # lazy="raise" because nothing should ever traverse these - any attempt is hidden I/O,
    # and under asyncio it would surface as MissingGreenlet rather than a clear error.
    profile: Mapped[ProfileModel] = relationship(lazy="raise")
    run: Mapped[PipelineRunModel] = relationship(lazy="raise")

    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    estimated_search_volume: Mapped[int] = mapped_column(Integer, nullable=False)
    competitive_difficulty: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    opportunity_score: Mapped[float] = mapped_column(Float, nullable=False)

    visibility_status: Mapped[VisibilityStatus] = mapped_column(
        visibility_status_enum, nullable=False
    )
    visibility_position: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # The observations this measurement was derived from, written by the normalization node
    # via NormalizedQuery.to_evidence(). Read back on recheck to report which competing
    # domains entered or left the result set, rather than only that a score moved.
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    discovered_at: Mapped[datetime] = now_defaulted_timestamp()

    def __repr__(self) -> str:
        return f"<DiscoveredQueryModel {self.query_text!r} score={self.opportunity_score}>"
