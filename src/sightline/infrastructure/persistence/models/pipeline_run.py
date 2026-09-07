"""``pipeline_runs`` table."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, Enum, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Text

from sightline.domain.value_objects.enums import DegradationReason, RunStatus
from sightline.infrastructure.persistence.base import Base
from sightline.infrastructure.persistence.models.profile import ProfileModel

#: Native PostgreSQL enums, storing the StrEnum *values* the spec prints ("not_visible",
#: "blog_post"), not the Python member names. ``values_callable`` is what makes that so.
run_status_enum = Enum(
    RunStatus,
    name="run_status",
    values_callable=lambda enum: [member.value for member in enum],
)

degradation_reason_enum = Enum(
    DegradationReason,
    name="degradation_reason",
    values_callable=lambda enum: [member.value for member in enum],
)


class PipelineRunModel(Base):
    """One DAG execution.

    ``report`` and ``metrics`` are ``JSONB`` rather than a wide set of columns: their shape is
    owned by the Report agent and will evolve, and they are read whole rather than filtered on.
    ``JSONB`` keeps them indexable if that changes, which a ``TEXT`` blob would not.
    """

    __tablename__ = "pipeline_runs"
    __table_args__ = (
        # The exact lookup behind "most recent run status" in GET /profiles/{uuid}.
        Index("ix_pipeline_runs_profile_started", "profile_uuid", "started_at"),
        Index("ix_pipeline_runs_correlation_id", "correlation_id"),
        CheckConstraint(
            "completed_at IS NULL OR completed_at >= started_at",
            name="completed_after_started",
        ),
        CheckConstraint("total_tokens >= 0", name="total_tokens_non_negative"),
    )

    uuid: Mapped[UUID] = mapped_column(primary_key=True)
    profile_uuid: Mapped[UUID] = mapped_column(
        ForeignKey("profiles.uuid", ondelete="CASCADE"), nullable=False, index=True
    )

    # Indexed because it is the field an on-call engineer greps logs by, then looks up here.
    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[RunStatus] = mapped_column(run_status_enum, nullable=False)
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    planned_retrieval_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successful_retrieval_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    normalized_record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    degradation_reason: Mapped[DegradationReason | None] = mapped_column(
        degradation_reason_enum, nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Declared purely so SQLAlchemy's unit of work knows this row depends on its parent and
    # orders INSERTs accordingly. Foreign-key *columns* alone do not establish that: the
    # flush sorts by mapper relationships, so without this a child can be inserted before
    # its parent whenever both are written in one transaction.
    # lazy="raise" because nothing should ever traverse these - any attempt is hidden I/O,
    # and under asyncio it would surface as MissingGreenlet rather than a clear error.
    profile: Mapped[ProfileModel] = relationship(lazy="raise")

    report: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<PipelineRunModel uuid={self.uuid} status={self.status.value}>"
