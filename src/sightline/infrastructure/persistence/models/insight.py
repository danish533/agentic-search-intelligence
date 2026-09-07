"""``insights`` table."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, Integer
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Text, Uuid

from sightline.infrastructure.persistence.base import Base
from sightline.infrastructure.persistence.models.pipeline_run import PipelineRunModel


class InsightModel(Base):
    """A finding produced by the Analysis agent."""

    __tablename__ = "insights"
    __table_args__ = (
        Index(
            "ix_insights_run_relevance",
            "run_uuid",
            "relevance_score",
            postgresql_ops={"relevance_score": "DESC"},
        ),
        CheckConstraint(
            "relevance_score >= 0 AND relevance_score <= 1", name="relevance_score_in_range"
        ),
    )

    uuid: Mapped[UUID] = mapped_column(primary_key=True)
    run_uuid: Mapped[UUID] = mapped_column(
        ForeignKey("pipeline_runs.uuid", ondelete="CASCADE"), nullable=False
    )

    # Declared purely so SQLAlchemy's unit of work knows this row depends on its parent and
    # orders INSERTs accordingly. Foreign-key *columns* alone do not establish that: the
    # flush sorts by mapper relationships, so without this a child can be inserted before
    # its parent whenever both are written in one transaction.
    # lazy="raise" because nothing should ever traverse these - any attempt is hidden I/O,
    # and under asyncio it would surface as MissingGreenlet rather than a clear error.
    run: Mapped[PipelineRunModel] = relationship(lazy="raise")

    headline: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)

    # An array rather than a join table: this is evidence attached to one insight, never
    # queried from the other direction, and it must survive even if a query row is later
    # rewritten by a recheck. A foreign-keyed join table would couple the two lifecycles.
    supporting_query_uuids: Mapped[list[UUID]] = mapped_column(
        ARRAY(Uuid), nullable=False, default=list, server_default="{}"
    )

    # Preserves the Analysis agent's own ordering as a stable tiebreak within equal scores.
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return f"<InsightModel {self.headline[:40]!r} score={self.relevance_score}>"
