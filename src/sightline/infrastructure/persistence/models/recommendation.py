"""``recommendations`` table."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import Enum, ForeignKey, Index, Integer
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import String, Text

from sightline.domain.value_objects.enums import ContentType, Priority
from sightline.infrastructure.persistence.base import Base
from sightline.infrastructure.persistence.models.discovered_query import DiscoveredQueryModel
from sightline.infrastructure.persistence.models.pipeline_run import PipelineRunModel

content_type_enum = Enum(
    ContentType,
    name="content_type",
    values_callable=lambda enum: [member.value for member in enum],
)

#: Declared HIGH, MEDIUM, LOW - and a native PostgreSQL enum sorts by declaration order, so
#: "ORDER BY priority" yields high-first with no CASE expression and no sort-key column.
priority_enum = Enum(
    Priority,
    name="priority",
    values_callable=lambda enum: [member.value for member in enum],
)


class RecommendationModel(Base):
    """A content action proposed to close a visibility gap."""

    __tablename__ = "recommendations"
    __table_args__ = (Index("ix_recommendations_run_priority", "run_uuid", "priority"),)

    uuid: Mapped[UUID] = mapped_column(primary_key=True)
    run_uuid: Mapped[UUID] = mapped_column(
        ForeignKey("pipeline_runs.uuid", ondelete="CASCADE"), nullable=False
    )
    target_query_uuid: Mapped[UUID] = mapped_column(
        ForeignKey("discovered_queries.uuid", ondelete="CASCADE"), nullable=False, index=True
    )

    # Declared purely so SQLAlchemy's unit of work knows this row depends on its parent and
    # orders INSERTs accordingly. Foreign-key *columns* alone do not establish that: the
    # flush sorts by mapper relationships, so without this a child can be inserted before
    # its parent whenever both are written in one transaction.
    # lazy="raise" because nothing should ever traverse these - any attempt is hidden I/O,
    # and under asyncio it would surface as MissingGreenlet rather than a clear error.
    run: Mapped[PipelineRunModel] = relationship(lazy="raise")
    target_query: Mapped[DiscoveredQueryModel] = relationship(lazy="raise")

    content_type: Mapped[ContentType] = mapped_column(content_type_enum, nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    target_keywords: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default="{}"
    )
    priority: Mapped[Priority] = mapped_column(priority_enum, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return f"<RecommendationModel {self.title[:40]!r} priority={self.priority.value}>"
