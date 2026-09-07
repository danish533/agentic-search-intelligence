"""``profiles`` table."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Text

from sightline.infrastructure.persistence.base import Base, now_defaulted_timestamp


class ProfileModel(Base):
    """A registered brand profile.

    ``domain`` is unique because it is the natural key a user reasons about - registering the
    same domain twice is a duplicate, not two profiles, and the constraint is what lets the
    API answer 409 instead of quietly creating a second one.
    """

    __tablename__ = "profiles"

    uuid: Mapped[UUID] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    industry: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # A genuine list of hostnames: ARRAY(TEXT) keeps it queryable with the containment
    # operators (@>, &&), which a JSON blob or a comma-joined string would not.
    competitors: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default="{}"
    )

    created_at: Mapped[datetime] = now_defaulted_timestamp()

    def __repr__(self) -> str:
        return f"<ProfileModel domain={self.domain!r} uuid={self.uuid}>"
