"""Declarative base and shared column types.

Two things are established here that pay off across every table.

**A constraint naming convention.** Without one, PostgreSQL invents names for indexes, checks
and foreign keys, and Alembic then generates migrations that cannot reliably drop what a
previous migration created. Naming them deterministically makes migrations reversible - which
is the difference between a schema you can evolve and one you can only rebuild.

**A type annotation map.** ``Mapped[str]`` alone would default to an unbounded ``VARCHAR``;
mapping Python types to deliberate PostgreSQL types once, here, keeps every model consistent
and keeps the intent visible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from sqlalchemy import DateTime, MetaData, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.type_api import TypeEngine

#: Deterministic names for every generated constraint, so Alembic can drop what it created.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for every persistence model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map: ClassVar[dict[Any, TypeEngine[Any] | type[TypeEngine[Any]]]] = {
        UUID: Uuid,
        str: Text,
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB,
    }


def now_defaulted_timestamp() -> Mapped[datetime]:
    """A timezone-aware timestamp with a database-side default.

    ``server_default`` rather than a Python default: the application always supplies these
    (the domain entities generate them), but the database clock is the one authority every
    writer shares, so a row inserted outside the application still gets a correct value.
    """
    return mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
