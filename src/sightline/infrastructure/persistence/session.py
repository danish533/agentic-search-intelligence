"""Async engine and session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from sightline.config.settings import DatabaseSettings


def create_database_engine(settings: DatabaseSettings) -> AsyncEngine:
    """Build the process-wide async engine.

    ``pool_pre_ping`` costs one trivial round trip per checkout and removes the entire class of
    "connection was closed by the server while idle" failures that otherwise surface as a
    random 500 on the first request after a quiet period.
    """
    return create_async_engine(
        settings.dsn,
        echo=settings.echo_sql,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_seconds,
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build the session factory the unit of work draws from.

    ``expire_on_commit=False`` because entities are mapped out to domain objects before the
    transaction ends; leaving it on would trigger a refresh - and therefore lazy I/O - on
    attribute access after commit.
    """
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
    )
