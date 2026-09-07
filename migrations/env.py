"""Alembic environment.

The DSN comes from application settings rather than alembic.ini, so migrations and the running
service can never disagree about which database they mean, and no credential is committed.

Importing the models package is load-bearing: Alembic compares ``Base.metadata`` against the
live database, and a model that is never imported is a table autogenerate would propose
dropping.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from sightline.config.settings import get_settings
from sightline.infrastructure.persistence import models  # noqa: F401  - populates metadata
from sightline.infrastructure.persistence.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
database_url = get_settings().database.dsn


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting - used to review a migration before applying."""
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(database_url, poolclass=None)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
