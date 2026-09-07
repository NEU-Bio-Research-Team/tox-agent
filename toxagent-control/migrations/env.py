"""Alembic environment.

The target metadata is the one the application uses, so autogenerate compares
against the schema that actually ships rather than a second copy of it. The URL
comes from the environment; there is no default, because a migration that picks
its own database is a migration that eventually picks the wrong one.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toxagent.persistence.schema import metadata  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata


def database_url() -> str:
    url = os.getenv("TOXAGENT_DATABASE_URL")
    if not url:
        raise RuntimeError("TOXAGENT_DATABASE_URL must be set to run migrations")
    # Migrations run synchronously. SQLite's built-in driver needs no explicit
    # suffix; PostgreSQL must name psycopg rather than falling back to the
    # legacy psycopg2 default, which this project does not install.
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "+psycopg", 1)
    return url.replace("+aiosqlite", "", 1)


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, compare_type=True
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
