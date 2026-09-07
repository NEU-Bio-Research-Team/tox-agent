"""Shared fixtures.

Async tests run on anyio's pytest plugin with the asyncio backend, so the suite
needs no extra plugin dependency. By default the database fixture uses a
temp-file SQLite rather than ``:memory:`` because the control plane opens more
than one connection and an in-memory database is private to the connection that
made it — which would make the outbox tests pass for the wrong reason.

The PostgreSQL CI job instead supplies ``TOXAGENT_TEST_DATABASE_URL`` after
running Alembic. In that mode this fixture deliberately does not call
``create_schema()``: every repository and API path therefore exercises the
migrated DDL. A database-wide truncate gives every test the isolation that its
own temporary SQLite file normally supplies.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from toxagent.persistence.schema import metadata
from toxagent.persistence.sql.database import Database


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
async def db(tmp_path):
    test_database_url = os.getenv("TOXAGENT_TEST_DATABASE_URL")
    database = Database(test_database_url or f"sqlite+aiosqlite:///{tmp_path / 'toxagent.db'}")
    if test_database_url:
        if database.engine.dialect.name != "postgresql":
            raise RuntimeError("TOXAGENT_TEST_DATABASE_URL must point to PostgreSQL")
        table_names = ", ".join(table.name for table in metadata.sorted_tables)
        async with database.engine.begin() as connection:
            await connection.execute(text(f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE"))
    else:
        await database.create_schema()
    try:
        yield database
    finally:
        await database.dispose()
