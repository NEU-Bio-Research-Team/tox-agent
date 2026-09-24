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

``postgres``-marked tests need that database rather than merely preferring it,
and used to ask for it under two different names, neither of which the CI job
set — so the two-replica admission gate and the migrated-DDL gate both reported
green by never running. They now share one name with the fixture above, and
``TOXAGENT_REQUIRE_POSTGRES`` turns "no database, so skipped" into a failure,
which is what a job that started a PostgreSQL service means by running them.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from toxagent.persistence.schema import metadata
from toxagent.persistence.sql.database import Database


TEST_DATABASE_URL = "TOXAGENT_TEST_DATABASE_URL"
REQUIRE_POSTGRES = "TOXAGENT_REQUIRE_POSTGRES"


def _configured_postgres_url() -> str | None:
    url = os.getenv(TEST_DATABASE_URL)
    if url and not url.startswith("postgresql+"):
        raise RuntimeError(f"{TEST_DATABASE_URL} must be an async PostgreSQL URL")
    return url or None


def pytest_runtest_setup(item) -> None:
    """A `postgres`-marked test either runs or says why, by marker not by fixture.

    Enforcing it here rather than inside each test means a new PostgreSQL test
    cannot quietly opt out of the gate by forgetting to ask for the URL.
    """
    if not any(mark.name == "postgres" for mark in item.iter_markers()):
        return
    if _configured_postgres_url() is not None:
        return
    if os.getenv(REQUIRE_POSTGRES):
        raise AssertionError(
            f"{REQUIRE_POSTGRES} is set but {TEST_DATABASE_URL} is not: this job "
            "started a PostgreSQL service and would have skipped the tests that "
            "are the reason it did"
        )
    pytest.skip(f"set {TEST_DATABASE_URL} to run the PostgreSQL gate")


@pytest.fixture
def postgres_url() -> str:
    """A URL for tests that build their own engines, past the `postgres` marker."""
    url = _configured_postgres_url()
    assert url is not None  # pytest_runtest_setup skipped or failed already
    return url


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
async def db(tmp_path):
    test_database_url = _configured_postgres_url()
    database = Database(test_database_url or f"sqlite+aiosqlite:///{tmp_path / 'toxagent.db'}")
    if test_database_url:
        if database.engine.dialect.name != "postgresql":
            raise RuntimeError(f"{TEST_DATABASE_URL} must point to PostgreSQL")
        table_names = ", ".join(table.name for table in metadata.sorted_tables)
        async with database.engine.begin() as connection:
            await connection.execute(text(f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE"))
    else:
        await database.create_schema()
    try:
        yield database
    finally:
        await database.dispose()
