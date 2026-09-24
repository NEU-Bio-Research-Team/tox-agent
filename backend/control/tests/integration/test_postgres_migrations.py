"""PostgreSQL migration contract (remaining-plan W6-09 / W4-01).

This test intentionally never calls ``Database.create_schema()``.  CI starts
the PostgreSQL service with no application tables, runs ``alembic upgrade
head``, and then this file checks the result through the same async repository
layer the application uses.  ``TOXAGENT_TEST_DATABASE_URL`` is opt-in so the
normal SQLite-focused developer suite stays self-contained, and is the same
variable the ``db`` fixture reads — one name, so a job cannot set the one that
puts every other test on PostgreSQL and miss the one that gates the DDL.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from toxagent.domain.events import EventType
from toxagent.domain.session import Session
from toxagent.persistence.schema import metadata
from toxagent.persistence.sql.database import Database

pytestmark = [pytest.mark.anyio, pytest.mark.postgres]


SERVICE_ROOT = Path(__file__).resolve().parents[2]


def alembic_head() -> str:
    """The head this repository's migration scripts define, read from them.

    The revision used to be written out here as a literal. It was `0002` while
    the chain had reached `0008`, so the one assertion that says "the database
    is fully migrated" was six revisions behind and would have failed the first
    time it ran — which it never did, because the CI job set neither of the two
    different environment variables this file and its neighbour asked for.
    Deriving it means adding a migration cannot leave this test stale.
    """
    config = Config(str(SERVICE_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(SERVICE_ROOT / "migrations"))
    return ScriptDirectory.from_config(config).get_current_head()


async def test_alembic_migrated_postgresql_schema_supports_repository_writes(postgres_url):
    database = Database(postgres_url)
    try:
        async with database.engine.connect() as connection:
            table_names = await connection.run_sync(
                lambda sync_connection: set(inspect(sync_connection).get_table_names())
            )
            assert set(metadata.tables) <= table_names

            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            assert revision == alembic_head()

            session_constraints = await connection.run_sync(
                lambda sync_connection: {
                    constraint["name"]
                    for constraint in inspect(sync_connection).get_unique_constraints("sessions")
                }
            )
            assert "uq_session_idempotency" in session_constraints

            run_constraints = await connection.run_sync(
                lambda sync_connection: {
                    constraint["name"]
                    for constraint in inspect(sync_connection).get_check_constraints("runs")
                }
            )
            assert "ck_deterministic_lane_has_no_runtime" in run_constraints

        session = Session.create(
            "postgres-ci",
            now=datetime.now(timezone.utc),
        )
        async with database.unit_of_work() as uow:
            await uow.sessions.add(session)
            uow.emit(
                session_id=session.id,
                type=EventType.SESSION_CREATED,
                entity_type="session",
                entity_id=session.id,
            )
            await uow.commit()

        async with database.unit_of_work() as uow:
            persisted = await uow.sessions.get(session.id, owner_id="postgres-ci")
        assert persisted is not None
        assert (await database.outbox().read_after(session.id, 0))[0].type is EventType.SESSION_CREATED
    finally:
        await database.dispose()
