"""PostgreSQL migration contract (remaining-plan W6-09 / W4-01).

This test intentionally never calls ``Database.create_schema()``.  CI starts
the PostgreSQL service with no application tables, runs ``alembic upgrade
head``, and then this file checks the result through the same async repository
layer the application uses.  The environment variable is opt-in so the normal
SQLite-focused developer suite stays self-contained.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, text

from toxagent.domain.events import EventType
from toxagent.domain.session import Session
from toxagent.persistence.schema import metadata
from toxagent.persistence.sql.database import Database

pytestmark = [pytest.mark.anyio, pytest.mark.postgres]


def _postgres_url() -> str:
    url = os.getenv("TOXAGENT_POSTGRES_TEST_URL")
    if not url:
        pytest.skip("set TOXAGENT_POSTGRES_TEST_URL to run PostgreSQL migration checks")
    if not url.startswith("postgresql+"):
        raise AssertionError("TOXAGENT_POSTGRES_TEST_URL must use an async PostgreSQL URL")
    return url


async def test_alembic_migrated_postgresql_schema_supports_repository_writes():
    database = Database(_postgres_url())
    try:
        async with database.engine.connect() as connection:
            table_names = await connection.run_sync(
                lambda sync_connection: set(inspect(sync_connection).get_table_names())
            )
            assert set(metadata.tables) <= table_names

            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            assert revision == "0002_runtime_usage_events"

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
