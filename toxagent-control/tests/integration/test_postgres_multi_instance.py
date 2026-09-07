"""Admission invariants with two independent control-plane database pools.

SQLite's ``FOR UPDATE`` is intentionally a no-op, so this test runs only when
the PostgreSQL CI job provides its migrated database. Each SubmitMessage has a
separate Database/engine and scheduler object, modelling two API processes
without pretending their in-memory task maps are shared.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from toxagent.application.policy import Actor
from toxagent.application.run_scheduler import RunContext
from toxagent.application.sessions import SessionService
from toxagent.application.submit_message import MessageSubmission, SubmitMessage
from toxagent.config import PolicySettings
from toxagent.domain.errors import AdmissionBusy, Conflict
from toxagent.domain.session import Session
from toxagent.persistence.sql.database import Database
from toxagent.streaming.sse import event_stream
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor

pytestmark = [pytest.mark.anyio, pytest.mark.postgres]

ACTOR = Actor(subject_id="user-1")


@dataclass
class _RecordingScheduler:
    """Per-instance scheduler stand-in that deliberately leaves runs queued."""

    submitted: list[RunContext] = field(default_factory=list)

    def submit(self, context: RunContext) -> None:
        self.submitted.append(context)


def _postgres_url() -> str:
    url = os.getenv("TOXAGENT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set TOXAGENT_TEST_DATABASE_URL to run PostgreSQL multi-instance checks")
    if not url.startswith("postgresql+"):
        raise AssertionError("TOXAGENT_TEST_DATABASE_URL must use an async PostgreSQL URL")
    return url


async def _seed_session(db) -> Session:
    session = Session.create(ACTOR.subject_id, now=datetime.now(timezone.utc))
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.commit()
    return session


def _two_instances(
    *, lock_timeout_ms: int = 1_000,
) -> tuple[Database, Database, SubmitMessage, SubmitMessage]:
    first_db = Database(_postgres_url())
    second_db = Database(_postgres_url())
    settings = PolicySettings(
        max_concurrent_runs_per_session=1,
        admission_lock_timeout_ms=lock_timeout_ms,
    )
    return (
        first_db,
        second_db,
        SubmitMessage(first_db, settings, _RecordingScheduler()),
        SubmitMessage(second_db, settings, _RecordingScheduler()),
    )


async def test_two_instances_replay_one_client_message_id_without_second_run(db):
    session = await _seed_session(db)
    first_db, second_db, first, second = _two_instances()
    try:
        gate = asyncio.Event()

        async def submit(service: SubmitMessage):
            await gate.wait()
            return await service.execute(
                actor=ACTOR,
                session_id=session.id,
                submission=MessageSubmission(
                    client_message_id="same-mobile-retry",
                    smiles="CCO",
                ),
            )

        attempts = [asyncio.create_task(submit(first)), asyncio.create_task(submit(second))]
        gate.set()
        first_result, second_result = await asyncio.wait_for(asyncio.gather(*attempts), timeout=3)

        assert first_result.run_id == second_result.run_id
        assert {first_result.duplicate_of, second_result.duplicate_of} == {None, first_result.message_id}
        async with db.unit_of_work() as uow:
            messages = await uow.messages.list_for_session(session.id)
            runs = await uow.runs.list_for_session(session.id)
        assert len(messages) == len(runs) == 1
    finally:
        await first_db.dispose()
        await second_db.dispose()


async def test_admission_lock_timeout_is_a_retryable_conflict(db):
    session = await _seed_session(db)
    holder = Database(_postgres_url())
    first_db, second_db, first, second = _two_instances(lock_timeout_ms=20)
    try:
        async with holder.unit_of_work() as uow:
            assert await uow.sessions.get_for_admission(
                session.id, owner_id=ACTOR.subject_id, lock_timeout_ms=1_000
            ) is not None
            with pytest.raises(AdmissionBusy) as raised:
                await first.execute(
                    actor=ACTOR,
                    session_id=session.id,
                    submission=MessageSubmission(client_message_id="blocked", smiles="CCO"),
                )
            assert raised.value.retryable is True
            assert raised.value.detail["retry_after_ms"] == 20
            await uow.rollback()
    finally:
        await holder.dispose()
        await first_db.dispose()
        await second_db.dispose()


async def test_another_instance_serves_rest_and_sse_from_the_committed_outbox(db):
    """A writer's in-process notifier is not shared; B must poll its outbox."""
    writer = Database(_postgres_url())
    reader = Database(_postgres_url())
    try:
        async with api_client(reader, StubPredictor()) as client:
            session = await SessionService(writer).create(ACTOR, title="written by A")

            # This HTTP response is served by B's app/database pool, not the
            # service object which performed A's write above.
            response = await client.get(f"/v1/sessions/{session.id}", headers=AUTH)
            assert response.status_code == 200
            projection = response.json()
            assert projection["session_id"] == session.id
            assert projection["title"] == "written by A"
            assert projection["latest_event_sequence"] == 1

            # A's commit hook belongs to a different process and cannot wake
            # B's notifier. The feed still reconciles from B's outbox reader
            # on its short poll, which is the multi-instance guarantee.
            stream = event_stream(
                reader.outbox(),
                client.app.state.notifier,
                session.id,
                poll_seconds=0.01,
                max_idle_seconds=0.05,
            )
            frame = await anext(stream)
            await stream.aclose()
            assert frame["event"] == "session.created"
            assert frame["id"] == "1"
    finally:
        await writer.dispose()
        await reader.dispose()


async def test_two_instances_admit_only_one_distinct_message_at_the_run_cap(db):
    session = await _seed_session(db)
    first_db, second_db, first, second = _two_instances()
    try:
        gate = asyncio.Event()

        async def submit(service: SubmitMessage, client_message_id: str):
            await gate.wait()
            return await service.execute(
                actor=ACTOR,
                session_id=session.id,
                submission=MessageSubmission(client_message_id=client_message_id, smiles="CCO"),
            )

        attempts = [
            asyncio.create_task(submit(first, "instance-a")),
            asyncio.create_task(submit(second, "instance-b")),
        ]
        gate.set()
        results = await asyncio.wait_for(asyncio.gather(*attempts, return_exceptions=True), timeout=3)

        accepted = [result for result in results if not isinstance(result, BaseException)]
        conflicts = [result for result in results if isinstance(result, Conflict)]
        assert len(accepted) == len(conflicts) == 1
        assert conflicts[0].code == "conflict"
        async with db.unit_of_work() as uow:
            runs = await uow.runs.list_for_session(session.id)
        assert len(runs) == 1
    finally:
        await first_db.dispose()
        await second_db.dispose()
