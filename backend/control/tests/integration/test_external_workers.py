"""The API accepts; workers execute (WS08 / PR-14).

These model separate processes the way `test_run_lease_ownership` does: several
`RunScheduler` instances over one database, each with its own task map. The
last test goes one step further and starts two whole applications — an `api`
role and a `worker` role — over the same database, which is the topology the
worker entrypoint exists for.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from toxagent.application.policy import Actor
from toxagent.application.runs.scheduler import RunContext, RunScheduler
from toxagent.application.runs.transitions import advance
from toxagent.platform.config import WorkerSettings
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run, RunStatus
from toxagent.domain.session import Session
from tests.support.api import AUTH, api_client, settings, wait_for_run
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 13, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def _seed(db, intent: Intent = Intent.REPORT_QA, lane: Lane = Lane.AGENTIC) -> RunContext:
    session = Session.create(ACTOR.subject_id, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, lane, intent, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.commit()
    return RunContext(actor=ACTOR, session_id=session.id, run_id=run.id, intent=intent, text="q")


async def _accept(db, api: RunScheduler, context: RunContext) -> int:
    async with db.unit_of_work() as uow:
        epoch = await api.enqueue(uow, context)
        await uow.commit()
    api.submit(context, epoch=epoch)
    return epoch


def _completing(db, executed: list[str], gate: asyncio.Event | None = None):
    async def handler(context: RunContext) -> None:
        executed.append(context.run_id)
        if gate is not None:
            await gate.wait()
        async with db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            await advance(uow, run, RunStatus.COMPLETED)
            await uow.commit()

    return handler


async def _status(db, run_id: str) -> RunStatus:
    async with db.unit_of_work() as uow:
        return (await uow.runs.get(run_id)).status


async def _until(predicate, *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


async def test_an_api_scheduler_writes_an_unowned_job_and_executes_nothing(db):
    executed: list[str] = []
    api = RunScheduler(db, external=True)
    api.register(Intent.REPORT_QA, _completing(db, executed))
    context = await _seed(db)

    epoch = await _accept(db, api, context)

    assert epoch == 0
    assert api.in_flight == 0 and executed == []
    async with db.unit_of_work() as uow:
        job = await uow.run_jobs.get(context.run_id)
    assert job["worker_id"] is None
    assert job["queue_name"] == "interactive"
    assert job["attempts"] == 0 and job["lease_epoch"] == 0


async def test_a_worker_claims_only_the_queues_it_serves(db):
    executed: list[str] = []
    api = RunScheduler(db, external=True)
    question = await _seed(db, Intent.REPORT_QA)
    report = await _seed(db, Intent.BUILD_REPORT, Lane.MIXED)
    await _accept(db, api, question)
    await _accept(db, api, report)

    interactive = RunScheduler(db, external=True, queues=("interactive",))
    reports = RunScheduler(db, external=True, queues=("report",))
    for scheduler in (interactive, reports):
        scheduler.register(Intent.REPORT_QA, _completing(db, executed))
        scheduler.register(Intent.BUILD_REPORT, _completing(db, executed))

    assert await interactive.adopt() == 1
    await _until(lambda: _is(db, question.run_id, RunStatus.COMPLETED))
    assert await _status(db, report.run_id) is RunStatus.QUEUED

    assert await reports.adopt() == 1
    await _until(lambda: _is(db, report.run_id, RunStatus.COMPLETED))
    assert executed == [question.run_id, report.run_id]


async def _is(db, run_id: str, status: RunStatus) -> bool:
    return await _status(db, run_id) is status


async def test_two_workers_sweeping_at_once_execute_each_run_exactly_once(db):
    executed: list[str] = []
    api = RunScheduler(db, external=True)
    contexts = [await _seed(db) for _ in range(10)]
    for context in contexts:
        await _accept(db, api, context)

    first = RunScheduler(db, external=True)
    second = RunScheduler(db, external=True)
    for scheduler in (first, second):
        scheduler.register(Intent.REPORT_QA, _completing(db, executed))

    claimed = await asyncio.gather(first.adopt(limit=10), second.adopt(limit=10))
    assert sum(claimed) == 10

    async def all_done() -> bool:
        return all([await _is(db, c.run_id, RunStatus.COMPLETED) for c in contexts])

    await _until(all_done)
    assert sorted(executed) == sorted(c.run_id for c in contexts)


async def test_a_saturated_report_worker_does_not_hold_up_a_question(db):
    executed: list[str] = []
    gate = asyncio.Event()
    api = RunScheduler(db, external=True)
    reports = [await _seed(db, Intent.BUILD_REPORT, Lane.MIXED) for _ in range(2)]
    question = await _seed(db, Intent.REPORT_QA)
    for context in (*reports, question):
        await _accept(db, api, context)

    report_worker = RunScheduler(db, external=True, queues=("report",), max_in_flight=1)
    report_worker.register(Intent.BUILD_REPORT, _completing(db, executed, gate))
    interactive = RunScheduler(db, external=True, queues=("interactive",))
    interactive.register(Intent.REPORT_QA, _completing(db, executed))

    assert await report_worker.adopt() == 1
    # At capacity: a second sweep takes nothing, however much is waiting.
    assert await report_worker.adopt() == 0
    assert await interactive.adopt() == 1
    await _until(lambda: _is(db, question.run_id, RunStatus.COMPLETED))

    gate.set()
    await _until(lambda: _is(db, reports[0].run_id, RunStatus.COMPLETED))
    assert await report_worker.adopt() == 1
    await _until(lambda: _is(db, reports[1].run_id, RunStatus.COMPLETED))


async def test_a_job_written_before_queue_classes_is_routed_by_its_intent(db):
    report = await _seed(db, Intent.BUILD_REPORT, Lane.MIXED)
    from toxagent.application.runs.envelope import to_envelope

    async with db.unit_of_work() as uow:
        await uow.run_jobs.enqueue(
            report.run_id, to_envelope(report), worker_id=None,
            lease_expires_at=datetime.now(timezone.utc), now=datetime.now(timezone.utc),
        )
        await uow.commit()

    executed: list[str] = []
    interactive = RunScheduler(db, external=True, queues=("interactive",))
    interactive.register(Intent.BUILD_REPORT, _completing(db, executed))
    reports = RunScheduler(db, external=True, queues=("report",))
    reports.register(Intent.BUILD_REPORT, _completing(db, executed))

    assert await interactive.adopt() == 0
    assert await reports.adopt() == 1
    await _until(lambda: _is(db, report.run_id, RunStatus.COMPLETED))


async def test_cancelling_a_queued_job_settles_it_without_a_worker(db):
    executed: list[str] = []
    api = RunScheduler(db, external=True)
    context = await _seed(db)
    await _accept(db, api, context)

    outcome = await api.cancel(context.run_id)

    assert outcome.action == "cancelled_before_execution"
    assert await _status(db, context.run_id) is RunStatus.CANCELLED
    worker = RunScheduler(db, external=True)
    worker.register(Intent.REPORT_QA, _completing(db, executed))
    assert await worker.adopt() == 0
    assert executed == []


async def test_an_api_process_and_a_worker_process_over_one_database(db, monkeypatch):
    """Two composition roots: the API never executes; the worker does."""
    monkeypatch.setenv("TOXAGENT_FLAG_EXTERNAL_WORKER_MODE", "1")
    from toxagent.api.app import create_app

    async with api_client(
        db, StubPredictor(), config=settings(worker=WorkerSettings(role="api"))
    ) as client:
        session = await client.post("/v1/sessions", json={}, headers=AUTH)
        session_id = session.json()["session_id"]
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"molecule": {"smiles": "CCO"}},
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        run_id = submitted.json()["run_id"]

        await asyncio.sleep(0.3)
        run = await client.get(f"/v1/sessions/{session_id}/runs/{run_id}", headers=AUTH)
        assert run.json()["status"] == "queued", "an api process must not execute runs"

        worker_app = create_app(
            settings(worker=WorkerSettings(role="worker", poll_interval_s=0.05)),
            database=db,
            predictor=StubPredictor().client(),
        )
        queue = await client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/queue", headers=AUTH
        )
        assert queue.status_code == 200, queue.text
        assert queue.json() | {"attempts": 0} == {
            "run_id": run_id,
            "status": "queued",
            "queue_name": "deterministic",
            "state": "waiting",
            "position_estimate": 0,
            "retry_after_s": None,
            "waiting_on": None,
            "attempts": 0,
        }

        async with worker_app.router.lifespan_context(worker_app):
            finished = await wait_for_run(client, session_id, run_id)
        assert finished["status"] == "completed"
        async with db.unit_of_work() as uow:
            assert await uow.run_jobs.get(run_id) is None
        after = await client.get(f"/v1/sessions/{session_id}/runs/{run_id}/queue", headers=AUTH)
        assert after.json()["state"] == "not_queued"
