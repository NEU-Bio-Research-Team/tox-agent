"""Who owns executing a run, and what happens when that owner dies.

The two failures these reproduce were both consequences of ownership being
implicit — "this process has no task for it, therefore nobody does":

- I17: starting a second replica failed the first replica's live runs, and
  marked them `potentially_billed` while their provider turn was still in
  flight. The trigger is a rolling deploy, modelled here by calling startup
  reconciliation while another worker holds a lease.
- I18: a `kill -9` destroyed the only copy of the request, so an accepted run
  could not be executed by anyone, and a cancellation requested through a
  replica that did not hold the run was written down and never acted on.

Two `RunScheduler` instances over one database model two processes: their task
maps are genuinely separate, which is the whole point, and a shared database is
what a real deployment has.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.policy import Actor
from toxagent.application.run_scheduler import LEASE_TTL_S, RunContext, RunScheduler
from toxagent.application.startup_reconciliation import reconcile_orphaned_runs
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run, RunStatus
from toxagent.domain.session import Session

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def _seed_run(db, *, lane: Lane = Lane.AGENTIC) -> tuple[str, str]:
    session = Session.create(ACTOR.subject_id, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, lane, Intent.REPORT_QA, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.commit()
    return session.id, run.id


def _context(session_id: str, run_id: str, **kwargs) -> RunContext:
    return RunContext(
        actor=ACTOR, session_id=session_id, run_id=run_id,
        intent=Intent.REPORT_QA, text="is this molecule cardiotoxic?", **kwargs
    )


def _long_ago() -> datetime:
    """A moment far enough back that a lease taken then has expired.

    Anchored to the real clock, not the fixed NOW above: `adopt` compares
    against `datetime.now`, so a lease dated from a literal in the test's
    future is never claimable and the test would pass for the wrong reason.
    """
    return datetime.now(timezone.utc) - timedelta(seconds=LEASE_TTL_S + 60)


async def _await_status(db, run_id: str, status: RunStatus, *, timeout: float = 10.0) -> None:
    """Wait for the run to actually reach a state, rather than draining first.

    `drain()` cancels tasks a second time, which would interrupt the very
    `_terminate` call this is waiting on — at shutdown that is correct, but
    here it would hide whether the cancellation was recorded.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        async with db.unit_of_work() as uow:
            run = await uow.runs.get(run_id)
        if run.status is status:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached {status}; it is {run.status}")


async def _enqueue(db, scheduler: RunScheduler, context: RunContext, *, now=None) -> int:
    async with db.unit_of_work() as uow:
        epoch = await scheduler.enqueue(uow, context, now=now)
        await uow.commit()
    return epoch


# --- I17: a starting replica must not touch another's live work -------------

async def test_startup_reconciliation_leaves_a_run_under_a_live_lease_alone(db):
    """The rolling-deploy trigger. Worker A is executing; worker B starts."""
    session_id, run_id = await _seed_run(db)
    worker_a = RunScheduler(db, worker_id="worker-a")
    await _enqueue(db, worker_a, _context(session_id, run_id))

    reconciled = await reconcile_orphaned_runs(db)

    assert reconciled == 0
    async with db.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
    assert not run.is_terminal
    assert run.status is RunStatus.QUEUED
    assert run.potentially_billed is False


async def test_a_run_with_no_durable_envelope_is_still_closed_out(db):
    """Runs from before this schema, and any submitter that cannot enqueue.

    Nothing can execute these, so leaving them open would hold their session
    under the one-active-run cap forever — the reason the original
    reconciliation existed.
    """
    _, run_id = await _seed_run(db)

    assert await reconcile_orphaned_runs(db) == 1

    async with db.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
    assert run.status is RunStatus.FAILED
    assert run.failure_code == "runtime_unavailable"


async def test_an_expired_lease_is_claimed_by_exactly_one_of_two_workers(db):
    """Kill A, wait out the lease: B and C both sweep, one wins."""
    session_id, run_id = await _seed_run(db)
    dead = RunScheduler(db, worker_id="worker-dead")
    await _enqueue(db, dead, _context(session_id, run_id), now=_long_ago())

    executed: list[str] = []
    ran = asyncio.Event()

    async def handler(context: RunContext) -> None:
        executed.append(context.run_id)
        ran.set()

    workers = [RunScheduler(db, worker_id=f"worker-{n}") for n in ("b", "c")]
    for worker in workers:
        worker.register(Intent.REPORT_QA, handler)

    claimed = await asyncio.gather(*(worker.adopt() for worker in workers))

    assert sorted(claimed) == [0, 1], "two workers must not both claim one run"
    await asyncio.wait_for(ran.wait(), timeout=5)
    for worker in workers:
        await worker.drain()
    assert executed == [run_id]


# --- I18: an accepted run survives the process that accepted it -------------

async def test_a_fresh_worker_executes_a_run_left_by_a_dead_one(db):
    """The `kill -9` case. The envelope is the only reason this is possible."""
    session_id, run_id = await _seed_run(db)
    dead = RunScheduler(db, worker_id="worker-dead")
    context = _context(
        session_id, run_id, smiles="CCO", endpoints=("herg",),
        model_selection={"herg": "herg-tox21-chemberta-v1"},
        explanation_targets=(("herg", None),), language="vi",
    )
    await _enqueue(db, dead, context, now=_long_ago())

    seen: list[RunContext] = []
    ran = asyncio.Event()

    async def handler(ctx: RunContext) -> None:
        seen.append(ctx)
        ran.set()

    survivor = RunScheduler(db, worker_id="worker-survivor")
    survivor.register(Intent.REPORT_QA, handler)

    assert await survivor.adopt() == 1
    # Before draining: drain cancels whatever has not finished, and the
    # adopted task has only just been created.
    await asyncio.wait_for(ran.wait(), timeout=5)
    await survivor.drain()

    assert len(seen) == 1
    recovered = seen[0]
    # Not merely "a run happened": the request the user actually made.
    assert recovered.run_id == run_id
    assert recovered.actor.subject_id == ACTOR.subject_id
    assert recovered.text == "is this molecule cardiotoxic?"
    assert recovered.smiles == "CCO"
    assert recovered.endpoints == ("herg",)
    assert recovered.model_selection == {"herg": "herg-tox21-chemberta-v1"}
    assert recovered.explanation_targets == (("herg", None),)
    assert recovered.language == "vi"


async def test_the_job_is_released_once_the_run_reaches_a_terminal_state(db):
    session_id, run_id = await _seed_run(db)
    worker = RunScheduler(db, worker_id="worker-a")
    finished = asyncio.Event()

    async def handler(context: RunContext) -> None:
        async with db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            from toxagent.application.runs import advance
            await advance(uow, run, RunStatus.RUNNING)
            run = await uow.runs.get(context.run_id)
            await advance(uow, run, RunStatus.COMPLETED)
            await uow.commit()
        finished.set()

    worker.register(Intent.REPORT_QA, handler)
    context = _context(session_id, run_id)
    epoch = await _enqueue(db, worker, context)
    worker.submit(context, epoch=epoch)
    await asyncio.wait_for(finished.wait(), timeout=5)

    # The release happens in `_execute`'s `finally`, after the handler returns.
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        async with db.unit_of_work() as uow:
            job = await uow.run_jobs.get(run_id)
        if job is None:
            break
        await asyncio.sleep(0.05)
    await worker.drain()
    assert job is None, "a finished run must not leave a job for the sweep to re-read"


async def test_a_cancellation_through_another_replica_reaches_the_owning_worker(db):
    """B has no task for this run; A does. B's flag must still stop it."""
    session_id, run_id = await _seed_run(db)
    worker_a = RunScheduler(db, worker_id="worker-a")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(context: RunContext) -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    worker_a.register(Intent.REPORT_QA, handler)
    context = _context(session_id, run_id)
    epoch = await _enqueue(db, worker_a, context)
    worker_a.submit(context, epoch=epoch)
    await asyncio.wait_for(started.wait(), timeout=5)

    worker_b = RunScheduler(db, worker_id="worker-b")
    outcome = await worker_b.cancel(run_id)

    # B says what it actually did — relayed, not "no local worker", which read
    # as "nothing will act on this".
    assert outcome.requested is True
    assert outcome.action == "cancellation_relayed_to_owning_worker"

    # And A's supervisor picks it up, within a poll interval.
    await asyncio.wait_for(cancelled.wait(), timeout=15)
    await _await_status(db, run_id, RunStatus.CANCELLED)
    await worker_a.drain()


async def test_a_fenced_worker_stops_executing_the_run_it_lost(db):
    """A was slow, not dead. B adopted. A must not keep working and commit."""
    session_id, run_id = await _seed_run(db)
    worker_a = RunScheduler(db, worker_id="worker-a")
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def handler(context: RunContext) -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            stopped.set()
            raise

    worker_a.register(Intent.REPORT_QA, handler)
    context = _context(session_id, run_id)
    epoch = await _enqueue(db, worker_a, context)
    worker_a.submit(context, epoch=epoch)
    await asyncio.wait_for(started.wait(), timeout=5)

    # Expire A's lease behind its back and let B take the run.
    async with db.unit_of_work() as uow:
        stale = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert await uow.run_jobs.claim(
            run_id, worker_id="worker-b", expected_epoch=epoch,
            lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=LEASE_TTL_S),
            now=datetime.now(timezone.utc),
        ) is None, "a live lease must not be claimable"
        await uow.run_jobs.renew(
            run_id, worker_id="worker-a", epoch=epoch, lease_expires_at=stale,
            now=datetime.now(timezone.utc),
        )
        new_epoch = await uow.run_jobs.claim(
            run_id, worker_id="worker-b", expected_epoch=epoch,
            lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=LEASE_TTL_S),
            now=datetime.now(timezone.utc),
        )
        await uow.commit()
    assert new_epoch == epoch + 1

    await asyncio.wait_for(stopped.wait(), timeout=15)
    await worker_a.drain()

    # A must not stamp a run it no longer owns: B is executing it now.
    async with db.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        job = await uow.run_jobs.get(run_id)
    assert not run.is_terminal
    assert job["worker_id"] == "worker-b"
