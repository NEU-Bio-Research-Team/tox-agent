"""Caps that hold across workers, and a shutdown that does not cancel (PR-15).

G4 in the remediation plan: global, tenant and provider caps enforced under
load across several workers; a draining worker that hands its runs on instead
of ending them; recovery bounded to one generation; a dead worker's slots and
leases reclaimed without anyone restarting anything.

Several `RunScheduler` instances over one database stand in for processes, as
in the lease tests. The handlers here do nothing but hold a slot for a moment
and record how many of them were running at once — which is the only number a
cap is about.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import update

from toxagent.application.runs.concurrency import ConcurrencyLimits, SlotLeaser
from toxagent.application.policy import Actor
from toxagent.application.runs.envelope import to_envelope
from toxagent.application.runs.scheduler import LEASE_TTL_S, RunContext, RunScheduler
from toxagent.application.runs.transitions import advance
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run, RunStatus
from toxagent.domain.session import Session
from toxagent.persistence.schema import concurrency_slots, run_jobs

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 13, tzinfo=timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _seed(db, owner: str = "user-1", intent: Intent = Intent.REPORT_QA) -> RunContext:
    session = Session.create(owner, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.AGENTIC, intent, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.commit()
    return RunContext(
        actor=Actor(subject_id=owner), session_id=session.id, run_id=run.id,
        intent=intent, text="q",
    )


async def _accept(db, context: RunContext) -> None:
    api = RunScheduler(db, external=True)
    async with db.unit_of_work() as uow:
        await api.enqueue(uow, context)
        await uow.commit()


class Meter:
    """How many handlers run at once, overall and per tenant."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.by_tenant: Counter[str] = Counter()
        self.peak_by_tenant: Counter[str] = Counter()
        self.finished: list[str] = []
        self.started: Counter[str] = Counter()

    def handler(self, db, *, hold_s: float = 0.05, gate: asyncio.Event | None = None):
        async def run(context: RunContext) -> None:
            tenant = context.actor.subject_id
            self.started[context.run_id] += 1
            self.active += 1
            self.by_tenant[tenant] += 1
            self.peak = max(self.peak, self.active)
            self.peak_by_tenant[tenant] = max(self.peak_by_tenant[tenant], self.by_tenant[tenant])
            try:
                if gate is not None:
                    await gate.wait()
                await asyncio.sleep(hold_s)
            finally:
                self.active -= 1
                self.by_tenant[tenant] -= 1
            async with db.unit_of_work() as uow:
                run_row = await uow.runs.get(context.run_id)
                await advance(uow, run_row, RunStatus.COMPLETED)
                await uow.commit()
            self.finished.append(context.run_id)

        return run


def _worker(db, meter: Meter, limits: ConcurrencyLimits, **kwargs) -> RunScheduler:
    scheduler = RunScheduler(
        db,
        external=True,
        max_in_flight=kwargs.pop("max_in_flight", 5),
        slots=SlotLeaser(db, limits, ttl_s=LEASE_TTL_S),
        quota_retry_s=0.02,
        max_attempts=kwargs.pop("max_attempts", 2),
        **kwargs,
    )
    scheduler.register(Intent.REPORT_QA, meter.handler(db))
    return scheduler


async def _run_until_done(db, workers, contexts, *, timeout: float = 20.0) -> None:
    loops = [asyncio.create_task(w.serve(poll_interval_s=0.01)) for w in workers]
    try:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            async with db.unit_of_work() as uow:
                statuses = [(await uow.runs.get(c.run_id)).status for c in contexts]
            if all(s is RunStatus.COMPLETED for s in statuses):
                return
            await asyncio.sleep(0.02)
        raise AssertionError(f"runs did not finish: {Counter(s.value for s in statuses)}")
    finally:
        for loop in loops:
            loop.cancel()
        await asyncio.gather(*loops, return_exceptions=True)


async def test_a_global_cap_holds_across_three_workers(db):
    meter = Meter()
    limits = ConcurrencyLimits(global_max=4)
    contexts = [await _seed(db) for _ in range(12)]
    for context in contexts:
        await _accept(db, context)

    workers = [_worker(db, meter, limits) for _ in range(3)]
    await _run_until_done(db, workers, contexts)

    assert meter.peak <= 4, meter.peak
    assert sorted(meter.finished) == sorted(c.run_id for c in contexts)
    # Waiting behind a cap is not an attempt: nothing ran twice.
    assert set(meter.started.values()) == {1}

    # A run is terminal a moment before its worker releases the slot (the
    # release is in the scheduler's `finally`), so wait for it rather than
    # asserting on the same tick.
    async def slots_free() -> bool:
        async with db.unit_of_work() as uow:
            return await uow.concurrency_slots.in_use(
                scope="global", scope_key="*", now=_now()
            ) == 0

    deadline = asyncio.get_running_loop().time() + 5
    while not await slots_free():
        assert asyncio.get_running_loop().time() < deadline, "slots were never released"
        await asyncio.sleep(0.02)


async def test_a_tenant_cap_keeps_one_owner_from_occupying_the_fleet(db):
    meter = Meter()
    limits = ConcurrencyLimits(global_max=3, per_tenant=1)
    heavy = [await _seed(db, "user-1") for _ in range(6)]
    light = await _seed(db, "user-2")
    for context in (*heavy, light):
        await _accept(db, context)

    workers = [_worker(db, meter, limits) for _ in range(2)]
    await _run_until_done(db, workers, [*heavy, light])

    assert meter.peak_by_tenant["user-1"] == 1
    assert meter.peak_by_tenant["user-2"] == 1
    # Enqueued last, finished well before the heavy tenant's backlog cleared.
    assert meter.finished.index(light.run_id) < len(meter.finished) - 1


async def test_racing_for_the_last_slots_grants_exactly_the_limit(db):
    leaser = SlotLeaser(db, ConcurrencyLimits(global_max=3), ttl_s=LEASE_TTL_S)
    contexts = [await _seed(db) for _ in range(10)]
    decisions = await asyncio.gather(*(
        leaser.acquire(
            run_id=c.run_id, worker_id=f"w{i}", tenant="user-1", provider=None,
            queue_name="interactive", now=_now(),
        )
        for i, c in enumerate(contexts)
    ))
    assert sum(d.granted for d in decisions) == 3
    assert {d.refused_scope for d in decisions if not d.granted} == {"global"}


async def test_a_draining_worker_hands_its_run_on_instead_of_cancelling_it(db):
    meter = Meter()
    gate = asyncio.Event()
    context = await _seed(db)
    await _accept(db, context)

    draining = RunScheduler(db, external=True, max_attempts=2)
    draining.register(Intent.REPORT_QA, meter.handler(db, gate=gate))
    assert await draining.adopt() == 1
    await asyncio.sleep(0.05)

    await draining.shutdown(grace_s=0.05)

    async with db.unit_of_work() as uow:
        run = await uow.runs.get(context.run_id)
        job = await uow.run_jobs.get(context.run_id)
    assert run.status is not RunStatus.CANCELLED, "shutdown must not end the user's run"
    assert job is not None and job["worker_id"] is None
    assert job["last_error_code"] == "worker_draining"
    assert await draining.adopt() == 0, "a draining worker takes nothing new"

    gate.set()
    successor = RunScheduler(db, external=True, max_attempts=2)
    successor.register(Intent.REPORT_QA, meter.handler(db))
    assert await successor.adopt() == 1
    await _run_until_done(db, [], [context], timeout=5)
    assert meter.started[context.run_id] == 2
    async with db.unit_of_work() as uow:
        assert await uow.run_jobs.get(context.run_id) is None


async def test_recovery_is_bounded_to_one_generation(db):
    meter = Meter()
    context = await _seed(db)
    async with db.unit_of_work() as uow:
        await uow.run_jobs.enqueue(
            context.run_id, to_envelope(context), worker_id=None,
            lease_expires_at=_now(), now=_now(), queue_name="interactive",
        )
        await uow.commit()
    # Two executions already ended with their worker gone.
    async with db.engine.begin() as conn:
        await conn.execute(
            update(run_jobs).where(run_jobs.c.run_id == context.run_id).values(attempts=2)
        )

    worker = RunScheduler(db, external=True, max_attempts=2)
    worker.register(Intent.REPORT_QA, meter.handler(db))
    assert await worker.adopt() == 0

    async with db.unit_of_work() as uow:
        run = await uow.runs.get(context.run_id)
        assert await uow.run_jobs.get(context.run_id) is None
    assert run.status is RunStatus.FAILED
    assert run.failure_code == "recovery_exhausted"
    assert meter.started[context.run_id] == 0


async def test_a_dead_workers_lease_and_slot_are_reclaimed_by_the_next(db):
    meter = Meter()
    limits = ConcurrencyLimits(global_max=1)
    context = await _seed(db)
    await _accept(db, context)

    # A worker claims and takes the only slot, then is killed: no release, no
    # renewal. Modelled by writing its state and letting both leases lapse.
    dead = RunScheduler(db, external=True)
    async with db.unit_of_work() as uow:
        job = await uow.run_jobs.get(context.run_id)
        epoch = await uow.run_jobs.claim(
            context.run_id, worker_id=dead.worker_id, expected_epoch=job["lease_epoch"],
            lease_expires_at=_now() + timedelta(seconds=LEASE_TTL_S), now=_now(),
        )
        await uow.commit()
    assert epoch == 1
    assert (await SlotLeaser(db, limits, ttl_s=LEASE_TTL_S).acquire(
        run_id=context.run_id, worker_id=dead.worker_id, tenant="user-1", provider=None,
        queue_name="interactive", now=_now(),
    )).granted
    past = _now() - timedelta(seconds=1)
    async with db.engine.begin() as conn:
        await conn.execute(update(run_jobs).values(lease_expires_at=past))
        await conn.execute(update(concurrency_slots).values(expires_at=past))

    successor = _worker(db, meter, limits)
    await _run_until_done(db, [successor], [context], timeout=5)
    assert meter.started[context.run_id] == 1


async def test_a_deferred_job_reports_why_and_when_it_can_run(db):
    first = await _seed(db)
    second = await _seed(db)
    await _accept(db, first)
    await _accept(db, second)

    async with db.unit_of_work() as uow:
        ahead_of_second = await uow.run_jobs.position(second.run_id, now=_now())
        assert ahead_of_second["ahead"] == 1
        assert ahead_of_second["queue_name"] == "interactive"

    blocked = RunScheduler(
        db, external=True, quota_retry_s=30,
        slots=SlotLeaser(db, ConcurrencyLimits(global_max=1), ttl_s=LEASE_TTL_S),
    )
    gate = asyncio.Event()
    meter = Meter()
    blocked.register(Intent.REPORT_QA, meter.handler(db, gate=gate))
    assert await blocked.adopt(limit=2) == 1

    async with db.unit_of_work() as uow:
        waiting = await uow.run_jobs.position(second.run_id, now=_now())
    assert waiting["last_error_code"] == "quota_wait:global"
    assert waiting["available_at"] is not None
    gate.set()
    await blocked.shutdown(grace_s=1)
