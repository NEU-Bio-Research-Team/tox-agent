"""Runs and everything leased or recorded per run: jobs, concurrency slots, configuration snapshots, runtime bindings and usage, tool calls, capability tokens."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import and_, delete, func, insert, literal, or_, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ....domain.errors import Conflict
from ....domain.run import Run
from ....domain.runtime import RuntimeBinding
from ....domain.usage import RuntimeUsageEvent
from ...schema import (
    capability_tokens,
    concurrency_slots,
    run_configuration_snapshots,
    run_jobs,
    runs,
    runtime_bindings,
    runtime_usage_events,
    tool_calls,
)
from .. import mapping as m


class SqlRunConfigurationSnapshotStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, run_id: str, *, ai_profile_id: str | None,
                  predictor_bindings: dict[str, str], now: datetime,
                  intent_decision: dict[str, Any] | None = None,
                  effective_budget: dict[str, Any] | None = None) -> None:
        await self._conn.execute(insert(run_configuration_snapshots).values(
            run_id=run_id, ai_profile_id=ai_profile_id,
            predictor_bindings=dict(predictor_bindings),
            intent_decision=dict(intent_decision) if intent_decision else None,
            effective_budget=dict(effective_budget) if effective_budget else None,
            created_at=now,
        ))

    async def get(self, run_id: str) -> dict[str, Any] | None:
        row = (await self._conn.execute(select(run_configuration_snapshots).where(
            run_configuration_snapshots.c.run_id == run_id
        ))).mappings().first()
        if row is None:
            return None
        return {
            "ai_profile_id": row["ai_profile_id"],
            "predictor_bindings": dict(row["predictor_bindings"] or {}),
            "intent_decision": dict(row["intent_decision"] or {}),
            "effective_budget": dict(row["effective_budget"]) if row.get("effective_budget") else None,
            "created_at": row["created_at"],
        }


class SqlRunStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, run: Run) -> None:
        await self._conn.execute(insert(runs).values(m.run_to_row(run)))

    async def get(self, run_id: str) -> Run | None:
        row = (await self._conn.execute(select(runs).where(runs.c.id == run_id))).mappings().first()
        return m.row_to_run(row) if row else None

    async def update(self, run: Run, *, expected_version: int) -> None:
        row = m.run_to_row(run)
        row.pop("id")
        result = await self._conn.execute(
            update(runs)
            .where(and_(runs.c.id == run.id, runs.c.version == expected_version))
            .values(**row)
        )
        if result.rowcount == 0:
            raise Conflict("run changed underneath this write", run_id=run.id)

    async def list_for_session(self, session_id: str, *, limit: int = 50) -> Sequence[Run]:
        rows = (
            await self._conn.execute(
                select(runs)
                .where(runs.c.session_id == session_id)
                .order_by(runs.c.created_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [m.row_to_run(r) for r in rows]

    async def count_by_session(self, session_ids: Sequence[str]) -> dict[str, int]:
        """How many runs each session actually has.

        I20: the session list reported `len(runs)` from a page capped at ten,
        so any session past its tenth run reported ten forever. A count is a
        count; if a cap were wanted the field would have to say so.
        """
        if not session_ids:
            return {}
        rows = (
            await self._conn.execute(
                select(runs.c.session_id, func.count().label("n"))
                .where(runs.c.session_id.in_(session_ids))
                .group_by(runs.c.session_id)
            )
        ).mappings().all()
        return {row["session_id"]: int(row["n"]) for row in rows}

    async def active_by_session(self, session_ids: Sequence[str]) -> dict[str, Run]:
        """The non-terminal run of each session, where there is one.

        At most one per session by the admission cap, so a newest-first read
        with no per-session limit is a single query over an indexed column.
        """
        if not session_ids:
            return {}
        rows = (
            await self._conn.execute(
                select(runs)
                .where(and_(
                    runs.c.session_id.in_(session_ids),
                    runs.c.status.in_(("queued", "running", "validating")),
                ))
                .order_by(runs.c.created_at.desc())
            )
        ).mappings().all()
        active: dict[str, Run] = {}
        for row in rows:
            active.setdefault(row["session_id"], m.row_to_run(row))
        return active

    async def list_non_terminal(self, *, limit: int = 1000) -> Sequence[Run]:
        """Every run still ``queued``/``running``/``validating``, across every
        session — used only by startup reconciliation, where "unowned by any
        in-process task" and "not terminal" are the same set (plan section
        6.5's cancellation policy assumes exactly one process owns a run)."""
        rows = (
            await self._conn.execute(
                select(runs)
                .where(runs.c.status.in_(("queued", "running", "validating")))
                .order_by(runs.c.created_at)
                .limit(limit)
            )
        ).mappings().all()
        return [m.row_to_run(r) for r in rows]

    async def request_cancel(self, run_id: str) -> bool:
        """Flag a cancellation request. Whether it is honoured, and how, is the
        gateway's business; this only records that it was asked for."""
        result = await self._conn.execute(
            update(runs)
            .where(and_(runs.c.id == run_id, runs.c.status.in_(("queued", "running", "validating"))))
            .values(cancel_requested=True)
        )
        return result.rowcount > 0

    async def cancel_requested(self, run_id: str) -> bool:
        return bool(
            (
                await self._conn.execute(
                    select(runs.c.cancel_requested).where(runs.c.id == run_id)
                )
            ).scalar()
        )


class SqlRunJobStore:
    """The durable execution record for a run, and who owns executing it.

    Ownership is a lease, not a lock: a worker that stops renewing loses the
    run to whoever claims it next, which is the only way a `kill -9` can be
    told from a worker that is merely busy (I17/I18). Every state change is a
    single conditional UPDATE — the condition *is* the concurrency control, so
    two workers racing produce one winner and one rowcount of zero rather than
    two owners.
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def enqueue(
        self,
        run_id: str,
        envelope: dict[str, Any],
        *,
        worker_id: str | None,
        lease_expires_at: datetime,
        now: datetime,
        queue_name: str | None = None,
        priority: int = 0,
    ) -> int:
        """Record the run's execution input, claimed by this worker — or by nobody.

        Written in the transaction that creates the run, so there is no window
        in which an accepted run exists with no way to execute it. Returns the
        fencing epoch the caller must present for every later write.

        ``worker_id=None`` is the external-worker path (WS08): the job is
        written unowned at epoch 0 with a lease that has already expired, which
        is exactly what makes it claimable by ``claim``. Nothing about a job
        waiting for its first worker differs from one whose worker died.
        """
        owned = worker_id is not None
        await self._conn.execute(
            insert(run_jobs).values(
                run_id=run_id, envelope=envelope, worker_id=worker_id,
                lease_expires_at=lease_expires_at,
                lease_epoch=1 if owned else 0, attempts=1 if owned else 0,
                queue_name=queue_name, priority=priority,
                created_at=now, updated_at=now,
            )
        )
        return 1 if owned else 0

    async def renew(
        self, run_id: str, *, worker_id: str, epoch: int, lease_expires_at: datetime,
        now: datetime,
    ) -> bool:
        """Extend this worker's lease. False means it no longer holds it.

        A false here is the fencing signal: something else claimed the run
        because this worker looked dead. The caller must stop working on it —
        continuing would mean two workers executing one run.
        """
        result = await self._conn.execute(
            update(run_jobs)
            .where(and_(
                run_jobs.c.run_id == run_id,
                run_jobs.c.worker_id == worker_id,
                run_jobs.c.lease_epoch == epoch,
            ))
            .values(lease_expires_at=lease_expires_at, updated_at=now)
        )
        return result.rowcount > 0

    async def release(self, run_id: str, *, worker_id: str, epoch: int) -> bool:
        """Drop the job once its run is terminal.

        Conditional on still holding the lease so a fenced worker finishing
        late cannot delete the row the new owner is working under.
        """
        result = await self._conn.execute(
            delete(run_jobs).where(and_(
                run_jobs.c.run_id == run_id,
                run_jobs.c.worker_id == worker_id,
                run_jobs.c.lease_epoch == epoch,
            ))
        )
        return result.rowcount > 0

    async def discard(self, run_id: str) -> None:
        """Remove the job regardless of owner — for a run reconciled to a
        terminal state, where there is nothing left for any worker to own."""
        await self._conn.execute(delete(run_jobs).where(run_jobs.c.run_id == run_id))

    async def claimable(
        self,
        *,
        now: datetime,
        limit: int = 100,
        queue_names: Sequence[str] | None = None,
    ) -> Sequence[dict[str, Any]]:
        """Jobs whose lease has expired, by priority then oldest first.

        A live lease is deliberately not returned: another replica is running
        that job right now, and the whole point of I17 is that starting a new
        process must not disturb it. Neither is a job deferred until later.

        ``queue_names`` narrows to the queues a worker serves. A row with no
        queue (written before 0014) is returned regardless, and the caller
        derives its queue from the envelope.
        """
        conditions = [
            run_jobs.c.lease_expires_at <= now,
            or_(run_jobs.c.available_at.is_(None), run_jobs.c.available_at <= now),
        ]
        if queue_names is not None:
            conditions.append(
                or_(run_jobs.c.queue_name.in_(list(queue_names)), run_jobs.c.queue_name.is_(None))
            )
        rows = (
            await self._conn.execute(
                select(run_jobs)
                .where(and_(*conditions))
                .order_by(run_jobs.c.priority, run_jobs.c.created_at)
                .limit(limit)
            )
        ).mappings().all()
        return [dict(row) for row in rows]

    async def claim(
        self, run_id: str, *, worker_id: str, expected_epoch: int,
        lease_expires_at: datetime, now: datetime,
    ) -> int | None:
        """Take over an expired lease. Returns the new epoch, or None if lost.

        `expected_epoch` is the epoch this worker read when it decided the job
        was claimable. If another worker claimed it in between, the epoch has
        moved and this returns None — one winner, no coordination needed
        beyond the row itself.
        """
        result = await self._conn.execute(
            update(run_jobs)
            .where(and_(
                run_jobs.c.run_id == run_id,
                run_jobs.c.lease_epoch == expected_epoch,
                run_jobs.c.lease_expires_at <= now,
            ))
            .values(
                worker_id=worker_id, lease_expires_at=lease_expires_at,
                lease_epoch=expected_epoch + 1, attempts=run_jobs.c.attempts + 1,
                updated_at=now,
            )
        )
        return expected_epoch + 1 if result.rowcount > 0 else None

    async def get(self, run_id: str) -> dict[str, Any] | None:
        row = (
            await self._conn.execute(select(run_jobs).where(run_jobs.c.run_id == run_id))
        ).mappings().first()
        return dict(row) if row else None

    async def held_by(self, worker_id: str) -> Sequence[str]:
        """Run ids this worker currently leases — the set it must watch for a
        cancellation requested through some other replica."""
        rows = (
            await self._conn.execute(
                select(run_jobs.c.run_id).where(run_jobs.c.worker_id == worker_id)
            )
        ).scalars().all()
        return list(rows)

    async def defer(
        self, run_id: str, *, worker_id: str, epoch: int, available_at: datetime,
        error_code: str, now: datetime,
    ) -> bool:
        """Give a claimed job back, not before ``available_at`` (PR-15).

        For a job that could not get its concurrency slots. It never started,
        so the claim is not counted as an attempt — a run that waited behind a
        cap ten times has not failed ten times.
        """
        result = await self._conn.execute(
            update(run_jobs)
            .where(and_(
                run_jobs.c.run_id == run_id,
                run_jobs.c.worker_id == worker_id,
                run_jobs.c.lease_epoch == epoch,
            ))
            .values(
                worker_id=None, lease_expires_at=available_at, available_at=available_at,
                last_error_code=error_code, attempts=run_jobs.c.attempts - 1, updated_at=now,
            )
        )
        return result.rowcount > 0

    async def hand_off(
        self, run_id: str, *, worker_id: str, epoch: int, now: datetime,
        error_code: str = "worker_draining",
    ) -> bool:
        """Release a job that was executing, for another worker to take now.

        Unlike ``defer`` this counts: the run did start, may have reached a
        provider, and the attempt bound is what stops a job that kills every
        worker it lands on from circulating forever.
        """
        result = await self._conn.execute(
            update(run_jobs)
            .where(and_(
                run_jobs.c.run_id == run_id,
                run_jobs.c.worker_id == worker_id,
                run_jobs.c.lease_epoch == epoch,
            ))
            .values(
                worker_id=None, lease_expires_at=now, last_error_code=error_code,
                updated_at=now,
            )
        )
        return result.rowcount > 0

    async def position(self, run_id: str, *, now: datetime) -> dict[str, Any] | None:
        """Where a waiting job stands in its queue. An estimate, and says so."""
        job = await self.get(run_id)
        if job is None:
            return None
        queue_name = job.get("queue_name")
        ahead = 0
        if job.get("worker_id") is None and queue_name:
            ahead = int(
                (
                    await self._conn.execute(
                        select(func.count())
                        .select_from(run_jobs)
                        .where(and_(
                            run_jobs.c.queue_name == queue_name,
                            run_jobs.c.worker_id.is_(None),
                            or_(
                                run_jobs.c.priority < job["priority"],
                                and_(
                                    run_jobs.c.priority == job["priority"],
                                    run_jobs.c.created_at < job["created_at"],
                                ),
                            ),
                        ))
                    )
                ).scalar()
                or 0
            )
        return {**job, "ahead": ahead}


class SqlConcurrencySlotStore:
    """Slot leases for the caps in ``application.concurrency`` (PR-15).

    Every write is conditional and says in its rowcount whether it won, so the
    caller never has to read the table to decide anything.
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    def _insert(self):
        if self._conn.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        else:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        return dialect_insert(concurrency_slots)

    async def take(
        self, *, scope: str, scope_key: str, limit: int, run_id: str, worker_id: str,
        expires_at: datetime, now: datetime,
    ) -> bool:
        slot = concurrency_slots.c
        # Already holding one in this scope — a job deferred and re-claimed by
        # the same run. Refresh it rather than take a second.
        refreshed = await self._conn.execute(
            update(concurrency_slots)
            .where(and_(
                slot.scope == scope, slot.scope_key == scope_key, slot.run_id == run_id,
                slot.expires_at > now,
            ))
            .values(worker_id=worker_id, expires_at=expires_at)
        )
        if refreshed.rowcount > 0:
            return True
        for index in range(limit):
            reclaimed = await self._conn.execute(
                update(concurrency_slots)
                .where(and_(
                    slot.scope == scope, slot.scope_key == scope_key,
                    slot.slot_index == index, slot.expires_at <= now,
                ))
                .values(
                    run_id=run_id, worker_id=worker_id, expires_at=expires_at,
                    acquired_at=now,
                )
            )
            if reclaimed.rowcount > 0:
                return True
            inserted = await self._conn.execute(
                self._insert()
                .values(
                    scope=scope, scope_key=scope_key, slot_index=index, run_id=run_id,
                    worker_id=worker_id, expires_at=expires_at, acquired_at=now,
                )
                .on_conflict_do_nothing()
            )
            if inserted.rowcount > 0:
                return True
        return False

    async def renew(self, *, run_id: str, worker_id: str, expires_at: datetime) -> int:
        result = await self._conn.execute(
            update(concurrency_slots)
            .where(and_(
                concurrency_slots.c.run_id == run_id,
                concurrency_slots.c.worker_id == worker_id,
            ))
            .values(expires_at=expires_at)
        )
        return result.rowcount

    async def release(self, *, run_id: str, worker_id: str) -> None:
        # Scoped to the worker: a worker that was fenced and finishes late must
        # not free the slots its successor now holds under the same run id.
        await self._conn.execute(
            delete(concurrency_slots).where(and_(
                concurrency_slots.c.run_id == run_id,
                concurrency_slots.c.worker_id == worker_id,
            ))
        )

    async def in_use(self, *, scope: str, scope_key: str, now: datetime) -> int:
        return int(
            (
                await self._conn.execute(
                    select(func.count())
                    .select_from(concurrency_slots)
                    .where(and_(
                        concurrency_slots.c.scope == scope,
                        concurrency_slots.c.scope_key == scope_key,
                        concurrency_slots.c.expires_at > now,
                    ))
                )
            ).scalar()
            or 0
        )


class SqlRuntimeBindingStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, binding: RuntimeBinding) -> None:
        await self._conn.execute(insert(runtime_bindings).values(m.binding_to_row(binding)))

    async def get(self, binding_id: str) -> RuntimeBinding | None:
        row = (
            await self._conn.execute(
                select(runtime_bindings).where(runtime_bindings.c.id == binding_id)
            )
        ).mappings().first()
        return m.row_to_binding(row) if row else None

    async def active_for_session(self, session_id: str) -> RuntimeBinding | None:
        row = (
            await self._conn.execute(
                select(runtime_bindings)
                .where(
                    and_(
                        runtime_bindings.c.session_id == session_id,
                        runtime_bindings.c.status == "active",
                    )
                )
                .order_by(runtime_bindings.c.created_at.desc())
                .limit(1)
            )
        ).mappings().first()
        return m.row_to_binding(row) if row else None

    async def set_status(self, binding_id: str, status: str, *, now: datetime) -> None:
        await self._conn.execute(
            update(runtime_bindings)
            .where(runtime_bindings.c.id == binding_id)
            .values(status=status, closed_at=now)
        )


class SqlRuntimeUsageStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, event: RuntimeUsageEvent) -> None:
        await self._conn.execute(insert(runtime_usage_events).values(m.usage_to_row(event)))

    async def list_for_run(self, run_id: str) -> Sequence[RuntimeUsageEvent]:
        rows = (
            await self._conn.execute(
                select(runtime_usage_events)
                .where(runtime_usage_events.c.run_id == run_id)
                .order_by(runtime_usage_events.c.reported_at, runtime_usage_events.c.id)
            )
        ).mappings().all()
        return [m.row_to_usage(row) for row in rows]


class SqlToolCallStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def try_reserve(
        self, *, call_id: str, session_id: str, run_id: str, tool_name: str,
        arguments_sha256: str, now: datetime, max_calls: int | None, max_identical: int,
    ) -> bool:
        """Atomically check the per-run budget and duplicate-call cap and, if
        both allow it, reserve the call by inserting its ``running`` row — all
        as one ``INSERT ... SELECT ... WHERE`` statement, so the count and the
        reservation cannot be observed and acted on separately by two
        concurrent calls (the race a plain "SELECT count, then INSERT" allows:
        several callers can each see room under the budget before any of them
        has actually taken a slot). ``status='denied'`` rows are excluded from
        both counts — a denied attempt is kept for audit but must not itself
        shrink the budget for the next real attempt. ``max_calls=None`` (used
        for the final-answer tool) skips the budget check entirely; the
        duplicate-call cap still applies.

        Returns whether the reservation succeeded.
        """
        if self._conn.dialect.name == "postgresql":
            # One statement is not one serialization point. Under READ
            # COMMITTED each statement takes its own snapshot and an
            # INSERT ... SELECT takes no lock that would stop a concurrent
            # transaction inserting the row its own count did not see, so five
            # concurrent calls against a budget of two were all admitted. The
            # docstring above was true of SQLite only, where a database-level
            # write lock serialized them for reasons that have nothing to do
            # with this query.
            #
            # Serializing on the parent run makes the claim true on both:
            # reservations for one run queue behind each other, and the count
            # below then runs in a statement whose snapshot includes whatever
            # the previous holder committed. NO KEY UPDATE for the same reason
            # `get_for_admission` uses it — a reservation never changes the
            # run's key, and child rows still pass their foreign-key check.
            await self._conn.execute(
                select(runs.c.id).where(runs.c.id == run_id).with_for_update(key_share=True)
            )
        not_denied = tool_calls.c.status != "denied"
        conditions = [
            (
                select(func.count())
                .select_from(tool_calls)
                .where(and_(tool_calls.c.run_id == run_id, not_denied))
                .scalar_subquery()
                < max_calls
            )
        ] if max_calls is not None else []
        conditions.append(
            select(func.count())
            .select_from(tool_calls)
            .where(
                and_(
                    tool_calls.c.run_id == run_id,
                    tool_calls.c.tool_name == tool_name,
                    tool_calls.c.arguments_sha256 == arguments_sha256,
                    not_denied,
                )
            )
            .scalar_subquery()
            < max_identical
        )

        source = select(
            literal(call_id), literal(session_id), literal(run_id), literal(tool_name),
            literal(arguments_sha256), literal("running"),
            literal([], type_=tool_calls.c.observation_ids.type), literal(now),
        ).where(and_(*conditions))
        stmt = insert(tool_calls).from_select(
            ["id", "session_id", "run_id", "tool_name", "arguments_sha256", "status",
             "observation_ids", "started_at"],
            source,
        )
        result = await self._conn.execute(stmt)
        return result.rowcount == 1

    async def record_denied(
        self, *, call_id: str, session_id: str, run_id: str, tool_name: str,
        arguments_sha256: str, error_code: str, now: datetime,
    ) -> None:
        """A denied call still leaves an audit trail, distinct from ``running``/
        ``completed``/``error`` (which describe a call that was actually
        admitted) so ``count_for_run``-style budget accounting can exclude it
        while `get_run`'s tool-call listing still shows every attempt."""
        await self._conn.execute(
            insert(tool_calls).values(
                id=call_id, session_id=session_id, run_id=run_id, tool_name=tool_name,
                arguments_sha256=arguments_sha256, status="denied", error_code=error_code,
                observation_ids=[], started_at=now, ended_at=now, duration_ms=0,
            )
        )

    async def finish(
        self, call_id: str, *, status: str, error_code: str | None,
        observation_ids: list[str], duration_ms: int, now: datetime,
    ) -> None:
        await self._conn.execute(
            update(tool_calls)
            .where(tool_calls.c.id == call_id)
            .values(
                status=status, error_code=error_code, observation_ids=observation_ids,
                duration_ms=duration_ms, ended_at=now,
            )
        )

    async def count_for_run(self, run_id: str) -> int:
        """Admitted calls only — same accounting ``try_reserve`` enforces, so
        a denied attempt never itself counts against the budget it was
        denied under."""
        return int(
            (
                await self._conn.execute(
                    select(func.count())
                    .select_from(tool_calls)
                    .where(and_(tool_calls.c.run_id == run_id, tool_calls.c.status != "denied"))
                )
            ).scalar()
            or 0
        )

    async def count_for_run_and_tool(self, run_id: str, tool_name: str) -> int:
        """Admitted calls to one tool, for a per-tool budget (ADS plan
        section 9.2/W4-04) distinct from ``count_for_run``'s whole-run cap —
        e.g. a decision_support run may search several times within its
        overall step budget, but only up to its own, tighter query budget."""
        return int(
            (
                await self._conn.execute(
                    select(func.count())
                    .select_from(tool_calls)
                    .where(
                        and_(
                            tool_calls.c.run_id == run_id,
                            tool_calls.c.tool_name == tool_name,
                            tool_calls.c.status != "denied",
                        )
                    )
                )
            ).scalar()
            or 0
        )

    async def duplicate_count(self, run_id: str, tool_name: str, arguments_sha256: str) -> int:
        return int(
            (
                await self._conn.execute(
                    select(func.count())
                    .select_from(tool_calls)
                    .where(
                        and_(
                            tool_calls.c.run_id == run_id,
                            tool_calls.c.tool_name == tool_name,
                            tool_calls.c.arguments_sha256 == arguments_sha256,
                            tool_calls.c.status != "denied",
                        )
                    )
                )
            ).scalar()
            or 0
        )

    async def list_for_run(self, run_id: str) -> Sequence[dict[str, Any]]:
        rows = (
            await self._conn.execute(
                select(tool_calls)
                .where(tool_calls.c.run_id == run_id)
                .order_by(tool_calls.c.started_at)
            )
        ).mappings().all()
        # Unlike every other repository, these rows go straight out as raw
        # dicts rather than through a row_to_* mapper — so they were the one
        # place SQLite's naive (no-tzinfo) datetimes reached a client
        # unnormalized, rendering as local time in whatever timezone the
        # browser happened to be in instead of UTC.
        results = [dict(r) for r in rows]
        for row in results:
            row["started_at"] = m.utc(row["started_at"])
            row["ended_at"] = m.utc(row["ended_at"])
        return results


class SqlCapabilityTokenStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def issue(
        self, *, jti: str, session_id: str, run_id: str, runtime_binding_id: str | None,
        allowed_tools: list[str], issued_at: datetime, expires_at: datetime,
    ) -> None:
        await self._conn.execute(
            insert(capability_tokens).values(
                jti=jti, session_id=session_id, run_id=run_id,
                runtime_binding_id=runtime_binding_id, allowed_tools=allowed_tools,
                issued_at=issued_at, expires_at=expires_at,
            )
        )

    async def is_valid(self, jti: str, *, now: datetime) -> bool:
        row = (
            await self._conn.execute(
                select(capability_tokens).where(capability_tokens.c.jti == jti)
            )
        ).mappings().first()
        if row is None or row["revoked_at"] is not None:
            return False
        return m.utc(row["expires_at"]) > now

    async def revoke(self, jti: str, *, now: datetime) -> None:
        await self._conn.execute(
            update(capability_tokens)
            .where(capability_tokens.c.jti == jti)
            .values(revoked_at=now)
        )
