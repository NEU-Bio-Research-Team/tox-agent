"""Runs execute in the background; this owns the tasks that do it.

``POST /messages`` answers 202 with a run id, so the work happens after the
response. Two things follow from that and are handled here rather than in each
handler: a run that raises must still reach a terminal state with a typed
failure code and an event, and a cancellation request must be able to reach the
task that is actually doing the work.

Cancellation reports what it did, never what the caller hoped for (plan section
6.5). An in-process task can genuinely be cancelled; a runtime turn usually
cannot, and saying so is the point.

I17/I18: the task map above is this process's memory, and a second replica has
its own. Two consequences used to follow, both wrong:

- Starting a process meant "every non-terminal run in the database was left by
  a dead process", which is true of exactly one deployment — a single process
  that owns the whole database. On a rolling deploy it made the *new* replica
  fail runs the *old* one was still executing.
- A cancellation requested through a replica that did not hold the run wrote
  the flag and stopped. No worker was watching it, so the run continued to
  completion and the caller was told a cancellation had been recorded.

Ownership is therefore a lease held in `run_jobs`, renewed while the worker
lives, and carrying a fencing epoch. A worker executes a run only while it
holds that run's lease; a lease that stops being renewed is what makes a run
adoptable, and adoption re-executes it from the durable envelope rather than
closing it out. Cancellation reaches the owner because the owner watches for
it, not because the canceller happened to be the right process.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping

from ..domain.errors import ToxAgentError
from ..domain.message import Message, PartType, Role
from ..domain.events import EventType
from ..domain.run import Intent, Lane, Run, RunStatus
from .policy import Actor
from .runs import advance

log = logging.getLogger("toxagent.runs")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RunContext:
    actor: Actor
    session_id: str
    run_id: str
    intent: Intent
    text: str = ""
    smiles: str | None = None
    batch_smiles: tuple[str, ...] = ()
    endpoints: tuple[str, ...] | None = None
    model_selection: Mapping[str, str] | None = None
    ai_profile_id: str | None = None
    threshold_overrides: Mapping[str, Any] | None = None
    explanation_mode: str = "on_demand"
    explanation_targets: tuple[tuple[str, str | None], ...] = ()
    analysis_id: str | None = None
    needs_snapshot_first: bool = False
    language: str = "en"
    #: remaining-plan W4-07: the uploaded image's bytes now live in the
    #: object store, addressed by this attachment row — never carried here
    #: in memory. RecognizeStructure reads them back through
    #: AttachmentStore + ObjectStore, scoped to this same actor, so a
    #: recovery run (a fresh RunContext, same attachment_id) can still reach
    #: them after a control-plane restart, unlike the old in-memory bytes.
    attachment_id: str | None = None


# Imported after `RunContext`: the codec is defined in terms of it, and takes
# the reverse import lazily so either module can be imported first.
from .run_envelope import UnreadableEnvelope, from_envelope, to_envelope  # noqa: E402

RunHandler = Callable[[RunContext], Awaitable[None]]


@dataclass
class CancelOutcome:
    """The honest answer to "cancel this" (plan section 6.5)."""

    run_id: str
    requested: bool
    runtime_cancel_supported: bool
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "requested": self.requested,
            "runtime_cancel_supported": self.runtime_cancel_supported,
            "action": self.action,
        }


#: How long a claim is good for without a renewal. Long enough that a worker
#: paused by GC, a slow query or a busy host is not declared dead; short
#: enough that a genuinely dead worker's runs are picked up while the person
#: who submitted them is still watching. Renewal happens at a third of it, so
#: two renewals must be missed before anyone else can claim.
LEASE_TTL_S = 30.0

#: How often a worker asks whether one of its own runs has been cancelled
#: through another replica. The delay this adds is bounded and visible; the
#: alternative — a database notification — is a second delivery mechanism to
#: get right, and `cancel()` already answers honestly about what it did.
CANCEL_POLL_S = 2.0


class RunScheduler:
    def __init__(
        self,
        database,
        handlers: Mapping[Intent, RunHandler] | None = None,
        *,
        worker_id: str | None = None,
    ) -> None:
        self._db = database
        self._handlers: dict[Intent, RunHandler] = dict(handlers or {})
        self._tasks: dict[str, asyncio.Task] = {}
        # Which process this is. Unique per scheduler instance, not per
        # deployment: two schedulers in one process (a test, an embedded
        # worker) must be able to fence each other exactly as two processes
        # would, and a hostname would make them indistinguishable.
        self._worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        #: Epoch this worker holds for each run it is executing. The value it
        #: must present to renew or release; losing the comparison means it
        #: has been fenced.
        self._epochs: dict[str, int] = {}
        self._supervisors: dict[str, asyncio.Task] = {}
        #: Runs this worker was fenced off. It must not write their
        #: terminal state — another worker owns them now.
        self._fenced: set[str] = set()

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def register(self, intent: Intent, handler: RunHandler) -> None:
        self._handlers[intent] = handler

    def handles(self, intent: Intent) -> bool:
        return intent in self._handlers

    async def enqueue(self, uow, context: RunContext, *, now: datetime | None = None) -> int:
        """Write this run's durable execution record, claimed by this worker.

        Called inside the transaction that creates the run, so the two commit
        together: there is no moment at which an accepted run exists with no
        record of what it was asked to do. `submit()` then starts the local
        task under the epoch returned here.
        """
        moment = now or _now()
        return await uow.run_jobs.enqueue(
            context.run_id,
            to_envelope(context),
            worker_id=self._worker_id,
            lease_expires_at=moment + timedelta(seconds=LEASE_TTL_S),
            now=moment,
        )

    def submit(self, context: RunContext, *, epoch: int | None = None) -> None:
        """Start executing a run this worker holds the lease for.

        `epoch` is what `enqueue`/`adopt` returned. None means no durable job
        row exists for this run — a caller that has not been migrated to
        `enqueue`, or a test driving the scheduler directly. Such a run still
        executes, and still reaches a terminal state; what it does not get is
        adoption by another worker if this process dies, because nothing was
        written down for another worker to adopt. That distinction is visible
        in `startup_reconciliation`, which closes those runs out.
        """
        if epoch is not None:
            self._epochs[context.run_id] = epoch
        task = asyncio.create_task(self._execute(context), name=f"run:{context.run_id}")
        self._tasks[context.run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(context.run_id, None))
        if epoch is not None:
            supervisor = asyncio.create_task(
                self._supervise(context.run_id, task), name=f"lease:{context.run_id}"
            )
            self._supervisors[context.run_id] = supervisor
            supervisor.add_done_callback(lambda _: self._supervisors.pop(context.run_id, None))

    async def adopt(self, limit: int = 20) -> int:
        """Take over runs whose owner stopped renewing, and execute them.

        This is what makes an accepted run survive the process that accepted
        it. A live lease is never touched: a job is adoptable only once its
        lease has expired, and the claim is conditional on the epoch that was
        read, so two workers sweeping at once produce one owner each.
        """
        adopted = 0
        now = _now()
        async with self._db.unit_of_work() as uow:
            candidates = await uow.run_jobs.claimable(now=now, limit=limit)
        for job in candidates:
            if job["run_id"] in self._tasks:
                continue
            try:
                context = from_envelope(job["envelope"])
            except UnreadableEnvelope as exc:
                # Written by a build this one does not understand. Leaving it
                # is correct: a worker that does understand it can still take
                # it, and guessing at the request would be worse than waiting.
                log.warning("run %s has an envelope this build cannot read: %s", job["run_id"], exc)
                continue
            async with self._db.unit_of_work() as uow:
                run = await uow.runs.get(job["run_id"])
                if run is None or run.is_terminal:
                    # Its owner finished it and died before releasing the job.
                    await uow.run_jobs.discard(job["run_id"])
                    await uow.commit()
                    continue
                epoch = await uow.run_jobs.claim(
                    job["run_id"], worker_id=self._worker_id,
                    expected_epoch=job["lease_epoch"],
                    lease_expires_at=now + timedelta(seconds=LEASE_TTL_S), now=now,
                )
                if epoch is None:
                    continue
                await uow.commit()
            log.info("adopted orphaned run %s at epoch %d", context.run_id, epoch)
            self.submit(context, epoch=epoch)
            adopted += 1
        return adopted

    async def _supervise(self, run_id: str, task: asyncio.Task) -> None:
        """Hold the lease while the run executes, and watch for a cancellation.

        Three jobs, one loop, because they share a cadence and a lifetime:

        - Renew, so no other worker adopts a run that is being executed.
        - Notice being fenced. A renewal that finds the lease gone means
          something else has already claimed this run — probably because this
          process was unreachable long enough to look dead. Two workers on one
          run is the failure this exists to prevent, so the local task is
          cancelled rather than allowed to finish and commit.
        - Notice a cancellation requested through another replica, which
          writes the flag and has no task of its own to cancel (I18).
        """
        deadline_missed = False
        try:
            while not task.done():
                await asyncio.sleep(CANCEL_POLL_S)
                if task.done():
                    return
                now = _now()
                epoch = self._epochs.get(run_id)
                if epoch is None:
                    return
                async with self._db.unit_of_work() as uow:
                    held = await uow.run_jobs.renew(
                        run_id, worker_id=self._worker_id, epoch=epoch,
                        lease_expires_at=now + timedelta(seconds=LEASE_TTL_S), now=now,
                    )
                    cancelled = await uow.runs.cancel_requested(run_id) if held else False
                    await uow.commit()
                if not held:
                    log.warning(
                        "run %s was claimed by another worker; stopping the local task", run_id
                    )
                    deadline_missed = True
                    self._fenced.add(run_id)
                    task.cancel()
                    return
                if cancelled:
                    log.info("run %s was cancelled elsewhere; stopping the local task", run_id)
                    task.cancel()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a supervisor must not kill the run
            log.exception("lease supervision failed for run %s", run_id)
        finally:
            if deadline_missed:
                # Fenced: the epoch is no longer ours, so `release` would
                # refuse anyway. Drop the local bookkeeping and leave the row
                # to its new owner.
                self._epochs.pop(run_id, None)

    async def _release(self, run_id: str) -> None:
        """Give up the lease once the run is terminal."""
        epoch = self._epochs.pop(run_id, None)
        if epoch is None:
            return
        try:
            async with self._db.unit_of_work() as uow:
                await uow.run_jobs.release(run_id, worker_id=self._worker_id, epoch=epoch)
                await uow.commit()
        except Exception:  # noqa: BLE001 — an unreleased lease expires by itself
            log.exception("could not release the lease on run %s", run_id)

    async def _execute(self, context: RunContext) -> None:
        handler = self._handlers.get(context.intent)
        if handler is None:
            await self._fail(
                context, "runtime_unavailable",
                f"no handler is registered for {context.intent.value} in this deployment",
            )
            return
        try:
            await handler(context)
        except asyncio.CancelledError:
            if context.run_id in self._fenced:
                # Cancelled because another worker took this run over, not
                # because anyone asked for it to stop. Writing `cancelled`
                # here would terminate a run that is being executed right now
                # by its new owner — the exact double-ownership the lease
                # exists to prevent, arriving through the loser's error path.
                log.warning(
                    "run %s was taken over; leaving its outcome to the new owner",
                    context.run_id,
                )
                raise
            await self._terminate(context, RunStatus.CANCELLED, "cancelled")
            raise
        except ToxAgentError as exc:
            await self._fail(context, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 — a run must not end in limbo
            log.exception("run %s failed unexpectedly", context.run_id)
            await self._fail(context, "internal_error", type(exc).__name__)
        finally:
            # The run is terminal one way or another by here, so nothing is
            # left for another worker to adopt. Releasing keeps the sweep from
            # re-reading a finished job for a whole lease period; failing to
            # release is survivable, because the lease expires and `adopt`
            # discards jobs whose run is already terminal.
            await self._release(context.run_id)
            self._fenced.discard(context.run_id)

    # --- terminal states ---------------------------------------------------

    async def _fail(self, context: RunContext, code: str, message: str) -> None:
        recovered = await self._terminate(context, RunStatus.FAILED, code, message)
        if recovered is not None:
            recovery, epoch = recovered
            self.submit(recovery, epoch=epoch)

    async def _terminate(
        self, context: RunContext, status: RunStatus, code: str, message: str = ""
    ) -> tuple[RunContext, int] | None:
        try:
            async with self._db.unit_of_work() as uow:
                run = await uow.runs.get(context.run_id)
                if run is None or run.is_terminal:
                    return None
                await advance(
                    uow, run, status,
                    # A cancelled run has no failure; recording one would make
                    # "the user stopped it" read as "it broke".
                    failure_code=code if status is RunStatus.FAILED else None,
                    payload={"message": message, "reason": code},
                )

                sequence = await uow.messages.next_sequence(context.session_id)
                notice = Message.create(
                    context.session_id, Role.SYSTEM_EVENT, sequence, now=_now(),
                    parts=(
                        (
                            PartType.ERROR,
                            {"code": code, "message": message, "run_id": context.run_id},
                        ),
                    ),
                )
                await uow.messages.add(notice)
                recovery: Run | None = None
                if _can_recover_runtime_loss(run, status=status, failure_code=code):
                    # PROD-10: never resume a terminal run.  This is a new
                    # auditable entity and deliberately has no assistant text
                    # appended to the failed run's transcript.
                    recovery = Run.create(
                        run.session_id,
                        run.trigger_message_id,
                        run.lane,
                        run.intent,
                        now=_now(),
                        recovery_of_run_id=run.id,
                    )
                    await uow.runs.add(recovery)
                    uow.emit(
                        session_id=run.session_id,
                        type=EventType.RUN_QUEUED,
                        entity_type="run",
                        entity_id=recovery.id,
                        run_id=recovery.id,
                        payload={
                            "intent": recovery.intent.value,
                            "lane": recovery.lane.value,
                            "recovery_of_run_id": run.id,
                        },
                    )
                    uow.emit(
                        session_id=run.session_id,
                        type=EventType.RUNTIME_RECOVERY_STARTED,
                        entity_type="run",
                        entity_id=recovery.id,
                        run_id=recovery.id,
                        payload={
                            "recovery_of_run_id": run.id,
                            "failure_code": code,
                            "reuses_product_observations": True,
                        },
                    )
                if recovery is None:
                    await uow.commit()
                    return None
                recovery_context = RunContext(
                    actor=context.actor,
                    session_id=context.session_id,
                    run_id=recovery.id,
                    intent=context.intent,
                    text=context.text,
                    smiles=context.smiles,
                    batch_smiles=context.batch_smiles,
                    endpoints=context.endpoints,
                    model_selection=context.model_selection,
                    ai_profile_id=context.ai_profile_id,
                    threshold_overrides=context.threshold_overrides,
                    explanation_mode=context.explanation_mode,
                    explanation_targets=context.explanation_targets,
                    analysis_id=context.analysis_id,
                    # Any deterministic snapshot/observation work completed
                    # before the loss remains product-owned.  A recovery must
                    # look it up rather than dispatching it again.
                    needs_snapshot_first=False,
                    language=context.language,
                    attachment_id=context.attachment_id,
                )
                # In the same transaction as the recovery run itself: a
                # recovery created but never enqueued would be the original
                # loss again, one run further along (I18).
                epoch = await self.enqueue(uow, recovery_context)
                await uow.commit()
                return recovery_context, epoch
        except Exception:  # noqa: BLE001
            log.exception("could not record the terminal state of run %s", context.run_id)
        return None

    # --- cancellation ------------------------------------------------------

    async def cancel(self, run_id: str, *, runtime_cancel_supported: bool = False) -> CancelOutcome:
        async with self._db.unit_of_work() as uow:
            requested = await uow.runs.request_cancel(run_id)
            job = await uow.run_jobs.get(run_id) if requested else None
            await uow.commit()

        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            # Wait for the task to actually unwind before answering. Reporting a
            # cancellation while the worker is still writing would be the exact
            # lie plan section 6.5 forbids.
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            return CancelOutcome(run_id, requested, runtime_cancel_supported, "worker_cancelled")

        if not requested:
            return CancelOutcome(run_id, False, runtime_cancel_supported, "run_already_terminal")
        if job is not None and job.get("worker_id") and job["worker_id"] != self._worker_id:
            # Another replica holds the lease, and its supervisor polls this
            # flag (I18). Saying so is different from the old answer, which
            # named the absence of a *local* worker and left the caller to
            # infer, wrongly, that nothing would act on the request.
            return CancelOutcome(
                run_id, True, runtime_cancel_supported, "cancellation_relayed_to_owning_worker"
            )
        return CancelOutcome(
            run_id, True, runtime_cancel_supported, "cancellation_recorded_no_local_worker"
        )

    async def drain(self, timeout: float = 10.0) -> None:
        supervisors = [t for t in self._supervisors.values() if not t.done()]
        for supervisor in supervisors:
            supervisor.cancel()
        tasks = [t for t in self._tasks.values() if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
        if supervisors:
            await asyncio.wait(supervisors, timeout=timeout)

    @property
    def in_flight(self) -> int:
        return len([t for t in self._tasks.values() if not t.done()])


def _can_recover_runtime_loss(run: Run, *, status: RunStatus, failure_code: str) -> bool:
    """The bounded automatic recovery policy (plan section 7.4).

    A health probe that fails before a binding exists has nothing to recover;
    a scheduler must not turn it into a hidden retry loop.  Conversely, a
    binding that was already created is proof that a runtime-local transcript
    may have vanished, so a single explicit recovery run is warranted.
    """
    return (
        status is RunStatus.FAILED
        and failure_code == "runtime_unavailable"
        and run.lane in {Lane.AGENTIC, Lane.MIXED}
        and run.runtime_binding_id is not None
        and run.recovery_of_run_id is None
    )
