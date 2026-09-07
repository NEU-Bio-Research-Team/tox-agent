"""Runs execute in the background; this owns the tasks that do it.

``POST /messages`` answers 202 with a run id, so the work happens after the
response. Two things follow from that and are handled here rather than in each
handler: a run that raises must still reach a terminal state with a typed
failure code and an event, and a cancellation request must be able to reach the
task that is actually doing the work.

Cancellation reports what it did, never what the caller hoped for (plan section
6.5). An in-process task can genuinely be cancelled; a runtime turn usually
cannot, and saying so is the point.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
    threshold_overrides: Mapping[str, Any] | None = None
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


class RunScheduler:
    def __init__(self, database, handlers: Mapping[Intent, RunHandler] | None = None) -> None:
        self._db = database
        self._handlers: dict[Intent, RunHandler] = dict(handlers or {})
        self._tasks: dict[str, asyncio.Task] = {}

    def register(self, intent: Intent, handler: RunHandler) -> None:
        self._handlers[intent] = handler

    def handles(self, intent: Intent) -> bool:
        return intent in self._handlers

    def submit(self, context: RunContext) -> None:
        task = asyncio.create_task(self._execute(context), name=f"run:{context.run_id}")
        self._tasks[context.run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(context.run_id, None))

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
            await self._terminate(context, RunStatus.CANCELLED, "cancelled")
            raise
        except ToxAgentError as exc:
            await self._fail(context, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 — a run must not end in limbo
            log.exception("run %s failed unexpectedly", context.run_id)
            await self._fail(context, "internal_error", type(exc).__name__)

    # --- terminal states ---------------------------------------------------

    async def _fail(self, context: RunContext, code: str, message: str) -> None:
        recovery = await self._terminate(context, RunStatus.FAILED, code, message)
        if recovery is not None:
            self.submit(recovery)

    async def _terminate(
        self, context: RunContext, status: RunStatus, code: str, message: str = ""
    ) -> RunContext | None:
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
                await uow.commit()
                if recovery is None:
                    return None
                return RunContext(
                    actor=context.actor,
                    session_id=context.session_id,
                    run_id=recovery.id,
                    intent=context.intent,
                    text=context.text,
                    smiles=context.smiles,
                    batch_smiles=context.batch_smiles,
                    endpoints=context.endpoints,
                    threshold_overrides=context.threshold_overrides,
                    analysis_id=context.analysis_id,
                    # Any deterministic snapshot/observation work completed
                    # before the loss remains product-owned.  A recovery must
                    # look it up rather than dispatching it again.
                    needs_snapshot_first=False,
                    language=context.language,
                    attachment_id=context.attachment_id,
                )
        except Exception:  # noqa: BLE001
            log.exception("could not record the terminal state of run %s", context.run_id)
        return None

    # --- cancellation ------------------------------------------------------

    async def cancel(self, run_id: str, *, runtime_cancel_supported: bool = False) -> CancelOutcome:
        async with self._db.unit_of_work() as uow:
            requested = await uow.runs.request_cancel(run_id)
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
        return CancelOutcome(
            run_id, True, runtime_cancel_supported, "cancellation_recorded_no_local_worker"
        )

    async def drain(self, timeout: float = 10.0) -> None:
        tasks = [t for t in self._tasks.values() if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

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
