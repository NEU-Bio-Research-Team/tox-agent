"""Startup reconciliation for runs orphaned by a crash or a `kill -9`.

``RunScheduler._tasks`` lives only in process memory, so the instant a new
process starts, it is empty — meaning every run this database still has
sitting in ``queued``/``running``/``validating`` was left there by a process
that is now gone. A clean shutdown already drains every in-flight task via
``RunScheduler.drain()`` (each task's own cancellation path marks it
``cancelled`` honestly), so a non-terminal run found here can only mean an
unclean exit: nothing else in this process could have created one.

Left alone, such a run blocks its session forever under the one-active-run
cap and makes ``cancel()`` a permanent no-op (there is no worker left to act
on the flag). Replaying the original request would need the actor, text and
SMILES it was submitted with, none of which this table retains — so this
reconciliation does not attempt automatic recovery. It closes the run out
honestly instead, the same way any other unrecoverable runtime loss does.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..domain.message import Message, PartType, Role
from ..domain.run import Run, RunStatus
from .runs import advance

log = logging.getLogger("toxagent.startup")


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def reconcile_orphaned_runs(db) -> int:
    """Fail every non-terminal run left behind by a previous process. Returns
    how many were reconciled."""
    reconciled = 0
    async with db.unit_of_work() as uow:
        orphans = await uow.runs.list_non_terminal()
        for run in orphans:
            reconciled += 1
            await _fail_orphan(uow, run)
        if reconciled:
            await uow.commit()
    return reconciled


async def _fail_orphan(uow, run: Run) -> None:
    was_cancel_requested = await uow.runs.cancel_requested(run.id)
    message = (
        "the control plane restarted while this run was in flight and a cancellation "
        "had already been requested for it; no worker survived to honour either"
        if was_cancel_requested
        else "the control plane restarted while this run was in flight; no worker "
        "survived to complete it"
    )
    # plan section 6.6 / remaining-plan W2-12: a runtime_binding_id means
    # AgentRuntimeGateway.execute got at least as far as creating the
    # binding, which happens before it sends the turn — the crash could have
    # landed anywhere from "about to send" to "provider mid-response". There
    # is no persisted signal finer than this to tell those apart (the
    # in-process receipt.accepted that would say so precisely dies with the
    # crashed process), so this deliberately over-approximates in the safe
    # direction: a run this reconciliation cannot rule out as having reached
    # the provider is marked potentially_billed, rather than defaulting to
    # "no" for lack of proof otherwise. A run that never got a binding
    # (queued, never scheduled) is untouched. Folded into this single
    # advance() call via transition()'s own parameter, not
    # Run.mark_potentially_billed() — that bumps version for a standalone
    # write, which would desync advance()'s own expected_version tracking if
    # composed with the status change below instead of replacing it.
    await advance(
        uow, run, RunStatus.FAILED,
        failure_code="runtime_unavailable",
        potentially_billed=True if run.runtime_binding_id is not None else None,
        payload={"message": message, "reason": "startup_reconciliation"},
    )
    sequence = await uow.messages.next_sequence(run.session_id)
    notice = Message.create(
        run.session_id, Role.SYSTEM_EVENT, sequence, now=_now(),
        parts=(
            (PartType.ERROR, {"code": "runtime_unavailable", "message": message, "run_id": run.id}),
        ),
    )
    await uow.messages.add(notice)
    # No dedicated MESSAGE_CREATED event, matching RunScheduler._terminate's
    # own system_event notice: the RUN_FAILED event `advance` already emitted
    # is what a client's `run.failed` handler already treats as "refetch
    # messages for this session," which is exactly what surfaces this notice.
