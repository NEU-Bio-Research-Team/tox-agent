"""What a starting process may do about runs it did not start.

The original answer was "fail all of them", and it rested on an assumption
stated in its own docstring: `RunScheduler._tasks` is empty at startup, so a
non-terminal run must have been left by a process that no longer exists. That
is true of exactly one deployment — a single process that owns the whole
database — and false of every rolling deploy and every scale-out, where the
new replica starts while the old one is still executing runs. There it did not
reconcile orphans; it killed live work, and marked it `potentially_billed`
while the provider turn it was describing was still in flight (I17).

Liveness cannot be inferred from "this process does not know about it". It has
to be asserted by the process doing the work, which is what the lease in
`run_jobs` is: renewed every few seconds while a worker executes a run, and
carrying a fencing epoch so a worker that was slow rather than dead cannot
write over its successor. This module therefore only ever considers runs whose
lease has expired or that never had one.

Two kinds of unowned run, with different honest answers:

- **It has a durable envelope.** The request is still on record, so the run can
  be executed, and `RunScheduler.adopt()` does that. Nothing here needs to
  close it out — a rolling restart should finish accepted work, not fail it.
- **It has none.** Runs from before this schema, and runs whose submitter has
  not been migrated to `enqueue`. There is no record of what the run was for,
  so it is closed out exactly as before: a run that cannot be executed and
  cannot be cancelled is worse left open, because it holds its session under
  the one-active-run cap forever.
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
    """Close out non-terminal runs that no worker owns and none could execute.

    Returns how many were reconciled. A run under a live lease, and a run whose
    durable envelope makes it adoptable, are both left alone — the first
    belongs to another replica, the second to `RunScheduler.adopt()`.
    """
    reconciled = 0
    now = _now()
    async with db.unit_of_work() as uow:
        orphans = await uow.runs.list_non_terminal()
        for run in orphans:
            job = await uow.run_jobs.get(run.id)
            if job is not None:
                lease_expires_at = job.get("lease_expires_at")
                if lease_expires_at is not None and _as_utc(lease_expires_at) > now:
                    # Another worker is executing this right now.
                    continue
                # Expired lease, but the envelope survives: adoptable, and
                # failing it here would throw away work this deployment can
                # still finish.
                continue
            reconciled += 1
            await _fail_orphan(uow, run)
        if reconciled:
            await uow.commit()
    return reconciled


def _as_utc(value: datetime) -> datetime:
    """SQLite hands back naive datetimes for a timezone-aware column.

    Comparing one of those against an aware `now` raises, which would turn a
    reconciliation sweep into a failed startup. Naive values from this column
    were written as UTC.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


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
