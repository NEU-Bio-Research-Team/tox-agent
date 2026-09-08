"""Run state transitions against the store.

One helper, used everywhere, because the mistake it prevents is easy and quiet:
a run that moves through two states in one transaction has advanced its version
twice in memory while the row has still only been written once, so the optimistic
``expected_version`` is the version that was *read*, never the one the last
transition produced.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..domain.events import EventType
from ..domain.run import Run, RunStatus

_TERMINAL_EVENT = {
    RunStatus.COMPLETED: EventType.RUN_COMPLETED,
    RunStatus.FAILED: EventType.RUN_FAILED,
    RunStatus.CANCELLED: EventType.RUN_CANCELLED,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def advance(
    uow,
    run: Run,
    target: RunStatus,
    *,
    failure_code: str | None = None,
    runtime_binding_id: str | None = None,
    potentially_billed: bool | None = None,
    payload: dict[str, Any] | None = None,
    emit: bool = True,
) -> Run:
    """Move a run to ``target``, filling in any intermediate state it needs.

    Returns the run as persisted. Emits the matching event unless the caller is
    composing several changes and will emit one of its own.
    """
    expected_version = run.version
    now = _now()
    current = run
    if target in (RunStatus.COMPLETED, RunStatus.VALIDATING) and current.status is RunStatus.QUEUED:
        current = current.transition(RunStatus.RUNNING, now=now)
    if target is RunStatus.FAILED and current.status is RunStatus.QUEUED:
        current = current.transition(RunStatus.RUNNING, now=now)

    updated = current.transition(
        target, now=now, failure_code=failure_code, runtime_binding_id=runtime_binding_id,
        potentially_billed=potentially_billed,
    )
    await uow.runs.update(updated, expected_version=expected_version)

    if emit:
        event = _TERMINAL_EVENT.get(target)
        if event is None:
            event = {
                RunStatus.RUNNING: EventType.RUN_STARTED,
                RunStatus.VALIDATING: EventType.RUN_VALIDATING,
            }[target]
        body = dict(payload or {})
        if failure_code:
            body["failure_code"] = failure_code
        uow.emit(
            session_id=updated.session_id, type=event, entity_type="run",
            entity_id=updated.id, run_id=updated.id, entity_version=updated.version,
            payload=body,
        )
    return updated
