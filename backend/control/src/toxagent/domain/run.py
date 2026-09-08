"""Run — one unit of product work, and the state machine that bounds it.

Plan section 5.3. The transition table is the whole point: a run that has
failed, been cancelled, or completed never becomes ``running`` again. Recovery
is a *new* run pointing at the old one (PROD-10), so a client that already saw
text from a failed attempt cannot have more text quietly appended to it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Final

from .ids import MESSAGE, RUN, RUNTIME_BINDING, SESSION, new_id, require_id


class Lane(str, Enum):
    DETERMINISTIC = "deterministic"
    AGENTIC = "agentic"
    MIXED = "mixed"


class Intent(str, Enum):
    ANALYSIS = "analysis"
    ANALYSIS_BATCH = "analysis_batch"
    REPORT_QA = "report_qa"
    EVIDENCE_RESEARCH = "evidence_research"
    ATTRIBUTION = "attribution"
    STRUCTURE_RECOGNITION = "structure_recognition"
    CLARIFICATION_REQUIRED = "clarification_required"
    OUT_OF_SCOPE = "out_of_scope"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL: Final[frozenset[RunStatus]] = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
)

ALLOWED_TRANSITIONS: Final[dict[RunStatus, frozenset[RunStatus]]] = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.VALIDATING, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.VALIDATING: frozenset({RunStatus.COMPLETED, RunStatus.FAILED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

DEFAULT_DEADLINE = timedelta(minutes=5)


class IllegalTransition(ValueError):
    def __init__(self, current: RunStatus, target: RunStatus) -> None:
        super().__init__(f"run cannot go {current.value} -> {target.value}")
        self.current, self.target = current, target


@dataclass(frozen=True, slots=True)
class Run:
    id: str
    session_id: str
    trigger_message_id: str
    lane: Lane
    intent: Intent
    status: RunStatus
    deadline_at: datetime
    created_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    runtime_binding_id: str | None = None
    failure_code: str | None = None
    recovery_of_run_id: str | None = None
    potentially_billed: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        require_id(self.id, RUN, field="run.id")
        require_id(self.session_id, SESSION, field="run.session_id")
        require_id(self.trigger_message_id, MESSAGE, field="run.trigger_message_id")
        if self.runtime_binding_id is not None:
            require_id(self.runtime_binding_id, RUNTIME_BINDING, field="run.runtime_binding_id")
        if self.recovery_of_run_id is not None:
            require_id(self.recovery_of_run_id, RUN, field="run.recovery_of_run_id")
        if self.lane is Lane.DETERMINISTIC and self.runtime_binding_id is not None:
            raise ValueError("a deterministic run must not bind a model runtime (lane D)")

    @classmethod
    def create(
        cls,
        session_id: str,
        trigger_message_id: str,
        lane: Lane,
        intent: Intent,
        *,
        now: datetime,
        deadline: timedelta = DEFAULT_DEADLINE,
        recovery_of_run_id: str | None = None,
    ) -> "Run":
        return cls(
            id=new_id(RUN),
            session_id=session_id,
            trigger_message_id=trigger_message_id,
            lane=lane,
            intent=intent,
            status=RunStatus.QUEUED,
            deadline_at=now + deadline,
            created_at=now,
            recovery_of_run_id=recovery_of_run_id,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    def mark_potentially_billed(self) -> "Run":
        """Plan section 6.6 / remaining-plan W2-12: the runtime confirmed
        receiving this turn (a provider request was actually sent and
        accepted) but the run did not reach a completion with known usage —
        the charge outcome is undetermined, not "nothing was spent". For a
        *standalone* write with no status change alongside it (harness/
        gateway.py's own quiet, best-effort persistence): it bumps version
        itself, so the caller's ``expected_version`` must be this run's
        version *before* calling this. Do not also pass the result through
        ``transition()`` in the same logical write — that bumps version
        again for a write that never actually happens twice; a status
        change and this flag landing together belong in one ``transition()``
        call via its own ``potentially_billed`` parameter instead. Idempotent
        — a run already flagged does not bump version again for no reason."""
        if self.potentially_billed:
            return self
        return replace(self, potentially_billed=True, version=self.version + 1)

    def transition(
        self,
        target: RunStatus,
        *,
        now: datetime,
        failure_code: str | None = None,
        runtime_binding_id: str | None = None,
        potentially_billed: bool | None = None,
    ) -> "Run":
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise IllegalTransition(self.status, target)
        if target in (RunStatus.FAILED,) and not failure_code:
            raise ValueError("a failed run must carry a failure_code")
        return replace(
            self,
            potentially_billed=(
                self.potentially_billed if potentially_billed is None else potentially_billed
            ),
            status=target,
            started_at=now if target is RunStatus.RUNNING else self.started_at,
            ended_at=now if target in TERMINAL else self.ended_at,
            failure_code=failure_code or self.failure_code,
            runtime_binding_id=runtime_binding_id or self.runtime_binding_id,
            version=self.version + 1,
        )

    def expired(self, now: datetime) -> bool:
        return not self.is_terminal and now >= self.deadline_at
