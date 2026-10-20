"""State machine for activity-guided toxicity screening.

Activity values and prioritisation policy live in their own modules. Imports
below intentionally re-export the original public surface so existing callers
of ``toxagent.domain.screening`` remain compatible after the split.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Any, Final

from .activity import (
    ActivityApplicability,
    ActivityEstimate,
    ActivityIdentityMismatch,
    ActivityModelProvenance,
    ActivitySnapshot,
    ActivitySubject,
    TargetIdentity,
    TargetRole,
)
from .ids import (
    ACTIVITY_SNAPSHOT,
    ANALYSIS,
    PRIORITISATION,
    RUN,
    SCREENING,
    SESSION,
    new_id,
    require_id,
)
from .prioritisation import (
    PolicyNotApplicable,
    PrioritisationDecision,
    PriorityStatus,
    TargetPriorityPolicy,
    UnsupportedTargetRole,
)

__all__ = [
    "ActivityApplicability",
    "ActivityEstimate",
    "ActivityGuidedScreening",
    "ActivityIdentityMismatch",
    "ActivityModelProvenance",
    "ActivitySnapshot",
    "ActivitySubject",
    "IllegalScreeningTransition",
    "PolicyNotApplicable",
    "PrioritisationDecision",
    "PriorityStatus",
    "ScreeningStatus",
    "TargetIdentity",
    "TargetPriorityPolicy",
    "TargetRole",
    "UnsupportedTargetRole",
]


class ScreeningStatus(str, Enum):
    CREATED = "created"
    ACTIVITY_RUNNING = "activity_running"
    ACTIVITY_COMPLETED = "activity_completed"
    ACTIVITY_FAILED = "activity_failed"
    PRIORITISED = "prioritised"
    REVIEW_REQUIRED = "review_required"
    LOW_PRIORITY = "low_priority"
    TOXICITY_RUNNING = "toxicity_running"
    TOXICITY_FAILED = "toxicity_failed"
    COMPLETED = "completed"
    COMPLETED_WITH_TOXICITY_NOT_RUN = "completed_with_toxicity_not_run"


TERMINAL_SCREENING_STATUSES: Final[frozenset[ScreeningStatus]] = frozenset(
    {
        ScreeningStatus.ACTIVITY_FAILED,
        ScreeningStatus.TOXICITY_FAILED,
        ScreeningStatus.COMPLETED,
        ScreeningStatus.COMPLETED_WITH_TOXICITY_NOT_RUN,
    }
)


class IllegalScreeningTransition(ValueError):
    def __init__(self, current: ScreeningStatus, action: str) -> None:
        super().__init__(f"screening in {current.value!r} cannot {action}")
        self.current = current
        self.action = action


@dataclass(frozen=True, slots=True)
class ActivityGuidedScreening:
    id: str
    session_id: str
    run_id: str
    input_smiles: str
    subject: ActivitySubject
    intended_target_role: TargetRole
    status: ScreeningStatus
    created_at: datetime
    updated_at: datetime
    activity_snapshot_id: str | None = None
    prioritisation_decision_id: str | None = None
    toxicity_analysis_id: str | None = None
    toxicity_authorised_by: str | None = None
    toxicity_authorisation_reason: str | None = None
    toxicity_not_run_reason: str | None = None
    failure_code: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        require_id(self.id, SCREENING, field="screening.id")
        require_id(self.session_id, SESSION, field="screening.session_id")
        require_id(self.run_id, RUN, field="screening.run_id")
        if self.activity_snapshot_id is not None:
            require_id(
                self.activity_snapshot_id,
                ACTIVITY_SNAPSHOT,
                field="screening.activity_snapshot_id",
            )
        if self.prioritisation_decision_id is not None:
            require_id(
                self.prioritisation_decision_id,
                PRIORITISATION,
                field="screening.prioritisation_decision_id",
            )
        if self.toxicity_analysis_id is not None:
            require_id(
                self.toxicity_analysis_id,
                ANALYSIS,
                field="screening.toxicity_analysis_id",
            )
        if not self.input_smiles.strip():
            raise ValueError("screening.input_smiles is required")
        if (self.toxicity_authorised_by is None) != (
            self.toxicity_authorisation_reason is None
        ):
            raise ValueError("toxicity authorisation requires both actor and reason")

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        run_id: str,
        input_smiles: str,
        subject: ActivitySubject,
        intended_target_role: TargetRole,
        now: datetime,
    ) -> "ActivityGuidedScreening":
        """Open a screening.

        ``input_smiles`` is kept as provenance of what the caller sent;
        ``subject`` carries the canonical identity every later artifact is
        checked against.  Normalising the caller's string into ``subject`` is
        the application boundary's job, not the domain's.
        """
        return cls(
            id=new_id(SCREENING),
            session_id=session_id,
            run_id=run_id,
            input_smiles=input_smiles,
            subject=subject,
            intended_target_role=intended_target_role,
            status=ScreeningStatus.CREATED,
            created_at=now,
            updated_at=now,
        )

    @property
    def target(self) -> TargetIdentity:
        return self.subject.target

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_SCREENING_STATUSES

    def _advance(
        self,
        status: ScreeningStatus,
        *,
        now: datetime,
        **changes: Any,
    ) -> "ActivityGuidedScreening":
        return replace(
            self,
            status=status,
            updated_at=now,
            version=self.version + 1,
            **changes,
        )

    def start_activity(self, *, now: datetime) -> "ActivityGuidedScreening":
        if self.status is not ScreeningStatus.CREATED:
            raise IllegalScreeningTransition(self.status, "start activity")
        return self._advance(ScreeningStatus.ACTIVITY_RUNNING, now=now)

    def record_activity(
        self, snapshot: ActivitySnapshot, *, now: datetime
    ) -> "ActivityGuidedScreening":
        if self.status is not ScreeningStatus.ACTIVITY_RUNNING:
            raise IllegalScreeningTransition(self.status, "record activity")
        if snapshot.session_id != self.session_id or snapshot.run_id != self.run_id:
            raise ValueError("activity snapshot belongs to a different session or run")
        self.subject.require_same(snapshot.subject, artifact="activity snapshot")
        return self._advance(
            ScreeningStatus.ACTIVITY_COMPLETED,
            now=now,
            activity_snapshot_id=snapshot.id,
        )

    def record_decision(
        self, decision: PrioritisationDecision, *, now: datetime
    ) -> "ActivityGuidedScreening":
        if self.status is not ScreeningStatus.ACTIVITY_COMPLETED:
            raise IllegalScreeningTransition(self.status, "record prioritisation decision")
        if decision.activity_snapshot_id != self.activity_snapshot_id:
            raise ValueError("prioritisation decision does not cite this screening's activity")
        if decision.target_id != self.subject.target.target_id:
            raise ValueError("prioritisation decision target does not match the screening target")
        if decision.assay_context != self.subject.assay_context:
            raise ActivityIdentityMismatch(
                ("assay_context",),
                f"prioritisation decision applies a threshold calibrated for "
                f"{decision.assay_context!r} to a screening in "
                f"{self.subject.assay_context!r}",
            )
        if decision.intended_target_role is not self.intended_target_role:
            raise ActivityIdentityMismatch(
                ("intended_target_role",),
                f"prioritisation decision assumes role "
                f"{decision.intended_target_role.value!r} but the screening declared "
                f"{self.intended_target_role.value!r}",
            )
        next_status = {
            PriorityStatus.PRIORITISE: ScreeningStatus.PRIORITISED,
            PriorityStatus.REVIEW: ScreeningStatus.REVIEW_REQUIRED,
            PriorityStatus.LOW_PRIORITY: ScreeningStatus.LOW_PRIORITY,
        }[decision.status]
        return self._advance(
            next_status,
            now=now,
            prioritisation_decision_id=decision.id,
        )

    def start_toxicity(
        self,
        *,
        now: datetime,
        authorised_by: str | None = None,
        authorisation_reason: str | None = None,
    ) -> "ActivityGuidedScreening":
        if self.status is ScreeningStatus.PRIORITISED:
            if authorised_by is not None or authorisation_reason is not None:
                raise ValueError(
                    "policy-prioritised toxicity does not require manual authorisation"
                )
            return self._advance(ScreeningStatus.TOXICITY_RUNNING, now=now)

        if self.status not in {
            ScreeningStatus.REVIEW_REQUIRED,
            ScreeningStatus.LOW_PRIORITY,
        }:
            raise IllegalScreeningTransition(self.status, "start toxicity")
        if not authorised_by or not authorised_by.strip():
            raise ValueError("manual toxicity authorisation requires an actor")
        if not authorisation_reason or not authorisation_reason.strip():
            raise ValueError("manual toxicity authorisation requires a reason")
        return self._advance(
            ScreeningStatus.TOXICITY_RUNNING,
            now=now,
            toxicity_authorised_by=authorised_by,
            toxicity_authorisation_reason=authorisation_reason,
        )

    def complete_toxicity(
        self, analysis_id: str, *, now: datetime
    ) -> "ActivityGuidedScreening":
        if self.status is not ScreeningStatus.TOXICITY_RUNNING:
            raise IllegalScreeningTransition(self.status, "complete toxicity")
        require_id(analysis_id, ANALYSIS, field="analysis_id")
        return self._advance(
            ScreeningStatus.COMPLETED,
            now=now,
            toxicity_analysis_id=analysis_id,
        )

    def complete_without_toxicity(
        self, *, reason: str, now: datetime
    ) -> "ActivityGuidedScreening":
        if self.status not in {
            ScreeningStatus.LOW_PRIORITY,
            ScreeningStatus.REVIEW_REQUIRED,
        }:
            raise IllegalScreeningTransition(self.status, "complete without toxicity")
        if not reason.strip():
            raise ValueError("toxicity_not_run reason is required")
        return self._advance(
            ScreeningStatus.COMPLETED_WITH_TOXICITY_NOT_RUN,
            now=now,
            toxicity_not_run_reason=reason,
        )

    def fail_activity(self, *, failure_code: str, now: datetime) -> "ActivityGuidedScreening":
        if self.status is not ScreeningStatus.ACTIVITY_RUNNING:
            raise IllegalScreeningTransition(self.status, "fail activity")
        if not failure_code.strip():
            raise ValueError("failure_code is required")
        return self._advance(
            ScreeningStatus.ACTIVITY_FAILED,
            now=now,
            failure_code=failure_code,
        )

    def fail_toxicity(self, *, failure_code: str, now: datetime) -> "ActivityGuidedScreening":
        if self.status is not ScreeningStatus.TOXICITY_RUNNING:
            raise IllegalScreeningTransition(self.status, "fail toxicity")
        if not failure_code.strip():
            raise ValueError("failure_code is required")
        return self._advance(
            ScreeningStatus.TOXICITY_FAILED,
            now=now,
            failure_code=failure_code,
        )
