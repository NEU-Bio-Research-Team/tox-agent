"""Durable, runtime-independent scientific investigation state.

Only decisions and evidence references are persisted.  Model reasoning and
private chain-of-thought have deliberately no field in these aggregates.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from .ids import CASE, CONFLICT, GAP, PLAN, SESSION, STEP, new_id, require_id


class GoalType(str, Enum):
    EXPLAIN_PREDICTION = "explain_prediction"
    ASSESS_EVIDENCE = "assess_evidence"
    COMPARE_ENDPOINTS = "compare_endpoints"
    FIND_CONTRADICTIONS = "find_contradictions"
    INVESTIGATE_MODEL_BEHAVIOR = "investigate_model_behavior"
    PLAN_VERIFICATION = "plan_verification"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ConflictStatus(str, Enum):
    UNRESOLVED = "unresolved"
    RESOLVED_BY_QUALITY = "resolved_by_quality"
    RESOLVED_BY_CONTEXT = "resolved_by_context"
    NOT_RESOLVABLE = "not_resolvable"


class GapSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKING = "blocking"


@dataclass(frozen=True, slots=True)
class InvestigationStep:
    id: str
    question: str
    capability: str
    input_refs: tuple[str, ...]
    expected_output: str
    success_condition: str
    case_revision: int
    status: StepStatus = StepStatus.PENDING
    output_refs: tuple[str, ...] = ()
    failure_reason: str | None = None

    @classmethod
    def create(cls, **values: Any) -> "InvestigationStep":
        return cls(id=new_id(STEP), **values)

    def __post_init__(self) -> None:
        require_id(self.id, STEP, field="step.id")
        if not self.question.strip() or not self.success_condition.strip():
            raise ValueError("step question and success condition are required")


@dataclass(frozen=True, slots=True)
class InvestigationPlan:
    id: str
    case_id: str
    revision: int
    reason: str
    steps: tuple[InvestigationStep, ...]
    created_at: datetime

    @classmethod
    def create(cls, *, case_id: str, reason: str, steps: tuple[InvestigationStep, ...],
               now: datetime, revision: int = 1) -> "InvestigationPlan":
        return cls(new_id(PLAN), case_id, revision, reason, steps, now)

    def __post_init__(self) -> None:
        require_id(self.id, PLAN, field="plan.id")
        require_id(self.case_id, CASE, field="plan.case_id")
        if self.revision < 1 or not self.reason.strip() or not self.steps:
            raise ValueError("plan needs a positive revision, reason, and at least one step")
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step id")


@dataclass(frozen=True, slots=True)
class EvidenceConflict:
    id: str
    proposition: str
    claim_ids: tuple[str, ...]
    status: ConflictStatus = ConflictStatus.UNRESOLVED
    resolution_reason: str | None = None

    @classmethod
    def create(cls, proposition: str, claim_ids: tuple[str, ...]) -> "EvidenceConflict":
        return cls(new_id(CONFLICT), proposition, claim_ids)


@dataclass(frozen=True, slots=True)
class EvidenceGap:
    id: str
    question: str
    severity: GapSeverity
    reason: str

    @classmethod
    def create(cls, question: str, severity: GapSeverity, reason: str) -> "EvidenceGap":
        return cls(new_id(GAP), question, severity, reason)


@dataclass(frozen=True, slots=True)
class Coverage:
    required_questions: tuple[str, ...]
    answered_questions: tuple[str, ...] = ()
    unresolved_conflict_ids: tuple[str, ...] = ()
    blocking_gap_ids: tuple[str, ...] = ()

    @property
    def ratio(self) -> float:
        required = set(self.required_questions)
        return 1.0 if not required else len(required & set(self.answered_questions)) / len(required)

    @property
    def sufficient(self) -> bool:
        return self.ratio == 1.0 and not self.blocking_gap_ids


@dataclass(frozen=True, slots=True)
class CaseState:
    id: str
    session_id: str
    subject: Mapping[str, Any]
    goal: GoalType
    active_analysis_id: str | None
    questions: tuple[str, ...]
    hypothesis_refs: tuple[str, ...]
    claim_refs: tuple[str, ...]
    conflicts: tuple[EvidenceConflict, ...]
    gaps: tuple[EvidenceGap, ...]
    plan_id: str | None
    coverage: Coverage
    action_refs: tuple[str, ...]
    revision: int
    revision_reason: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, *, session_id: str, subject: Mapping[str, Any], goal: GoalType,
               questions: tuple[str, ...], now: datetime) -> "CaseState":
        return cls(
            id=new_id(CASE), session_id=session_id, subject=dict(subject), goal=goal,
            active_analysis_id=None, questions=questions, hypothesis_refs=(), claim_refs=(),
            conflicts=(), gaps=(), plan_id=None, coverage=Coverage(questions), action_refs=(),
            revision=1, revision_reason="case_created", created_at=now, updated_at=now,
        )

    def __post_init__(self) -> None:
        require_id(self.id, CASE, field="case.id")
        require_id(self.session_id, SESSION, field="case.session_id")
        if self.revision < 1 or not self.revision_reason.strip():
            raise ValueError("case revision and revision reason are required")

    def revise(self, *, reason: str, now: datetime, **changes: Any) -> "CaseState":
        if not reason.strip():
            raise ValueError("revision reason is required")
        forbidden = {"id", "session_id", "created_at", "revision"} & set(changes)
        if forbidden:
            raise ValueError(f"immutable case fields cannot change: {sorted(forbidden)}")
        return replace(
            self, **changes, revision=self.revision + 1,
            revision_reason=reason, updated_at=now,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.id, "session_id": self.session_id, "subject": dict(self.subject),
            "goal": self.goal.value, "active_analysis_id": self.active_analysis_id,
            "questions": list(self.questions), "hypothesis_refs": list(self.hypothesis_refs),
            "claim_refs": list(self.claim_refs),
            "conflicts": [
                {"id": c.id, "proposition": c.proposition, "claim_ids": list(c.claim_ids),
                 "status": c.status.value, "resolution_reason": c.resolution_reason}
                for c in self.conflicts
            ],
            "gaps": [
                {"id": g.id, "question": g.question, "severity": g.severity.value,
                 "reason": g.reason} for g in self.gaps
            ],
            "plan_id": self.plan_id,
            "coverage": {
                "required_questions": list(self.coverage.required_questions),
                "answered_questions": list(self.coverage.answered_questions),
                "unresolved_conflict_ids": list(self.coverage.unresolved_conflict_ids),
                "blocking_gap_ids": list(self.coverage.blocking_gap_ids),
                "ratio": self.coverage.ratio, "sufficient": self.coverage.sufficient,
            },
            "action_refs": list(self.action_refs), "revision": self.revision,
            "revision_reason": self.revision_reason,
            "created_at": self.created_at.isoformat(), "updated_at": self.updated_at.isoformat(),
        }
