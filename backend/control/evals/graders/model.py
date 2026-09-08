"""Shared grader data types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskOutcome:
    """Everything observed through the product API after a task's conversation.

    The runner fills this from REST reads only — the same surface a client
    has — so grading never depends on internal state a user could not see.
    """

    #: ``GET /v1/sessions/{id}/runs/{run_id}`` of the last run in the task.
    run: dict[str, Any]
    #: ``GET /v1/sessions/{id}``.
    session: dict[str, Any]
    #: The committed answer projection, or ``None`` if the run produced none.
    answer: dict[str, Any] | None = None
    #: Analyses created during the task.
    analyses: list[dict[str, Any]] = field(default_factory=list)
    #: Accepted evidence for the session.
    evidence: list[dict[str, Any]] = field(default_factory=list)
    #: ``run["tool_calls"]`` — name, status, timings.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    #: Canonical messages/parts.
    messages: list[dict[str, Any]] = field(default_factory=list)
    #: The product error envelope, if the triggering request failed outright.
    error: dict[str, Any] | None = None
    #: Set by the runner when it restarted the control plane and re-read the
    #: session: did every claim in ``answer`` still resolve to a live
    #: observation/evidence record? ``None`` when the runner did not test it.
    reconstructed_ok: bool | None = None
    #: Observation ids the runner confirmed belong to this session (used by the
    #: cross-session hard gate). Empty when not gathered.
    session_observation_ids: frozenset[str] = frozenset()
    #: Evidence ids the runner confirmed belong to this session.
    session_evidence_ids: frozenset[str] = frozenset()
    #: ``observation_id -> canonical payload`` for observations this task's
    #: answer cites, so ``claims_match_source`` can recompute the number from
    #: the frozen source rather than trusting the commit-time check. Empty when
    #: the runner could not gather it (then the gate relies on commit-time
    #: validation and only flags a claim with no basis).
    observation_values: dict[str, Any] = field(default_factory=dict)

    def answer_claims(self) -> list[dict[str, Any]]:
        return list((self.answer or {}).get("claims", []))

    def answer_limitation_codes(self) -> set[str]:
        return {lim.get("code") for lim in (self.answer or {}).get("limitations", [])}

    def answer_markdown(self) -> str:
        return (self.answer or {}).get("answer_markdown", "") or ""

    def called_tool_names(self) -> list[str]:
        return [c.get("tool_name") for c in self.tool_calls]


@dataclass(frozen=True)
class GradeResult:
    grader: str
    passed: bool
    reasons: tuple[str, ...] = ()

    @classmethod
    def ok(cls, grader: str) -> "GradeResult":
        return cls(grader, True, ())

    @classmethod
    def fail(cls, grader: str, *reasons: str) -> "GradeResult":
        return cls(grader, False, tuple(reasons))


@dataclass(frozen=True)
class TaskReport:
    task_id: str
    category: str
    critical: bool
    results: tuple[GradeResult, ...]
    deferred_graders: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def hard_gate_failed(self) -> bool:
        return any(r.grader == "hard_gates" and not r.passed for r in self.results)

    def reasons(self) -> list[str]:
        out: list[str] = []
        for result in self.results:
            out.extend(f"[{result.grader}] {reason}" for reason in result.reasons)
        return out
