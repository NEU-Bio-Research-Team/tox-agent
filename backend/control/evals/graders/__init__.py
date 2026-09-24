"""Deterministic graders (plan section 16.4, rows "Code/schema" and
"State/outcome", plus the section 16.5 hard gates).

Every grader is a pure function ``(task, outcome) -> GradeResult``. They never
touch the network or a database — the runner gathers a :class:`TaskOutcome` from
the product API and hands the same frozen snapshot to each grader, so a grading
result is reproducible from the recorded outcome alone.

``GRADER_REGISTRY`` declares each grader's version and the outcome fields it
reads. The manifest records the versions, so two runs graded by different
grader code are visibly not comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .artifacts import grade_artifacts
from .budget import grade_budget
from .decision_state import grade_decision_state
from .hard_gates import grade_hard_gates
from .model import GradeResult, TaskOutcome, TaskReport
from .outcome_split import grade_outcome_split
from .run_shape import grade_run
from .schema import grade_schema
from .state import grade_state
from .trajectory import grade_trajectory
from .transcript import grade_transcript


@dataclass(frozen=True)
class GraderSpec:
    name: str
    version: str
    fn: Callable[[dict, TaskOutcome], GradeResult] | None
    #: TaskOutcome fields read. A driver that cannot fill one of them cannot
    #: honestly run this grader.
    inputs: tuple[str, ...]
    kind: str = "deterministic"  # deterministic | model | human


GRADER_REGISTRY: dict[str, GraderSpec] = {
    spec.name: spec
    for spec in (
        GraderSpec("run", "run-v1", grade_run, ("run", "error", "answer")),
        GraderSpec("hard_gates", "hard-gates-v2", None,
                   ("answer", "evidence", "tool_calls", "observation_values",
                    "session_observation_ids", "session_evidence_ids", "reconstructed_ok")),
        GraderSpec("schema", "schema-v1", grade_schema, ("answer",)),
        GraderSpec("state", "state-v1", grade_state, ("analyses", "answer", "evidence")),
        GraderSpec("transcript", "transcript-v1", grade_transcript, ("tool_calls",)),
        GraderSpec("trajectory", "trajectory-v1", grade_trajectory, ("run", "tool_calls")),
        GraderSpec("budget", "budget-v1", grade_budget, ("tool_calls", "answer", "budget")),
        GraderSpec("outcome_split", "outcome-split-v1", grade_outcome_split, ("answer",)),
        GraderSpec("decision_state", "decision-state-v1", grade_decision_state,
                   ("decision_state",)),
        GraderSpec("artifacts", "artifacts-v1", grade_artifacts,
                   ("analyses", "answer", "reports", "evidence", "decision_state")),
        GraderSpec("rubric", "rubric-dimensions-v1", None, ("answer", "evidence"), kind="model"),
        GraderSpec("semantic", "semantic-judge-v1", None, ("answer", "evidence"), kind="model"),
        GraderSpec("sme", "sme-protocol-v1", None, ("answer", "evidence"), kind="human"),
    )
}

#: Name -> grader. ``run`` is always applied; the rest are opt-in per task via
#: the task's ``graders`` array (default ``["schema", "state"]``).
GRADERS = {
    name: spec.fn
    for name, spec in GRADER_REGISTRY.items()
    if spec.fn is not None and name != "run"
}

DEFERRED_KINDS = frozenset({"model", "human"})


def grader_versions() -> dict[str, str]:
    return {name: spec.version for name, spec in GRADER_REGISTRY.items()}


def grade_task(task: dict, outcome: TaskOutcome) -> TaskReport:
    """Apply ``run`` + hard gates + the task's declared deterministic graders.

    Model and human graders are recorded as deferred, never as pass, so a suite
    summary cannot silently count an un-run judgement as green.
    """
    results: list[GradeResult] = [grade_run(task, outcome)]
    hard = grade_hard_gates(task, outcome)
    if hard is not None:
        results.append(hard)
    declared = list(dict.fromkeys(
        list(task.get("graders", ["schema", "state"])) + list(task.get("required_graders", ()))
    ))
    for name in declared:
        grader = GRADERS.get(name)
        if grader is not None:
            results.append(grader(task, outcome))
    deferred = [
        g for g in declared
        if g in GRADER_REGISTRY and GRADER_REGISTRY[g].kind in DEFERRED_KINDS
    ]
    return TaskReport(
        task_id=task["task_id"],
        category=task["category"],
        critical=task.get("critical", False),
        results=tuple(results),
        deferred_graders=tuple(deferred),
    )


__all__ = [
    "GRADERS",
    "GRADER_REGISTRY",
    "GradeResult",
    "GraderSpec",
    "TaskOutcome",
    "TaskReport",
    "grade_task",
    "grade_hard_gates",
    "grade_run",
    "grade_schema",
    "grade_state",
    "grade_transcript",
    "grader_versions",
]
