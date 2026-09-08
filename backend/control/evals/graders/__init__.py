"""Deterministic graders (plan section 16.4, rows "Code/schema" and
"State/outcome", plus the section 16.5 hard gates).

Every grader is a pure function ``(task, outcome) -> GradeResult``. They never
touch the network or a database — the runner gathers a :class:`TaskOutcome` from
the product API and hands the same frozen snapshot to each grader, so a grading
result is reproducible from the recorded outcome alone.
"""
from __future__ import annotations

from .hard_gates import grade_hard_gates
from .model import GradeResult, TaskOutcome, TaskReport
from .run_shape import grade_run
from .schema import grade_schema
from .state import grade_state
from .transcript import grade_transcript

#: Name -> grader. ``run`` is always applied; the rest are opt-in per task via
#: the task's ``graders`` array (default ``["schema", "state"]``).
GRADERS = {
    "schema": grade_schema,
    "state": grade_state,
    "transcript": grade_transcript,
}


def grade_task(task: dict, outcome: TaskOutcome) -> TaskReport:
    """Apply ``run`` + hard gates + the task's declared deterministic graders.

    ``rubric`` and ``sme`` are recorded as deferred, never as pass, so a suite
    summary cannot silently count an un-run model/human judgement as green.
    """
    results: list[GradeResult] = [grade_run(task, outcome)]
    hard = grade_hard_gates(task, outcome)
    if hard is not None:
        results.append(hard)
    for name in task.get("graders", ["schema", "state"]):
        grader = GRADERS.get(name)
        if grader is not None:
            results.append(grader(task, outcome))
    deferred = [g for g in task.get("graders", []) if g in {"rubric", "sme"}]
    return TaskReport(
        task_id=task["task_id"],
        category=task["category"],
        critical=task.get("critical", False),
        results=tuple(results),
        deferred_graders=tuple(deferred),
    )


__all__ = [
    "GRADERS",
    "GradeResult",
    "TaskOutcome",
    "TaskReport",
    "grade_task",
    "grade_hard_gates",
    "grade_run",
    "grade_schema",
    "grade_state",
    "grade_transcript",
]
