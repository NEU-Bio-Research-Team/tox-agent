"""The ``budget`` grader: success under budget, read from the trace.

The limits come from the task's ``resource_budget`` when it declares one, else
from the EffectiveRunBudgetV1 the runner recorded for the run. A run that
spent more than it was allowed failed this grader even if the product did not
stop it — that is the finding. A run with no known budget is not graded as a
pass: it records why nothing could be checked.
"""
from __future__ import annotations

from .model import GradeResult, TaskOutcome

_GRADER = "budget"
VERSION = "budget-v1"

_LIMITS = (
    ("max_tool_calls", "tool_calls_excluding_submit"),
    ("max_searches", "searches"),
    ("max_evidence_reads", "evidence_reads"),
)


def grade_budget(task: dict, outcome: TaskOutcome) -> GradeResult:
    from evals.trace import project

    limits = task.get("resource_budget") or outcome.budget
    if not limits:
        return GradeResult.fail(_GRADER, "no budget was declared or recorded for this run")
    trace = project(outcome)
    counters = dict(trace.counters)
    counters["tool_calls_excluding_submit"] = counters["tool_calls"] - counters["submits"]
    reasons = [
        f"{counter}={counters[counter]} exceeds {key}={limits[key]}"
        for key, counter in _LIMITS
        if limits.get(key) is not None and counters[counter] > limits[key]
    ]
    candidates = limits.get("max_answer_candidates")
    if candidates is not None and (trace.candidate_generation or 0) > candidates:
        reasons.append(
            f"answer candidate generation {trace.candidate_generation} exceeds {candidates}"
        )
    return GradeResult(_GRADER, not reasons, tuple(reasons))
