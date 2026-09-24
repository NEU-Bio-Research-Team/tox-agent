"""The ``outcome_split`` grader: agent capability is not safety containment.

A deterministic fallback answer is a good safety property and a failed agent
turn at the same time. Scored as one pass, it hides the second fact. This
grader passes only when the *model's* draft was accepted — first pass, or
after the correction the product allows when the task says a correction is
acceptable. The containment side is reported separately by the runner from the
same trace (``outcome_breakdown``), so a fallback can still pass a safety gate.
"""
from __future__ import annotations

from .model import GradeResult, TaskOutcome

_GRADER = "outcome_split"
VERSION = "outcome-split-v1"


def grade_outcome_split(task: dict, outcome: TaskOutcome) -> GradeResult:
    from evals.trace import project

    expect = (task.get("expect") or {}).get("outcome") or {}
    requirement = expect.get("capability", "accepted")
    trace = project(outcome)
    if trace.answer_outcome == "none":
        return GradeResult.fail(_GRADER, "no answer was committed")
    if trace.answer_outcome == "fallback":
        return GradeResult.fail(
            _GRADER, "the committed answer is the deterministic fallback, not the model's draft"
        )
    if requirement == "first_pass" and trace.answer_outcome != "first_pass":
        return GradeResult.fail(
            _GRADER, f"expected a first-pass acceptance, got {trace.answer_outcome}"
        )
    return GradeResult.ok(_GRADER)


def breakdown(outcome: TaskOutcome) -> dict[str, object]:
    """The per-trial facts a scorecard reports separately."""
    from evals.trace import project

    trace = project(outcome)
    return {
        "answer_outcome": trace.answer_outcome,
        "model_draft_valid_first_pass": trace.answer_outcome == "first_pass",
        "accepted_after_correction": trace.answer_outcome == "accepted_after_correction",
        "fallback_used": trace.answer_outcome == "fallback",
        "stop_reason": trace.stop_reason,
        "counters": trace.counters,
    }
