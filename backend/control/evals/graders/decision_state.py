"""The ``decision_state`` grader: plan, coverage, conflict and stop are data.

Reads the DecisionSupportStateV1 the run persisted. Without it a benchmark
cannot tell a run that stopped because it had enough from one that stopped
because it ran out, so a task that declares this grader and gets no state
fails with that reason rather than passing on the answer alone.
"""
from __future__ import annotations

from .model import GradeResult, TaskOutcome

_GRADER = "decision_state"
VERSION = "decision-state-v1"


def grade_decision_state(task: dict, outcome: TaskOutcome) -> GradeResult:
    expect = (task.get("expect") or {}).get("decision_state") or {}
    state = outcome.decision_state
    if state is None:
        return GradeResult.fail(_GRADER, "the run persisted no decision-support state")
    reasons: list[str] = []
    stop = state.get("stop_reason")
    if stop is None:
        reasons.append("the state has no stop_reason; the run did not say why it stopped")
    allowed = expect.get("stop_reason_in")
    if allowed and stop not in allowed:
        reasons.append(f"stop_reason {stop!r} not in {allowed!r}")
    propositions = state.get("propositions") or []
    if len(propositions) < expect.get("min_propositions", 1):
        reasons.append(
            f"{len(propositions)} propositions; expected at least {expect.get('min_propositions', 1)}"
        )
    statuses = [p.get("status") for p in propositions]
    for status in expect.get("statuses_include", ()):
        if status not in statuses:
            reasons.append(f"no proposition reached status {status!r}")
    for status in expect.get("statuses_exclude", ()):
        if status in statuses:
            reasons.append(f"a proposition has forbidden status {status!r}")
    if stop == "sufficient" and "open" in statuses:
        reasons.append("stop_reason is 'sufficient' while a proposition is still open")
    for proposition in propositions:
        if proposition.get("status") in ("supported", "conflicted") and not proposition.get(
            "artifact_refs"
        ):
            reasons.append(
                f"proposition {proposition.get('id')!r} is {proposition.get('status')} "
                "with no artifact_refs"
            )
    coverage = state.get("coverage") or {}
    if coverage.get("resolved", 0) > coverage.get("required", 0):
        reasons.append("coverage.resolved exceeds coverage.required")
    return GradeResult(_GRADER, not reasons, tuple(reasons))
