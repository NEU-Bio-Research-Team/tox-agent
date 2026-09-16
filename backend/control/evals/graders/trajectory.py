"""The ``trajectory`` grader: what the run did, read from EvalTraceV1.

``transcript`` checks the tool budget a task pinned by name. This grader reads
the canonical trace instead, so it grades a scripted, live or replayed run the
same way: an ordered subsequence of tools that must appear, a ceiling on
failed and denied calls, and duplicate calls counted by argument hash rather
than by tool name — two searches with different queries are not a loop.
"""
from __future__ import annotations

from .model import GradeResult, TaskOutcome

_GRADER = "trajectory"
VERSION = "trajectory-v1"


def grade_trajectory(task: dict, outcome: TaskOutcome) -> GradeResult:
    from evals.trace import project

    expect = (task.get("expect") or {}).get("trajectory")
    if expect is None:
        return GradeResult.ok(_GRADER)
    trace = project(outcome)
    names = [event.tool_name for event in trace.tools]
    reasons: list[str] = []

    wanted = list(expect.get("ordered_subsequence", ()))
    position = 0
    for name in names:
        if position < len(wanted) and name == wanted[position]:
            position += 1
    if position < len(wanted):
        reasons.append(
            f"tools {wanted[position:]!r} did not appear in order after {wanted[:position]!r}"
        )
    for key, counter in (
        ("max_failed_calls", "failed_calls"),
        ("max_denied_calls", "denied_calls"),
        ("max_duplicate_calls", "duplicate_calls"),
        ("max_searches", "searches"),
        ("max_evidence_reads", "evidence_reads"),
    ):
        limit = expect.get(key)
        if limit is not None and trace.counters[counter] > limit:
            reasons.append(f"{counter}={trace.counters[counter]} exceeds {key}={limit}")
    for key, counter in (("min_searches", "searches"), ("min_evidence_reads", "evidence_reads")):
        floor = expect.get(key)
        if floor is not None and trace.counters[counter] < floor:
            reasons.append(f"{counter}={trace.counters[counter]} is below {key}={floor}")
    return GradeResult(_GRADER, not reasons, tuple(reasons))
