"""The ``transcript`` grader (plan section 16.4, "Transcript heuristic").

Plan section 16.4 is explicit that outcome matters more than the exact tool
trajectory, so this only checks what the task pinned: tools that had to be
called, tools that must never have been (a denied tool, or web fetch), a call
budget, and a duplicate-call budget. It does not assert an ordering.
"""
from __future__ import annotations

from collections import Counter

from .model import GradeResult, TaskOutcome

_GRADER = "transcript"


def grade_transcript(task: dict, outcome: TaskOutcome) -> GradeResult:
    expect = (task.get("expect") or {}).get("tools")
    if expect is None:
        return GradeResult.ok(_GRADER)

    called = outcome.called_tool_names()
    counts = Counter(called)
    reasons: list[str] = []

    for name in expect.get("required", []):
        if name not in counts:
            reasons.append(f"required tool {name!r} was never called")
    for name in expect.get("forbidden", []):
        if name in counts:
            reasons.append(f"forbidden tool {name!r} was called {counts[name]}x")

    max_calls = expect.get("max_calls")
    if max_calls is not None and len(called) > max_calls:
        reasons.append(f"{len(called)} tool calls exceeds the budget of {max_calls}")

    max_dupes = expect.get("max_duplicate_calls")
    if max_dupes is not None:
        worst = max((c - 1 for c in counts.values()), default=0)
        if worst > max_dupes:
            reasons.append(f"a tool was called {worst + 1}x; duplicate budget is {max_dupes}")

    return GradeResult(_GRADER, not reasons, tuple(reasons))
