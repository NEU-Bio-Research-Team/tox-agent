"""The ``run`` grader — always applied (plan section 16.4, "State/outcome").

Checks the coarse outcome the task declared: did the run end where it should
(``completed``/``failed``/``cancelled``), with the routed intent and lane the
task expects, and — for tasks that must fail — the right typed product error.
Everything finer-grained is left to ``schema``/``state``/``transcript``.
"""
from __future__ import annotations

from .model import GradeResult, TaskOutcome

_GRADER = "run"


def grade_run(task: dict, outcome: TaskOutcome) -> GradeResult:
    expect = task.get("expect", {})
    reasons: list[str] = []

    run_expect = expect.get("run", {})
    for field in ("status", "intent", "lane", "failure_code"):
        want = run_expect.get(field)
        if want is None:
            continue
        got = outcome.run.get(field)
        if got != want:
            reasons.append(f"run.{field}: expected {want!r}, got {got!r}")

    want_error = expect.get("error_code")
    if want_error is not None:
        # A synchronous 4xx carries the code in the error envelope; an async run
        # that fails carries it in run.failure_code. Either satisfies the task.
        sync_code = (outcome.error or {}).get("error", {}).get("code")
        async_code = outcome.run.get("failure_code")
        if want_error not in (sync_code, async_code):
            reasons.append(
                f"error_code: expected {want_error!r}, got "
                f"envelope={sync_code!r} / run.failure_code={async_code!r}"
            )

    # A task that expects an answer must have produced one; a task that expects
    # none (a pure failure/clarification task) must not have committed one.
    answer_expect = expect.get("answer")
    if answer_expect is not None:
        wants_accepted = answer_expect.get("accepted", True)
        has_answer = outcome.answer is not None
        if wants_accepted and not has_answer:
            reasons.append("expected a committed answer, none was produced")
        if not wants_accepted and has_answer and not outcome.answer.get("is_fallback"):
            reasons.append("expected no model answer, but one was committed")

    return GradeResult(_GRADER, not reasons, tuple(reasons))
