"""The ``artifacts`` grader: the durable things a task was supposed to leave.

Reads the task's v3 ``expected_artifacts`` (kind with min/max counts) and, for
reports, ``expect.report`` (allowed statuses and gap reasons that must be
recorded). A report build that "completed" without a report, or published one
with a status the task rules out, fails here rather than passing on the run
status alone.
"""
from __future__ import annotations

from .model import GradeResult, TaskOutcome

_GRADER = "artifacts"
VERSION = "artifacts-v1"


def _count(kind: str, outcome: TaskOutcome) -> int | None:
    return {
        "analysis": len(outcome.analyses),
        "answer": 1 if outcome.answer else 0,
        "report": len(outcome.reports),
        "evidence": len(outcome.evidence),
        "decision_state": 1 if outcome.decision_state else 0,
        "development_posture": 1 if (outcome.answer or {}).get("development_posture") else 0,
    }.get(kind)


def grade_artifacts(task: dict, outcome: TaskOutcome) -> GradeResult:
    reasons: list[str] = []
    for spec in task.get("expected_artifacts", ()):
        got = _count(spec["kind"], outcome)
        if got is None:
            reasons.append(f"{spec['kind']}: this driver cannot count that artifact kind")
            continue
        if "min" in spec and got < spec["min"]:
            reasons.append(f"{spec['kind']}: {got} < min {spec['min']}")
        if "max" in spec and got > spec["max"]:
            reasons.append(f"{spec['kind']}: {got} > max {spec['max']}")
    report_expect = (task.get("expect") or {}).get("report")
    if report_expect:
        latest = outcome.reports[0] if outcome.reports else None
        if latest is None:
            reasons.append("expected a report artifact, none was published")
        else:
            allowed = report_expect.get("status_in")
            if allowed and latest.get("status") not in allowed:
                reasons.append(f"report status {latest.get('status')!r} not in {allowed!r}")
            gaps = {g.get("reason") for g in latest.get("gaps") or ()}
            for reason in report_expect.get("gap_reasons_include", ()):
                if reason not in gaps:
                    reasons.append(f"report records no gap with reason {reason!r}")
            if report_expect.get("schema_version") and latest.get("schema_version") != report_expect["schema_version"]:
                reasons.append(
                    f"report schema {latest.get('schema_version')!r} is not {report_expect['schema_version']!r}"
                )
    return GradeResult(_GRADER, not reasons, tuple(reasons))
