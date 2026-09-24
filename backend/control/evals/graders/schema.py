"""The ``schema`` grader (plan section 16.4, "Code/schema").

Structural checks on the committed answer: it has the ``grounded-answer-v1``
shape, it contains the claims/limitations the task said it must, it avoids the
ones it must not, and its markdown mentions (or never mentions) the strings the
task pinned. This is deterministic string/shape work — no judgement of whether
the prose is *good*, which is the rubric grader's job.
"""
from __future__ import annotations

from .model import GradeResult, TaskOutcome

_GRADER = "schema"
_CLAIM_KINDS = {"numeric", "classification", "scientific", "comparison", "limitation", "recommendation"}


def grade_schema(task: dict, outcome: TaskOutcome) -> GradeResult:
    expect = (task.get("expect") or {}).get("answer")
    if expect is None:
        return GradeResult.ok(_GRADER)

    answer = outcome.answer
    if answer is None:
        if expect.get("accepted", True):
            return GradeResult.fail(_GRADER, "no answer to check")
        return GradeResult.ok(_GRADER)

    reasons: list[str] = []

    if answer.get("schema_version") != "grounded-answer-v1":
        reasons.append(f"schema_version is {answer.get('schema_version')!r}")
    claims = answer.get("claims", [])
    for index, claim in enumerate(claims):
        if claim.get("kind") not in _CLAIM_KINDS:
            reasons.append(f"claims[{index}].kind is {claim.get('kind')!r}")
        if not claim.get("claim_id"):
            reasons.append(f"claims[{index}] has no claim_id")

    reasons += _check_required_claims(expect.get("required_claims", []), claims)

    declared_limits = outcome.answer_limitation_codes()
    missing = [c for c in expect.get("required_limitations", []) if c not in declared_limits]
    if missing:
        reasons.append(f"missing required limitation(s): {missing}")
    present_forbidden = [c for c in expect.get("forbidden_limitations", []) if c in declared_limits]
    if present_forbidden:
        reasons.append(f"forbidden limitation(s) present: {present_forbidden}")

    markdown = outcome.answer_markdown().lower()
    for needle in expect.get("must_mention", []):
        if needle.lower() not in markdown:
            reasons.append(f"answer never mentions {needle!r}")
    for needle in expect.get("must_not_mention", []):
        if needle.lower() in markdown:
            reasons.append(f"answer mentions the forbidden string {needle!r}")
    # OR semantics, unlike must_mention's AND: a task states a concept (e.g.
    # "the answer disagrees with a class-wide claim") that a model may express
    # several equally-valid ways, so any one alternative phrasing passes. Found
    # live 2026-09-06 (progress log section 14.4): evsyn-03's answer correctly
    # surfaced a real disagreement in the literature ("not... a class effect",
    # "not proof that every member... blocks hERG") without ever writing the
    # literal word "disagree" that a bare must_mention required.
    any_of = expect.get("must_mention_any_of", [])
    if any_of and not any(needle.lower() in markdown for needle in any_of):
        reasons.append(f"answer matches none of {any_of!r}")

    min_citations = expect.get("min_citations")
    if min_citations is not None:
        total = sum(len(c.get("citation_ids", [])) for c in claims)
        if total < min_citations:
            reasons.append(f"expected >= {min_citations} citations, found {total}")

    return GradeResult(_GRADER, not reasons, tuple(reasons))


def _check_required_claims(required: list[dict], claims: list[dict]) -> list[str]:
    reasons: list[str] = []
    for spec in required:
        if not any(_claim_matches(spec, claim) for claim in claims):
            reasons.append(f"no claim matching {spec}")
    return reasons


def _claim_matches(spec: dict, claim: dict) -> bool:
    if spec.get("kind") and claim.get("kind") != spec["kind"]:
        return False
    if "field_path" in spec and claim.get("field_path") != spec["field_path"]:
        return False
    if "rendered_value" in spec and claim.get("rendered_value") != spec["rendered_value"]:
        return False
    if "source_value" in spec and claim.get("source_value") != spec["source_value"]:
        return False
    return True
