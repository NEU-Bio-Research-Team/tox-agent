"""The ``state`` grader (plan section 16.4, "State/outcome").

Checks the product's persisted state after the task, read back over REST:
how many analysis snapshots and accepted answers exist, whether enough
evidence was accepted, whether every claim in the answer resolves to a source
in this session, and — when the runner tested it — whether the session
reconstructs after a control-plane restart (plan sections 13, 16.5 #10).
"""
from __future__ import annotations

from .model import GradeResult, TaskOutcome

_GRADER = "state"


def grade_state(task: dict, outcome: TaskOutcome) -> GradeResult:
    expect = (task.get("expect") or {}).get("state")
    if expect is None:
        return GradeResult.ok(_GRADER)

    reasons: list[str] = []

    want_snapshots = expect.get("analysis_snapshots")
    if want_snapshots is not None and len(outcome.analyses) != want_snapshots:
        reasons.append(
            f"analysis snapshots: expected {want_snapshots}, found {len(outcome.analyses)}"
        )

    want_answers = expect.get("accepted_answers")
    if want_answers is not None:
        found = 1 if outcome.answer is not None else 0
        if found != want_answers:
            reasons.append(f"accepted answers: expected {want_answers}, found {found}")

    want_evidence = expect.get("evidence_accepted_min")
    if want_evidence is not None and len(outcome.evidence) < want_evidence:
        reasons.append(
            f"accepted evidence: expected >= {want_evidence}, found {len(outcome.evidence)}"
        )

    if expect.get("claim_source_graph_complete"):
        reasons += _incomplete_source_graph(outcome)

    if expect.get("reconstructable_after_restart"):
        if outcome.reconstructed_ok is None:
            reasons.append("runner did not test post-restart reconstruction")
        elif not outcome.reconstructed_ok:
            reasons.append("session did not reconstruct after a control-plane restart")

    return GradeResult(_GRADER, not reasons, tuple(reasons))


def _incomplete_source_graph(outcome: TaskOutcome) -> list[str]:
    """Every fact-kind claim must name a source that resolves in this session
    (plan section 16.5 #10, PROD-01)."""
    reasons: list[str] = []
    known_obs = outcome.session_observation_ids
    known_evd = outcome.session_evidence_ids
    for claim in outcome.answer_claims():
        kind = claim.get("kind")
        if kind in ("limitation", "recommendation"):
            continue
        obs = claim.get("observation_id")
        cites = claim.get("citation_ids", [])
        if not obs and not cites and not claim.get("input_claim_ids"):
            reasons.append(f"claim {claim.get('claim_id')} has no source at all")
            continue
        if obs and known_obs and obs not in known_obs:
            reasons.append(f"claim {claim.get('claim_id')} cites observation {obs} not in this session")
        for evidence_id in cites:
            if known_evd and evidence_id not in known_evd:
                reasons.append(
                    f"claim {claim.get('claim_id')} cites evidence {evidence_id} not in this session"
                )
    return reasons
