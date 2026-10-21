"""The publication policy as a pure table, and the case's conclusion isolation (#47).

Every review status meets exactly one disposition under each declared policy;
publication, review and task success stay separate fields; a conclusion a run
proposed becomes the case's only when that run's answer is published, and a
later turn is never told a proposal is what the case can say.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.domain import scientific_case as sc
from toxagent.domain.ids import ANSWER, RUN, SESSION, new_id
from toxagent.domain.run_result import (
    Disposition,
    ExecutionOutcome,
    KnowledgeOutcome,
    ReleasePolicy,
    ReviewStatus,
    RunResultV1,
    decide_publication,
    execution_outcome_of,
    knowledge_outcome_of,
    policy_sha256,
    review_status_of,
)

STRICT = ReleasePolicy.STRICT_REVIEWED_V1


@pytest.mark.parametrize("status, disposition", [
    (ReviewStatus.PASSED, Disposition.PUBLISH),
    (ReviewStatus.REPAIRED_AND_PASSED, Disposition.PUBLISH),
    (ReviewStatus.NOT_APPLICABLE, Disposition.PUBLISH),
    (ReviewStatus.BLOCKED, Disposition.PARTIAL_FACTS),
    (ReviewStatus.INCOMPLETE, Disposition.PARTIAL_FACTS),
    (ReviewStatus.NOT_COMPLETED, Disposition.PARTIAL_FACTS),
])
def test_strict_publishes_only_what_the_review_cleared(status, disposition):
    assert decide_publication(STRICT, review_status=status, has_candidate=True).disposition \
        is disposition


def test_no_candidate_is_withheld_and_a_fallback_is_partial_facts():
    assert decide_publication(STRICT, review_status=ReviewStatus.PASSED,
                              has_candidate=False).disposition is Disposition.WITHHOLD
    fallback = decide_publication(STRICT, review_status=None, has_candidate=True,
                                  is_fallback=True)
    assert fallback.disposition is Disposition.PARTIAL_FACTS
    assert fallback.review_status is ReviewStatus.NOT_APPLICABLE


def test_hard_only_is_a_declared_mode_not_a_failed_review():
    decision = decide_publication(ReleasePolicy.HARD_ONLY_V1,
                                  review_status=ReviewStatus.NOT_COMPLETED, has_candidate=True)
    assert decision.disposition is Disposition.PUBLISH
    assert decision.review_status is ReviewStatus.NOT_REVIEWED_BY_DESIGN
    with pytest.raises(ValueError):
        decide_publication(ReleasePolicy.LEGACY, review_status=None, has_candidate=True)


def test_review_outcomes_map_to_statuses_by_why_the_review_could_not_gate():
    assert review_status_of("reviewed") is ReviewStatus.PASSED
    assert review_status_of("released_with_unresolved_review") is ReviewStatus.BLOCKED
    assert review_status_of(
        "ungated", ungated_reason="nothing blocked, but 1 claim(s) were not judged "
                                  "(cannot_judge or no verdict)",
    ) is ReviewStatus.INCOMPLETE
    assert review_status_of(
        "ungated", ungated_reason="the review did not complete (failed: the reviewer "
                                  "submitted no verdicts)",
    ) is ReviewStatus.NOT_COMPLETED
    assert review_status_of("ungated", ungated_reason="the answer makes no reviewable claim") \
        is ReviewStatus.NOT_APPLICABLE


def test_the_two_stop_axes_never_fold_a_provider_failure_into_no_evidence():
    assert execution_outcome_of("evidence_unavailable", run_status="failed") \
        is ExecutionOutcome.PROVIDER_FAILED
    assert execution_outcome_of("deadline_exceeded", run_status="failed") \
        is ExecutionOutcome.BUDGET_EXHAUSTED
    assert execution_outcome_of(None, run_status="cancelled") is ExecutionOutcome.CANCELLED
    assert knowledge_outcome_of("sufficient", published=False) \
        is KnowledgeOutcome.EVIDENCE_INSUFFICIENT
    assert knowledge_outcome_of("sufficient", published=True) \
        is KnowledgeOutcome.SUPPORTED_ASSESSMENT


def test_a_result_keeps_publication_and_review_apart_and_names_its_policy():
    now = datetime(2026, 10, 21, tzinfo=timezone.utc)
    base = dict(run_id=new_id(RUN), session_id=new_id(SESSION), created_at=now,
                execution_outcome=ExecutionOutcome.COMPLETED,
                knowledge_outcome=KnowledgeOutcome.NONE, release_policy=STRICT)
    with pytest.raises(ValueError):
        RunResultV1(disposition=Disposition.PUBLISH, review_status=ReviewStatus.PASSED, **base)
    with pytest.raises(ValueError):
        RunResultV1(disposition=Disposition.WITHHOLD, review_status=ReviewStatus.NOT_COMPLETED,
                    public_answer_id=new_id(ANSWER), **base)
    result = RunResultV1(disposition=Disposition.PARTIAL_FACTS,
                         review_status=ReviewStatus.BLOCKED,
                         candidate_answer_id=new_id(ANSWER), **base)
    data = result.to_dict()
    assert data["release_policy_sha256"] == policy_sha256(STRICT)
    assert "task_success" not in data


def _update(op: str, actor: str, run_id: str, **payload) -> sc.CaseUpdate:
    return sc.CaseUpdate(op=op, payload=payload, actor=actor, at="2026-10-21T00:00:00+00:00",
                         run_id=run_id)


def _case_with_evidence(run_id: str) -> sc.ScientificCaseV1:
    case = sc.apply(None, _update("open", "server", run_id, case_id="scase_" + "1" * 32,
                                  session_id=new_id(SESSION), subject_key="analysis:x"))
    return sc.apply(case, _update(
        "record_evidence", "model", run_id, claim="read", source_class="predictor_fact",
        source_ref="observation:obs_" + "2" * 32, stance="contextual",
    ))


def _conclude(case, run_id, text, *, status=sc.CONCLUSION_PROPOSED):
    return sc.apply(case, _update(
        "set_conclusion", "model", run_id, status=status,
        can_say=[{"text": text, "evidence_ids": [case.evidence[0].id]}],
    ))


def test_a_proposed_conclusion_is_promoted_only_for_its_own_run():
    first, second = new_id(RUN), new_id(RUN)
    case = _conclude(_case_with_evidence(first), first, "trusted", status=sc.CONCLUSION_RELEASED)
    case = _conclude(case, second, "proposal")
    assert case.conclusion.status == sc.CONCLUSION_PROPOSED
    assert case.released_conclusion.can_say[0].text == "trusted"
    assert sc.apply(case, _update("promote_conclusion", "server", first)) is case
    promoted = sc.apply(case, _update("promote_conclusion", "server", second))
    assert promoted.conclusion.status == sc.CONCLUSION_RELEASED
    assert promoted.released_conclusion is None
    with pytest.raises(sc.InvalidCaseUpdate):
        sc.apply(case, _update("promote_conclusion", "model", second))


def test_a_later_turn_is_told_a_proposal_is_not_the_case_conclusion():
    first, second = new_id(RUN), new_id(RUN)
    case = _conclude(_case_with_evidence(first), first, "trusted", status=sc.CONCLUSION_RELEASED)
    case = _conclude(case, second, "unreleased claim")
    summary = sc.checkpoint_summary(case)
    assert "Conclusion so far: can say: trusted" in summary
    assert "not published -- not the case's conclusion: can say: unreleased claim" in summary
    replayed = sc.ScientificCaseV1.from_dict(case.to_dict())
    assert replayed == case
