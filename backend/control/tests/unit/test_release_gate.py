"""Whether a reviewed answer may be released, and what the repair cost.

The second half is why the first is safe: any support check can be passed by
deleting every claim, and a final pass rate shows the precision that buys
while hiding the content it loses.
"""
from __future__ import annotations

from dataclasses import dataclass

from toxagent.application.investigation.release_gate import (
    BLOCKING_VERDICTS,
    release_decision,
    repair_effect,
)
from toxagent.domain.answer import LimitationCode


@dataclass(frozen=True)
class _Claim:
    claim_id: str
    text: str


@dataclass(frozen=True)
class _Limitation:
    code: LimitationCode


@dataclass(frozen=True)
class _Answer:
    claims: tuple[_Claim, ...]
    limitations: tuple[_Limitation, ...] = ()


def _review(claim_id: str, verdict: str, reason: str = "") -> dict:
    return {"claim_id": claim_id, "verdict": verdict, "reason": reason}


# ── the release decision ──────────────────────────────────────────────────

def test_a_fully_supported_answer_is_released():
    decision = release_decision(
        [_review("c1", "supported"), _review("c2", "supported")],
        reviewed_claim_ids=["c1", "c2"],
    )
    assert decision.release
    assert decision.violations == ()
    assert decision.counts == {"supported": 2}


def test_an_unsupported_claim_blocks_release_and_names_its_exits():
    decision = release_decision(
        [_review("c1", "not_supported", "the paper measures a different endpoint")],
        reviewed_claim_ids=["c1"],
    )
    assert not decision.release
    assert decision.blocking_claim_ids == ("c1",)
    violation = decision.violations[0]
    assert violation.code == "claim_not_supported"
    assert violation.path == "claims[c1]"
    # the three honest exits, and no fourth
    assert "correct it" in violation.message and "remove it" in violation.message
    assert "different endpoint" in violation.message


def test_a_partially_supported_claim_blocks_and_asks_for_narrowing():
    decision = release_decision(
        [_review("c1", "partially_supported")], reviewed_claim_ids=["c1"]
    )
    assert not decision.release
    assert "narrow the claim" in decision.violations[0].message


def test_cannot_judge_is_recorded_and_does_not_block():
    """The reviewer has no read tool and judges a bundle, so this is usually a
    fact about the bundle rather than about the claim."""
    decision = release_decision(
        [_review("c1", "cannot_judge"), _review("c2", "supported")],
        reviewed_claim_ids=["c1", "c2"],
    )
    assert decision.release
    assert decision.counts["cannot_judge"] == 1


def test_a_claim_under_review_with_no_verdict_is_counted_not_assumed_supported():
    decision = release_decision(
        [_review("c1", "supported")], reviewed_claim_ids=["c1", "c2"]
    )
    assert decision.counts["no_verdict"] == 1
    assert decision.release


def test_a_verdict_for_a_claim_that_was_not_under_review_is_ignored():
    decision = release_decision(
        [_review("c1", "supported"), _review("c9", "not_supported")],
        reviewed_claim_ids=["c1"],
    )
    assert decision.release
    assert "c9" not in decision.blocking_claim_ids


def test_a_review_that_produced_nothing_releases_but_says_it_was_not_a_gate():
    """A missing reviewer must not silently read as a passing review."""
    decision = release_decision([], reviewed_claim_ids=["c1"])
    assert decision.release
    assert decision.ungated_reason == "the reviewer returned no usable verdict"


def test_an_answer_with_no_reviewable_claim_is_released_and_says_so():
    decision = release_decision([], reviewed_claim_ids=[])
    assert decision.release
    assert decision.ungated_reason == "the answer makes no reviewable claim"


def test_only_the_two_blocking_verdicts_block():
    assert BLOCKING_VERDICTS == {"not_supported", "partially_supported"}


# ── what the repair cost ──────────────────────────────────────────────────

def test_a_repair_that_only_deletes_is_reported_as_such():
    """Precision goes up, the answer gets emptier, and the pass rate alone
    shows the first and hides the second."""
    before = _Answer(claims=(_Claim("c1", "IC50 is 1 nM"), _Claim("c2", "it is cardiotoxic")))
    after = _Answer(claims=(_Claim("c1", "IC50 is 1 nM"),))

    effect = repair_effect(before, after)

    assert effect.removed_claim_ids == ("c2",)
    assert effect.added_claim_ids == ()
    assert effect.removal_only
    assert effect.claims_before == 2 and effect.claims_after == 1


def test_a_repair_that_narrows_a_claim_is_not_a_deletion():
    before = _Answer(claims=(_Claim("c1", "the compound blocks hERG in patients"),))
    after = _Answer(claims=(_Claim("c1", "the model predicts hERG block in vitro"),))

    effect = repair_effect(before, after)

    assert effect.reworded_claim_ids == ("c1",)
    assert effect.removed_claim_ids == ()
    assert not effect.removal_only


def test_a_limitation_dropped_during_a_repair_is_named():
    before = _Answer(
        claims=(_Claim("c1", "x"),),
        limitations=(
            _Limitation(LimitationCode.UNCALIBRATED_PROBABILITY),
            _Limitation(LimitationCode.SCREENING_NOT_SAFETY_ASSESSMENT),
        ),
    )
    after = _Answer(
        claims=(_Claim("c1", "x"),),
        limitations=(_Limitation(LimitationCode.UNCALIBRATED_PROBABILITY),),
    )

    effect = repair_effect(before, after)

    assert effect.lost_limitation_codes == ("screening_not_safety_assessment",)
    assert effect.limitations_before == 2 and effect.limitations_after == 1


def test_a_repair_that_adds_a_claim_is_visible_too():
    before = _Answer(claims=(_Claim("c1", "x"),))
    after = _Answer(claims=(_Claim("c1", "x"), _Claim("c2", "y")))
    effect = repair_effect(before, after)
    assert effect.added_claim_ids == ("c2",)
    assert not effect.removal_only


def test_the_relation_counts_travel_with_the_effect():
    before = _Answer(claims=(_Claim("c1", "x"),))
    effect = repair_effect(before, before, relations_before=4, relations_after=1)
    assert effect.relations_before == 4
    assert effect.relations_after == 1
    assert effect.to_dict()["relations_after"] == 1


def test_an_untouched_answer_reports_no_repair():
    answer = _Answer(claims=(_Claim("c1", "x"),))
    effect = repair_effect(answer, answer)
    assert effect.removed_claim_ids == ()
    assert effect.reworded_claim_ids == ()
    assert not effect.removal_only


@dataclass(frozen=True)
class _V2Claim:
    claim_id: str
    text: str
    local_ref: str


def test_a_repaired_answer_is_compared_claim_by_claim_through_local_refs():
    """The audit's case: every submission gets new claim ids, so by id an
    untouched claim reads as one deletion and one addition, and a narrowed
    claim as nothing at all. Lineage follows the draft's own label."""
    before = _Answer(claims=(
        _V2Claim("clm_a1", "IC50 is 1 nM", "potency"),
        _V2Claim("clm_a2", "the compound blocks hERG in patients", "scope"),
        _V2Claim("clm_a3", "it is cardiotoxic", "overreach"),
    ))
    after = _Answer(claims=(
        _V2Claim("clm_b1", "IC50 is 1 nM", "potency"),
        _V2Claim("clm_b2", "the model predicts hERG block in vitro", "scope"),
    ))

    effect = repair_effect(before, after)

    assert effect.removed_claim_ids == ("overreach",)
    assert effect.reworded_claim_ids == ("scope",)
    assert effect.added_claim_ids == ()
    assert not effect.removal_only
