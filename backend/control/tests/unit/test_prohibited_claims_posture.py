"""W6/RC-07: a development posture's wording must not conflate a scoped R&D
recommendation with a safety verdict this product never issues.

Bilingual corpus (VI/EN), built before the gate it tests (ADS remaining-plan
PR-5 step 1): every "must fail" case below was confirmed to actually fail
against a no-op stub (always returns no violations) before
``validate_posture_wording`` existed, so this corpus is known to discriminate
rather than passing by construction. The "must pass" half doubles as a
regression guard on the five existing prohibited_claims gates: this new gate
must not tighten what already passes.
"""
from __future__ import annotations

import pytest

from toxagent.domain.development_posture import DevelopmentPosture, DevelopmentScope, PostureValue
from toxagent.validation.prohibited_claims import (
    validate_answer_markdown,
    validate_posture_wording,
)

CLAIM = "clm_" + "a" * 32


def _posture(rationale: str, *, value=PostureValue.HOLD, conditions=()) -> DevelopmentPosture:
    return DevelopmentPosture(
        value=value,
        scope=DevelopmentScope.DRUG_CANDIDATE,
        confidence_band="moderate",
        basis_claim_ids=(CLAIM,),
        rationale=rationale,
        conditions=conditions,
        recommended_next_steps=("Confirm hERG signal with an orthogonal assay",),
    )


# --- must fail: safety/posture conflation, not caught by the existing gates -


CONFLATION_CASES = [
    "This compound appears safe to proceed with further development.",
    "We recommend proceeding safely with additional screening.",
    "Chất này an toàn để tiếp tục phát triển.",
    "Safe enough to advance to the next development stage.",
    "An toàn, có thể tiếp tục phát triển sản phẩm.",
]


@pytest.mark.parametrize("rationale", CONFLATION_CASES)
def test_safety_posture_conflation_is_rejected(rationale: str) -> None:
    violations = validate_posture_wording(rationale, _posture(rationale))
    assert any(v.code == "safety_posture_conflation" for v in violations), violations


# --- must pass: a scoped R&D posture with no safety word nearby ------------


SCOPED_POSTURE_CASES = [
    "We recommend holding this candidate pending additional hERG confirmation.",
    "Khuyến nghị tạm dừng phát triển để bổ sung dữ liệu hERG.",
    "Deprioritize in this R&D program given the applicability limitation on the predictor.",
    "Given the current evidence, screening supports moving to the next verification step.",
]


@pytest.mark.parametrize("rationale", SCOPED_POSTURE_CASES)
def test_a_scoped_posture_with_no_safety_word_nearby_is_not_flagged(rationale: str) -> None:
    violations = validate_posture_wording(rationale, _posture(rationale))
    assert violations == []


# --- regression: existing gates still catch absolute safety claims ---------


ABSOLUTE_SAFETY_CASES = [
    "This compound is safe.",
    "Chất này an toàn.",
    "The compound is considered safe for clinical use.",
]


@pytest.mark.parametrize("text", ABSOLUTE_SAFETY_CASES)
def test_existing_safety_verdict_gate_still_catches_absolute_claims(text: str) -> None:
    violations = validate_answer_markdown(text)
    assert any(v.code == "safety_verdict_out_of_scope" for v in violations), violations


# --- regression: negated/hedged limitations still pass ---------------------


NEGATED_CASES = [
    "This result does not provide an overall toxicity score.",
    "The model does not establish that the compound is safe.",
]


@pytest.mark.parametrize("text", NEGATED_CASES)
def test_existing_negation_awareness_is_unaffected(text: str) -> None:
    assert validate_answer_markdown(text) == []
    assert validate_posture_wording(text, _posture(text)) == []


# --- the new gate also scans posture.conditions, not only rationale --------


def test_a_conflation_hidden_in_a_condition_is_also_caught() -> None:
    posture = _posture(
        "Predictor and evidence disagree on applicability.",
        conditions=("proceed only once the compound is confirmed safe",),
    )
    violations = validate_posture_wording(posture.rationale, posture)
    assert any(v.code == "safety_posture_conflation" for v in violations)
