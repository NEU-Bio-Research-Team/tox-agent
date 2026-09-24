"""domain/development_posture.py — the decision_support R&D recommendation
shape (ADS plan section 10.1, ADR 0010)."""
from __future__ import annotations

import pytest

from toxagent.domain.development_posture import (
    DevelopmentPosture,
    DevelopmentScope,
    PostureValue,
)

CLAIM_1 = "clm_" + "1" * 32
CLAIM_2 = "clm_" + "2" * 32


def _posture(**overrides):
    defaults = dict(
        value=PostureValue.PROCEED,
        scope=DevelopmentScope.DRUG_CANDIDATE,
        confidence_band="moderate",
        basis_claim_ids=(CLAIM_1,),
        rationale="Predictor and cited literature agree within scope.",
    )
    defaults.update(overrides)
    return DevelopmentPosture(**defaults)


def test_proceed_hold_and_deprioritize_need_a_basis():
    for value in (PostureValue.PROCEED, PostureValue.HOLD, PostureValue.DEPRIORITIZE):
        with pytest.raises(ValueError, match="basis"):
            _posture(
                value=value, basis_claim_ids=(),
                recommended_next_steps=("Run an in-vitro confirmatory assay.",),
            )


def test_insufficient_and_not_applicable_do_not_need_a_basis():
    DevelopmentPosture(
        value=PostureValue.INSUFFICIENT, scope=DevelopmentScope.UNKNOWN,
        confidence_band="not_assessed", basis_claim_ids=(),
        rationale="Not enough evidence exists to choose a posture yet.",
        recommended_next_steps=("Search for direct assay data on this scaffold.",),
    )
    DevelopmentPosture(
        value=PostureValue.NOT_APPLICABLE, scope=DevelopmentScope.UNKNOWN,
        confidence_band="not_assessed", basis_claim_ids=(),
        rationale="The question asked is not a development-posture question.",
    )


def test_hold_and_insufficient_need_recommended_next_steps():
    with pytest.raises(ValueError, match="recommended_next_steps"):
        _posture(value=PostureValue.HOLD, recommended_next_steps=())


def test_proceed_does_not_require_next_steps():
    _posture(value=PostureValue.PROCEED, recommended_next_steps=())


def test_rationale_cannot_be_blank():
    with pytest.raises(ValueError, match="rationale"):
        _posture(rationale="   ")


def test_an_unknown_confidence_band_is_rejected():
    with pytest.raises(ValueError, match="confidence_band"):
        _posture(confidence_band="very sure")


def test_claim_ids_must_be_well_formed():
    with pytest.raises(ValueError):
        _posture(basis_claim_ids=("not-a-claim-id",))


def test_contrary_claim_ids_are_also_validated_but_optional():
    posture = _posture(contrary_claim_ids=(CLAIM_2,))
    assert posture.contrary_claim_ids == (CLAIM_2,)
    payload = posture.to_dict()
    assert payload["value"] == "proceed"
    assert payload["contrary_claim_ids"] == [CLAIM_2]


def test_posture_values_match_the_plan():
    assert {member.value for member in PostureValue} == {
        "proceed", "hold", "deprioritize", "insufficient", "not_applicable",
    }
