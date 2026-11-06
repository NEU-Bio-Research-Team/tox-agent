"""Verification by claim type (#72): what code settles, and how the checkers are measured."""
from __future__ import annotations

from evals.reviewer_agreement import agreement
from toxagent.application.investigation import claim_verification as cv
from toxagent.domain import exposure_margin


def _margin(ic50=300, cmax=10):
    return exposure_margin.compute(
        exposure_margin.Concentration(ic50, "nM", "context:c1"),
        exposure_margin.Concentration(cmax, "nM", "context:c2"),
    )


def test_a_recorded_margin_is_recomputed_from_its_inputs():
    agrees, why = cv.recompute_margin(_margin())
    assert agrees and "recomputed 30" in why


def test_a_margin_that_does_not_follow_from_its_inputs_is_caught():
    tampered = {**_margin(), "margin": 3.0}
    agrees, why = cv.recompute_margin(tampered)
    assert not agrees and "recorded 3.0" in why


def test_a_margin_with_missing_inputs_is_not_trusted():
    agrees, _ = cv.recompute_margin({"margin": 30, "inputs": {"ic50": {"value": 300}}})
    assert not agrees


def test_a_quoted_measurement_matches_its_record_across_units():
    record = {"value": 300.0, "unit": "nM", "relation": "=", "state": "measured"}
    assert cv.numbers_match_record("hERG IC50 was 0.3 µM in patch clamp.", record)[0]
    assert cv.numbers_match_record("hERG IC50 was 300 nM.", record)[0]
    assert not cv.numbers_match_record("hERG IC50 was 3 µM.", record)[0]
    # No concentration stated: nothing for code to settle.
    assert not cv.numbers_match_record("The compound blocks hERG.", record)[0]


def test_a_bound_cannot_be_quoted_as_a_measurement():
    record = {"value": 30.0, "unit": "uM", "relation": ">", "state": "measured"}
    ok, why = cv.numbers_match_record("IC50 is 30 µM.", record)
    assert not ok and "bound" in why
    assert cv.numbers_match_record("IC50 > 30 µM.", record)[0]


def test_every_type_reviewers_see_has_a_rule():
    reviewed = {cv.EXTRACTED_NUMERIC, cv.DERIVED_NUMERIC, cv.SCOPE_RELATION, cv.SCIENTIFIC_ASSESSMENT}
    assert reviewed <= set(cv.RULES)
    assert set(cv.CLAIM_TYPES) == reviewed | {cv.ARTIFACT_FACT, cv.RECOMMENDATION}


def test_false_accept_and_false_block_are_measured_per_checker_and_type():
    verdicts = [
        {"claim_id": "a", "verdict": "supported", "checked_by": "reviewer", "claim_type": "scope_relation"},
        {"claim_id": "b", "verdict": "partially_supported", "checked_by": "reviewer", "claim_type": "scope_relation"},
        {"claim_id": "c", "verdict": "cannot_judge", "checked_by": "reviewer", "claim_type": "scope_relation"},
        {"claim_id": "d", "verdict": "supported", "checked_by": "code", "claim_type": "derived_numeric"},
        {"claim_id": "e", "verdict": "supported", "checked_by": "reviewer", "claim_type": "scope_relation"},
    ]
    gold = {"a": "not_supported", "b": "supported", "c": "supported", "d": "supported",
            "e": "partially_supported"}
    table = {(r["checker"], r["claim_type"]): r for r in agreement(verdicts, gold)["table"]}
    scope = table[("reviewer", "scope_relation")]
    assert (scope["false_accept"], scope["false_block"], scope["cannot_judge"]) == (2, 1, 1)
    assert scope["false_accept_rate"] == 1.0 and scope["false_block_rate"] == 0.5
    assert table[("code", "derived_numeric")]["false_block"] == 0
    assert agreement(verdicts, {})["unlabelled"] == 5
