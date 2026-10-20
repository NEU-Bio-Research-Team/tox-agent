"""The ToxBench scorer is itself under test.

A score is evidence only once the instrument has been shown to catch mistakes.
These tests run deliberately wrong answers through the real scorer and assert
both directions of its error: wrong answers it must flag, and acceptable
answers it must leave alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.benchmark_comparative.metrics import (
    SystemScorecard,
    build_comparative_report,
    limitation_is_stated,
)
from evals.benchmark_comparative.positive_controls import (
    CONTROLS,
    DATASET,
    evaluate_control,
    load_cases,
    run_positive_controls,
)


def test_every_control_answers_a_case_that_exists():
    cases = load_cases()
    for control in CONTROLS:
        assert control.case_id in cases, control.control_id


def test_the_scorer_catches_every_defect_it_claims_to_cover():
    """A deterministic control the scorer misses is a defect in the scorer,
    not an acceptable gap."""
    report = run_positive_controls()
    rates = report["evaluator_error_rates"]
    assert rates["deterministic_false_negatives"] == []
    assert rates["deterministic_false_negative_rate"] == 0.0


def test_the_scorer_does_not_flag_an_acceptable_answer():
    report = run_positive_controls()
    assert report["evaluator_error_rates"]["false_positives"] == []


def test_naming_a_forbidden_phrase_in_order_to_refuse_it_is_not_a_finding():
    cases = load_cases()
    control = next(
        c for c in CONTROLS if c.control_id == "nc_refuses_an_injected_instruction"
    )
    outcome = evaluate_control(control, cases[control.case_id])
    assert outcome.findings == []


def test_asserting_the_score_is_calibrated_does_not_state_the_limitation():
    """The pattern looks for the word "calibrat", so a response claiming the
    probability *is* calibrated used to score as having stated the
    uncalibrated-probability limitation."""
    assert not limitation_is_stated(
        "uncalibrated_probability",
        ["The probability is perfectly calibrated, so 0.94 means a 94% chance."],
    )
    # a denial anywhere keeps the limitation counted, so hedged answers are safe
    for hedged in (
        "This score has not been calibrated.",
        "The output is uncalibrated.",
        "Scores are calibrated for ranking but the model is not calibrated for risk.",
    ):
        assert limitation_is_stated("uncalibrated_probability", [hedged]), hedged


def test_the_uncovered_defect_classes_are_named_rather_than_assumed_absent():
    """Four defect classes need a source read against a claim. The score does
    not cover them, and the report has to say so."""
    gaps = run_positive_controls()["known_gaps"]
    assert set(gaps["uncovered_findings"]) == {
        "unverifiable_citation",
        "unsupported_claim",
        "scope_mismatch",
        "overinterpretation",
    }


def _scorecard(name: str, *, strict: float, committed: float, coverage: float):
    return SystemScorecard(
        system_name=name,
        hallucination_metrics={"hallucination_rate": 0.0},
        predictive_metrics={
            "herg_strict_accuracy": strict,
            "herg_classification_accuracy": committed,
            "herg_coverage": coverage,
            "herg_labelled_count": 50,
        },
        faithfulness_metrics={},
        safety_metrics={"overall_pass_rate": 1.0},
    )


def test_abstaining_on_the_hard_cases_does_not_win_predictive_accuracy():
    """36/36 right at coverage 36/50 is accuracy-when-committed 1.0 and strict
    accuracy 0.72. It cannot outrank a system that answered all 50."""
    selective = _scorecard("selective", strict=0.72, committed=1.0, coverage=0.72)
    committed = _scorecard("committed", strict=0.90, committed=0.90, coverage=1.0)

    report = build_comparative_report([selective, committed])

    assert report["winners"]["predictive_accuracy"] == "committed"
    basis = report["predictive_accuracy_basis"]
    assert basis["ranked_on"] == "herg_strict_accuracy"
    assert basis["per_system"]["selective"]["coverage"] == 0.72
    assert basis["per_system"]["selective"]["accuracy_when_committed"] == 1.0


def test_the_dataset_is_labelled_as_a_development_set():
    """It was used to develop against and its results have been seen, so it is
    a dev/regression set, not an independent final test."""
    data = json.loads(Path(DATASET).read_text(encoding="utf-8"))
    assert data["intended_use"] == "development_and_regression"
    assert "not an independent final test set" in data["description"]


def test_the_provenance_gaps_in_the_gold_labels_are_recorded_not_guessed():
    data = json.loads(Path(DATASET).read_text(encoding="utf-8"))
    audit = data["provenance_audit"]
    missing = [
        case["case_id"]
        for case in data["cases"]
        if not str(case["ground_truth"].get("source_id") or "").strip()
    ]
    assert audit["cases_without_source_id"] == len(missing)
    assert audit["cases_total"] == len(data["cases"])
    for field in ("assay_id", "readout", "unit", "comparator", "verified_by"):
        assert field in audit["fields_absent_from_every_case"], field


@pytest.mark.parametrize("control", CONTROLS, ids=lambda c: c.control_id)
def test_each_control_states_what_it_is_for(control):
    assert control.defect
    assert control.detectable_by in {"deterministic", "semantic"}
