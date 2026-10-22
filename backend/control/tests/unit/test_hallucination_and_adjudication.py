"""Hallucination (atomic claims) and predictor adjudication, on hand-counted
controls: every number below can be checked by counting the rows."""
from __future__ import annotations

import csv

import pytest

from evals.investigation.adjudication import adjudication
from evals.investigation.claims import hallucination, read_claim_sheets
from evals.investigation.scorecard import GradeError

KEY = {"responses": [
    {"case_id": "c1", "response_id": "R01", "system_id": "D", "trial": 1},
    {"case_id": "c1", "response_id": "R02", "system_id": "B", "trial": 1},
    {"case_id": "c2", "response_id": "R01", "system_id": "D", "trial": 1},
    {"case_id": "c2", "response_id": "R02", "system_id": "B", "trial": 1},
    {"case_id": "c3", "response_id": None, "system_id": "B", "trial": 1,
     "missing_reason": "error"},
    {"case_id": "c3", "response_id": "R01", "system_id": "D", "trial": 1},
]}


def _claim(grader, case, response, label, claim="x"):
    return {"grader_id": grader, "case_id": case, "response_id": response, "turn": 1,
            "claim": claim, "label": label, "source": "", "note": ""}


def test_precision_counts_checkable_claims_and_no_claim_replies_are_not_perfect():
    claims = [
        _claim("g1", "c1", "R01", "supported"), _claim("g1", "c1", "R01", "supported"),
        _claim("g1", "c1", "R01", "unsupported", "invented 12 nM"),
        _claim("g1", "c1", "R01", "not_checkable"),
        _claim("g1", "c2", "R01", "no_claims"),
        _claim("g1", "c3", "R01", "contradicted", "says non-blocker; source says 0.3 uM"),
        _claim("g1", "c1", "R02", "supported"),
        # c2/R02 is not graded by g1
    ]
    out = hallucination(key=KEY, claims=claims)["by_grader"]["g1"]
    d, b = out["D"], out["B"]
    assert (d["responses"], d["graded"], d["no_claims"], d["with_checkable"]) == (3, 3, 1, 2)
    assert (d["checkable"], d["supported"], d["unsupported"], d["contradicted"]) == (4, 2, 1, 1)
    assert d["claim_precision"] == 0.5 and d["hallucination_rate"] == 0.5
    assert (d["with_hallucination"], d["responses_with_hallucination_rate"]) == (2, 1.0)
    assert d["not_checkable"] == 1
    # The missing reply (c3/B) is not a response here; c2/R02 is ungraded.
    assert (b["responses"], b["graded"], b["claim_precision"]) == (2, 1, 1.0)


def test_graders_are_compared_per_reply_never_claim_by_claim():
    claims = [_claim("g1", "c1", "R01", "unsupported"), _claim("g2", "c1", "R01", "supported"),
              _claim("g1", "c1", "R02", "supported"), _claim("g2", "c1", "R02", "supported"),
              _claim("g2", "c1", "R02", "supported")]
    agreement = hallucination(key=KEY, claims=claims)["agreement"]
    assert (agreement["replies_graded_by_two"], agreement["agree_on_any_hallucination"]) == (2, 1)
    assert agreement["disagreements"] == ["c1/R01"]


def test_an_unknown_label_is_refused(tmp_path):
    path = tmp_path / "claims.csv"
    with path.open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["grader_id", "case_id", "response_id", "turn", "claim_index", "claim",
                    "label", "source", "note"])
        w.writerow(["g", "c1", "R01", "1", "1", "x", "mostly_true", "", ""])
    with pytest.raises(GradeError):
        read_claim_sheets([path])


GOLD = {
    # predictor right, decidable: corroborate expected
    "c1": {"adjudication": {"predictor_call": "positive", "verified": "positive",
                            "decidable_from_corpus": True, "expected_stance": "corroborate"}},
    # predictor wrong, decidable: challenge expected
    "c2": {"adjudication": {"predictor_call": "negative", "verified": "positive",
                            "decidable_from_corpus": True, "expected_stance": "challenge"}},
    # not decidable: insufficient expected
    "c3": {"adjudication": {"predictor_call": "positive", "verified": "negative",
                            "decidable_from_corpus": False, "expected_stance": "insufficient"}},
    "c4": {},  # an ordinary case is skipped
}


def _grade(grader, case, response, stance, label, grounded=""):
    return {"grader_id": grader, "case_id": case, "response_id": response,
            "agent_stance": stance, "agent_label": label, "evidence_grounded": grounded}


def test_adjudication_rates_on_a_hand_counted_set():
    grades = [
        _grade("g", "c1", "R01", "corroborate", "positive", "1"),   # D right
        _grade("g", "c1", "R02", "challenge", "negative", "0"),     # B false alarm
        _grade("g", "c2", "R01", "challenge", "positive", "1"),     # D corrects the error
        _grade("g", "c2", "R02", "corroborate", "negative", "0"),   # B uncaught error
        _grade("g", "c3", "R01", "corroborate", "positive", "0"),   # D unsupported corroboration
    ]
    out = adjudication(key=KEY, grades=grades, gold=GOLD)["by_grader"]["g"]
    d, b = out["D"], out["B"]
    assert (d["cases"], d["stance_correct"]) == (3, 2)
    assert (d["correction_rate"], d["uncaught_error_rate"]) == (1.0, 0.0)
    assert (d["corroboration_rate"], d["false_alarm_rate"]) == (1.0, 0.0)
    assert d["unsupported_corroboration_rate"] == 1.0
    assert (d["agent_label_accuracy"], d["predictor_accuracy"], d["lift"]) == (1.0, 0.5, 0.5)
    assert (d["grounded_positions"], d["positions"]) == (2, 3)
    # B: the missing c3 reply is stance "none", kept in the denominator.
    assert (b["cases"], b["stance_correct"], b["stance_counts"]["none"]) == (3, 0, 1)
    assert (b["uncaught_error_rate"], b["false_alarm_rate"]) == (1.0, 1.0)
    assert (b["agent_label_accuracy"], b["lift"]) == (0.0, -0.5)


def test_a_set_without_adjudication_gold_has_no_section():
    assert adjudication(key=KEY, grades=[], gold={"c4": {}}) == {}
