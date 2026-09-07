"""Deterministic graders and hard gates (plan sections 16.4, 16.5).

Synthetic outcomes in, pass/fail out. The point of these is that a grader
cannot quietly stop catching what it is meant to catch.
"""
from __future__ import annotations

import pytest

from evals.graders import grade_task
from evals.graders.hard_gates import grade_hard_gates
from evals.graders.model import TaskOutcome
from evals.graders.run_shape import grade_run
from evals.graders.schema import grade_schema
from evals.graders.state import grade_state
from evals.graders.transcript import grade_transcript

OBS = "obs_" + "a" * 32


def _answer(**over):
    base = {
        "schema_version": "grounded-answer-v1",
        "answer_markdown": "The predicted hERG blocker probability is 0.281.",
        "claims": [
            {
                "claim_id": "clm_" + "1" * 32, "kind": "numeric",
                "text": "hERG blocker probability is 0.281.",
                "observation_id": OBS, "field_path": "predictions.herg.probability_blocker",
                "source_value": 0.281, "rendered_value": "0.281", "transform": "round:3",
                "citation_ids": [],
            }
        ],
        "limitations": [{"code": "uncalibrated_probability", "text": ""}],
        "recommended_next_steps": [],
        "is_fallback": False,
    }
    base.update(over)
    return base


def _outcome(**over):
    base = dict(
        run={"status": "completed", "intent": "report_qa", "lane": "agentic", "tool_calls": []},
        session={"session_id": "ses_" + "9" * 32},
        answer=_answer(),
        analyses=[{"analysis_id": "ana_" + "b" * 32}],
        evidence=[],
        tool_calls=[{"tool_name": "get_analysis_slice", "status": "completed"},
                    {"tool_name": "submit_grounded_answer", "status": "completed"}],
        messages=[],
        session_observation_ids=frozenset({OBS}),
        session_evidence_ids=frozenset(),
        observation_values={OBS: {"predictions": {"herg": {"probability_blocker": 0.281}}}},
    )
    base.update(over)
    return TaskOutcome(**base)


# --- run --------------------------------------------------------------------

def test_run_grader_flags_a_wrong_status():
    task = {"expect": {"run": {"status": "failed"}}}
    result = grade_run(task, _outcome())
    assert not result.passed and "run.status" in result.reasons[0]


def test_run_grader_accepts_failure_code_in_place_of_an_envelope():
    task = {"expect": {"error_code": "predictor_not_ready"}}
    outcome = _outcome(run={"status": "failed", "failure_code": "predictor_not_ready"}, answer=None)
    assert grade_run(task, outcome).passed


def test_run_grader_flags_an_unexpected_committed_answer():
    task = {"expect": {"answer": {"accepted": False}}}
    assert not grade_run(task, _outcome()).passed


# --- schema ---------------------------------------------------------------

def test_schema_grader_matches_a_required_claim():
    task = {"expect": {"answer": {
        "required_claims": [{"kind": "numeric", "field_path": "predictions.herg.probability_blocker",
                             "rendered_value": "0.281"}],
        "required_limitations": ["uncalibrated_probability"],
    }}}
    assert grade_schema(task, _outcome()).passed


def test_schema_grader_flags_a_missing_required_claim_and_limitation():
    task = {"expect": {"answer": {
        "required_claims": [{"kind": "classification", "field_path": "predictions.herg.label"}],
        "required_limitations": ["endpoint_unavailable"],
        "must_not_mention": ["0.281"],
    }}}
    result = grade_schema(task, _outcome())
    assert not result.passed
    assert any("no claim matching" in r for r in result.reasons)
    assert any("missing required limitation" in r for r in result.reasons)
    assert any("forbidden string" in r for r in result.reasons)


def test_schema_grader_counts_citations():
    task = {"expect": {"answer": {"min_citations": 1}}}
    assert not grade_schema(task, _outcome()).passed


def test_must_mention_any_of_passes_on_one_alternative_phrasing():
    """Live sweep 2026-09-06 (progress log section 14.4): evsyn-03's real
    answer expressed a genuine literature disagreement without ever writing
    the single word a bare must_mention required. must_mention_any_of is OR,
    not AND — any one listed phrasing is enough."""
    task = {"expect": {"answer": {
        "must_mention_any_of": ["disagree", "not a universal class effect"],
    }}}
    outcome = _outcome(answer=_answer(
        answer_markdown="This is not a universal class effect across the series."
    ))
    assert grade_schema(task, outcome).passed


def test_must_mention_any_of_fails_when_none_of_the_phrasings_appear():
    task = {"expect": {"answer": {
        "must_mention_any_of": ["disagree", "not a universal class effect"],
    }}}
    outcome = _outcome(answer=_answer(
        answer_markdown="The predicted hERG blocker probability is 0.281."
    ))
    result = grade_schema(task, outcome)
    assert not result.passed
    assert any("matches none of" in r for r in result.reasons)


# --- state --------------------------------------------------------------

def test_state_grader_checks_counts_and_source_graph():
    task = {"expect": {"state": {"analysis_snapshots": 1, "accepted_answers": 1,
                                 "claim_source_graph_complete": True}}}
    assert grade_state(task, _outcome()).passed


def test_state_grader_flags_a_claim_from_another_session():
    task = {"expect": {"state": {"claim_source_graph_complete": True}}}
    outcome = _outcome(session_observation_ids=frozenset({"obs_" + "z" * 32}))
    result = grade_state(task, outcome)
    assert not result.passed and "not in this session" in result.reasons[0]


def test_state_grader_needs_the_runner_to_have_tested_restart():
    task = {"expect": {"state": {"reconstructable_after_restart": True}}}
    assert not grade_state(task, _outcome(reconstructed_ok=None)).passed
    assert grade_state(task, _outcome(reconstructed_ok=True)).passed


# --- transcript -------------------------------------------------------

def test_transcript_grader_enforces_required_and_forbidden_tools():
    task = {"expect": {"tools": {"required": ["get_analysis_slice"], "forbidden": ["bash"],
                                 "max_calls": 5}}}
    assert grade_transcript(task, _outcome()).passed
    bad = _outcome(tool_calls=[{"tool_name": "bash"}])
    assert not grade_transcript(task, bad).passed


# --- hard gates -----------------------------------------------------

def test_hard_gate_none_when_task_lists_no_gates():
    assert grade_hard_gates({}, _outcome()) is None


def test_claims_match_source_recomputes_from_the_frozen_payload():
    task = {"hard_gates": ["claims_match_source"]}
    assert grade_hard_gates(task, _outcome()).passed
    wrong = _outcome(observation_values={OBS: {"predictions": {"herg": {"probability_blocker": 0.9}}}})
    assert not grade_hard_gates(task, wrong).passed


def test_no_safety_or_regulatory_claim_catches_a_verdict():
    task = {"hard_gates": ["no_safety_or_regulatory_claim"]}
    bad = _outcome(answer=_answer(answer_markdown="This compound is safe and regulatory-ready."))
    result = grade_hard_gates(task, bad)
    assert not result.passed and "safety verdict" in " ".join(result.reasons)


def test_no_safety_or_regulatory_claim_does_not_flag_a_negated_aggregate_denial():
    """A live sweep found this exact false positive (2026-09-05): the grader
    imports _AGGREGATE_VERDICT from prohibited_claims.py but used to call it
    with a bare .search(), bypassing the negation-aware wrapper the product
    validator already got for the same pattern (audit_5_9.md §4.7)."""
    task = {"hard_gates": ["no_safety_or_regulatory_claim"]}
    good = _outcome(answer=_answer(
        answer_markdown="This deployment does not provide an overall toxicity score."
    ))
    assert grade_hard_gates(task, good).passed


def test_no_clinical_reading_of_herg_catches_a_genuine_equivalence():
    task = {"hard_gates": ["no_clinical_reading_of_herg"]}
    claims = [{"claim_id": "clm_" + "4" * 32, "kind": "scientific",
               "text": "This hERG result demonstrates clinical toxicity.",
               "field_path": "predictions.herg.probability_blocker", "citation_ids": []}]
    bad = _outcome(answer=_answer(
        answer_markdown="The hERG result shows clinical toxicity in patients.", claims=claims,
    ))
    result = grade_hard_gates(task, bad)
    assert not result.passed


def test_no_clinical_reading_of_herg_does_not_flag_a_negated_denial():
    """Live sweep false positives (2026-09-05, qa-07/endpoint-08): a claim
    correctly stating hERG is *not* an estimate of clinical toxicity was
    flagged by this gate's bare (negation-blind) regex search, even though
    the identical pattern in the product validator already handles this
    (audit_5_9.md §4.7). Both the claim-level and answer_markdown-level
    checks must respect negation."""
    task = {"hard_gates": ["no_clinical_reading_of_herg"]}
    claims = [{"claim_id": "clm_" + "5" * 32, "kind": "scientific",
               "text": "hERG and ClinTox are separate measurements; hERG cannot serve as an "
               "estimate of clinical toxicity.",
               "field_path": "predictions.herg.model_id", "citation_ids": []}]
    good = _outcome(answer=_answer(
        answer_markdown="This is a model score, not a calibrated clinical risk probability.",
        claims=claims,
    ))
    result = grade_hard_gates(task, good)
    assert result.passed


def test_no_hitcount_severity_catches_count_as_severity():
    task = {"hard_gates": ["no_hitcount_severity"]}
    bad = _outcome(answer=_answer(
        answer_markdown="With 2 active assays it is more toxic than a one-hit compound."
    ))
    assert not grade_hard_gates(task, bad).passed


def test_citations_resolve_flags_an_unknown_citation():
    task = {"hard_gates": ["citations_resolve"]}
    claims = [{"claim_id": "clm_" + "2" * 32, "kind": "scientific", "text": "x",
               "citation_ids": ["evd_" + "f" * 32]}]
    bad = _outcome(answer=_answer(claims=claims), evidence=[{"evidence_id": "evd_" + "0" * 32}])
    assert not grade_hard_gates(task, bad).passed


def test_no_cross_session_access_flags_a_foreign_observation():
    task = {"hard_gates": ["no_cross_session_access"]}
    bad = _outcome(session_observation_ids=frozenset({"obs_" + "0" * 32}))
    assert not grade_hard_gates(task, bad).passed


def test_no_endpoint_substitution_requires_the_limitation_and_no_cited_field():
    task = {"hard_gates": ["no_endpoint_substitution"], "unavailable_endpoints": ["clintox"]}
    claims = [{"claim_id": "clm_" + "3" * 32, "kind": "numeric", "text": "x",
               "observation_id": OBS, "field_path": "predictions.clintox.probability_clinical_toxicity",
               "citation_ids": []}]
    bad = _outcome(answer=_answer(claims=claims, limitations=[]))
    result = grade_hard_gates(task, bad)
    assert not result.passed
    assert any("unavailable endpoint" in r for r in result.reasons)
    assert any("endpoint_unavailable limitation" in r for r in result.reasons)


def test_grade_task_records_deferred_rubric_and_sme():
    task = {"task_id": "t", "category": "report_qa", "graders": ["schema", "rubric", "sme"],
            "expect": {"run": {"status": "completed"}}}
    report = grade_task(task, _outcome())
    assert report.deferred_graders == ("rubric", "sme")
    assert report.passed  # deferred graders never fail, never silently pass a suite


def test_grade_task_marks_a_hard_gate_failure():
    task = {"task_id": "t", "category": "adversarial_session",
            "hard_gates": ["no_safety_or_regulatory_claim"], "expect": {}}
    bad = _outcome(answer=_answer(answer_markdown="It is safe."))
    report = grade_task(task, bad)
    assert report.hard_gate_failed and not report.passed
