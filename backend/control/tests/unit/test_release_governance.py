"""Wave 5: signed multi-dimensional scorecard, regression drafts, judge drift."""
from __future__ import annotations

import json

from evals.calibration import drift
from evals.regression_from_run import draft_from, scrub
from evals.scorecard import build, sign, verify
from evals.task_packs import validate_task


def _write_run(tmp_path, stamp, *, runtime="opencode", rows, summary=None, trials=3, blockers=()):
    manifest = {
        "runtime_kind": runtime, "trial_count": trials, "eval_suite_hash": "h",
        "summary": {"invalid": 0, "conservation_violations": [], "infra_error": 0,
                    "not_evaluated_packs": [], **(summary or {})},
        "release_evidence": {"eligible": not blockers, "blockers": list(blockers)},
        "effective_product": {"effective_product_hash": "p"},
    }
    (tmp_path / f"manifest-{stamp}.json").write_text(json.dumps(manifest))
    (tmp_path / f"results-{stamp}.json").write_text(json.dumps(rows))
    return tmp_path / f"manifest-{stamp}.json"


def _row(task_id, status, pack="core", critical=False, outcome="first_pass", semantic=None):
    return {"task_id": task_id, "status": status, "capability_pack": pack, "critical": critical,
            "trials": [{"answer_outcome": outcome, **({"semantic": semantic} if semantic else {})}]}


GOOD_CONFORMANCE = {"schema_version": "control-plane-conformance-v1", "summary": {"fail": 0}}
CALIBRATED = {"schema_version": "judge-calibration-v1",
              "rubrics": {"evidence-synthesis@1": {"calibrated": True}}}
SIGNOFF = {"sme": {"name": "Reviewer", "date": "2026-09-16"}}


def test_a_complete_release_is_go_and_one_critical_failure_is_no_go(tmp_path):
    manifest = _write_run(tmp_path, "a", rows=[
        _row("core-1", "pass"), _row("sec-1", "pass", pack="security", critical=True),
    ])
    card = build(level="release", manifests=[manifest], conformance=GOOD_CONFORMANCE,
                 calibration=CALIBRATED, signoff=SIGNOFF)
    assert card["decision"] == "go", json.dumps(card["dimensions"], indent=1)

    failing = _write_run(tmp_path, "b", rows=[
        _row("core-1", "pass"), _row("sec-1", "fail", pack="security", critical=True),
    ])
    card = build(level="release", manifests=[failing], conformance=GOOD_CONFORMANCE,
                 calibration=CALIBRATED, signoff=SIGNOFF)
    assert card["decision"] == "no_go"
    assert card["dimensions"]["safety_reliability"]["critical_tasks_all_pass"]["verdict"] == "fail"


def test_missing_inputs_are_insufficient_data_never_pass(tmp_path):
    manifest = _write_run(tmp_path, "a", rows=[_row("core-1", "pass")])
    card = build(level="release", manifests=[manifest], conformance=None, calibration=None,
                 signoff=None)
    verdicts = {k: g["verdict"] for d in card["dimensions"].values() for k, g in d.items()}
    assert verdicts["drills"] == "insufficient_data"
    assert verdicts["judge_calibrated"] == "insufficient_data"
    assert verdicts["sme_signoff"] == "insufficient_data"
    assert card["decision"] == "no_go"


def test_a_fallback_heavy_run_still_reports_capability_separately(tmp_path):
    manifest = _write_run(tmp_path, "a", rows=[_row("q", "pass", outcome="fallback")])
    card = build(level="nightly", manifests=[manifest], conformance=GOOD_CONFORMANCE,
                 calibration=None, signoff=None)
    detail = card["dimensions"]["capability"]["outcomes_reported"]["detail"]
    assert detail["fallback_rate"] == 1.0 and detail["first_pass_rate"] == 0.0


def test_a_signature_detects_tampering():
    card = {"schema_version": "x", "decision": "no_go", "dimensions": {}}
    signed = sign(card, "k" * 32)
    assert verify(signed, "k" * 32)
    assert not verify({**signed, "decision": "go"}, "k" * 32)
    assert not verify(signed, "other-key")


def test_a_production_run_becomes_a_scrubbed_review_draft():
    run = {"run_id": "run_" + "a" * 32, "intent": "decision_support", "status": "completed",
           "tool_calls": [{"tool_name": "get_analysis_slice"}]}
    messages = [
        {"role": "user", "parts": [{"type": "text", "content": {"text": "Analyse this"}},
                                   {"type": "molecule", "content": {"smiles": "c1ccccc1"}}]},
        {"role": "assistant", "parts": [{"type": "text", "content": {"text": "secret reply"}}]},
        {"role": "user", "parts": [{"type": "text", "content": {
            "text": "Email me at a.b@example.com or +84 912 345 678 about ses_" + "b" * 32}}]},
    ]
    draft = draft_from(task_id="ads-99-draft", run=run, messages=messages,
                       answer={"is_fallback": True, "candidate_generation": 2})
    text = json.dumps(draft)
    assert "a.b@example.com" not in text and "912 345" not in text and "b" * 32 not in text
    assert "a" * 32 not in text and "secret reply" not in text
    assert draft["conversation"][0]["molecule"] == {"smiles": "c1ccccc1"}
    assert "fallback" in draft["rationale"]
    problems = validate_task(draft)
    assert problems == [], problems


def test_scrub_keeps_scientific_text():
    assert scrub("hERG IC50 of 3.2 uM at 37 C") == "hERG IC50 of 3.2 uM at 37 C"


def test_judge_drift_is_flagged():
    before = {"rubrics": {"r@1": {"dimensions": {"claim_support": {"kappa": 0.8, "false_pass_rate": 0.01}}}}}
    after = {"rubrics": {"r@1": {"dimensions": {"claim_support": {"kappa": 0.62, "false_pass_rate": 0.05}}}}}
    report = drift(before, after)
    assert report["drifted"] == ["r@1"] and len(report["findings"]["r@1"]) == 2
    assert drift(before, before)["drifted"] == []
