"""G5 is decided from data, and a missing number is never a pass (PR-18)."""
from __future__ import annotations

import json

from evals import gates

GOOD_TELEMETRY = {
    "window_start": "2026-10-01T00:00:00+00:00",
    "window_end": "2026-10-08T06:00:00+00:00",
    "model_id": "pinned-model",
    "qa_latency_p50_s": 12.0,
    "qa_latency_p95_s": 27.0,
    "report_latency_p50_s": 48.0,
    "report_latency_p95_s": 110.0,
    "report_cumulative_input_tokens_p50": 7200,
    "first_pass_acceptance": 0.96,
    "duplicate_normalized_usage": 0,
    "usage_unknown_fraction": 0.01,
    "semantic_contradictions": 0,
    "promoted_evidence_precision": 0.93,
    "happy_path_report_tool_failures": 0,
    "cancel_settlement_p95_s": 3.2,
}

LIVE_MANIFEST = {
    "generated_at": "2026-10-08T07:00:00+00:00",
    "runtime_kind": "opencode",
    "eval_suite_hash": "abc",
    "summary": {"executed": 50, "pass_rate": 0.94, "critical_executed": 10, "critical_all_pass": True},
}

SIGNOFF = {
    key: {"by": "reviewer", "at": "2026-10-08"}
    for key in ("sme_semantic_fixtures", "rollback_rehearsal", "runbooks", "fault_drills")
}


def _verdicts(report):
    return {item["key"]: item["verdict"] for item in report["criteria"]}


def test_every_criterion_met_is_a_go():
    report = gates.evaluate(GOOD_TELEMETRY, [LIVE_MANIFEST], SIGNOFF)
    assert report["decision"] == "go", [c for c in report["criteria"] if c["verdict"] != "pass"]
    assert report["eval_manifest"]["eval_suite_hash"] == "abc"


def test_a_missing_number_is_insufficient_data_not_a_pass():
    telemetry = dict(GOOD_TELEMETRY)
    del telemetry["promoted_evidence_precision"]
    report = gates.evaluate(telemetry, [LIVE_MANIFEST], SIGNOFF)
    assert _verdicts(report)["promoted_evidence_precision"] == "insufficient_data"
    assert report["decision"] == "no_go"


def test_an_alpha_shorter_than_seven_days_cannot_pass():
    telemetry = {**GOOD_TELEMETRY, "window_end": "2026-10-05T00:00:00+00:00"}
    report = gates.evaluate(telemetry, [LIVE_MANIFEST], SIGNOFF)
    assert _verdicts(report)["alpha_window_days"] == "fail"
    assert report["decision"] == "no_go"


def test_a_scripted_manifest_is_not_evidence_about_a_live_model():
    scripted = {**LIVE_MANIFEST, "runtime_kind": "scripted"}
    verdicts = _verdicts(gates.evaluate(GOOD_TELEMETRY, [scripted], SIGNOFF))
    assert verdicts["live_eval_pass_rate"] == "insufficient_data"
    assert verdicts["live_eval_critical_all_pass"] == "insufficient_data"


def test_a_signoff_needs_a_name_and_a_date():
    signoff = {**SIGNOFF, "rollback_rehearsal": True}
    verdicts = _verdicts(gates.evaluate(GOOD_TELEMETRY, [LIVE_MANIFEST], signoff))
    assert verdicts["rollback_rehearsal"] == "insufficient_data"


def test_a_regression_fails_and_carries_the_baseline():
    telemetry = {**GOOD_TELEMETRY, "semantic_contradictions": 1}
    report = gates.evaluate(telemetry, [LIVE_MANIFEST], SIGNOFF)
    row = next(c for c in report["criteria"] if c["key"] == "semantic_contradictions")
    assert row["verdict"] == "fail" and row["baseline"] == 2


def test_the_cli_exits_non_zero_on_no_go(tmp_path):
    telemetry = tmp_path / "t.json"
    telemetry.write_text(json.dumps({}))
    out = tmp_path / "report.json"
    assert gates.main(["--telemetry", str(telemetry), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["decision"] == "no_go"
