"""G5: is the agentic flow ready for production? Answered from data (PR-18).

The remediation plan's last gate is a list of numbers and sign-offs. What it
must never be is a judgement made by reading a dashboard and feeling good. This
module takes three inputs a release manager can attach to the decision —

* an **alpha telemetry summary** exported from the metrics store for the alpha
  window (percentiles, rates, counts);
* one or more **eval manifests** written by ``evals.runner``;
* a **sign-off record** naming who signed what, and when —

and returns a verdict per criterion: ``pass``, ``fail`` or
``insufficient_data``. A missing number is never a pass. The overall decision
is ``go`` only when every criterion passes.

Thresholds are the plan's §11/§12 targets. They are data here, not code, so a
product-approved revision (the plan allows one, with a measured distribution)
is a reviewed change to ``CRITERIA`` rather than an edit somebody makes to get
a green result.

    python -m evals.gates --telemetry alpha.json --manifest manifests/manifest-X.json \
        --signoff signoff.json --out g5-report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "toxagent-g5-report-v1"

MIN_ALPHA_DAYS = 7

#: The audit's measured baseline, for the before/after column (§12).
BASELINE: Mapping[str, float] = {
    "qa_latency_p50_s": 46.6,
    "report_latency_p50_s": 155.5,
    "report_cumulative_input_tokens_p50": 28_408,
    "first_pass_acceptance": 0.0,
    "duplicate_normalized_usage": 3.0,
    "semantic_contradictions": 2,
    "promoted_evidence_precision": 0.0,
    "happy_path_report_tool_failures": 3,
}


@dataclass(frozen=True)
class Criterion:
    key: str
    description: str
    source: str  # telemetry | eval | signoff
    check: Callable[[Any], bool]
    target: str


def _at_most(limit: float) -> Callable[[Any], bool]:
    return lambda value: float(value) <= limit


def _at_least(limit: float) -> Callable[[Any], bool]:
    return lambda value: float(value) >= limit


CRITERIA: tuple[Criterion, ...] = (
    Criterion("alpha_window_days", "internal alpha telemetry window", "telemetry",
              _at_least(MIN_ALPHA_DAYS), f">= {MIN_ALPHA_DAYS} days"),
    Criterion("qa_latency_p50_s", "Q&A end-to-end p50", "telemetry", _at_most(15), "<= 15 s"),
    Criterion("qa_latency_p95_s", "Q&A end-to-end p95", "telemetry", _at_most(30), "<= 30 s"),
    Criterion("report_latency_p50_s", "minimal report p50", "telemetry", _at_most(60), "<= 60 s"),
    Criterion("report_latency_p95_s", "minimal report p95", "telemetry", _at_most(120), "<= 120 s"),
    Criterion("report_cumulative_input_tokens_p50", "minimal report cumulative input",
              "telemetry", _at_most(8000), "<= 8000 tokens"),
    Criterion("first_pass_acceptance", "first-pass QA validation", "telemetry",
              _at_least(0.95), ">= 0.95"),
    Criterion("duplicate_normalized_usage", "duplicate normalized usage rows", "telemetry",
              _at_most(0), "== 0"),
    Criterion("usage_unknown_fraction", "usage summaries with unknown completeness",
              "telemetry", _at_most(0.05), "<= 0.05"),
    Criterion("semantic_contradictions", "semantic contradiction hard-gate failures published",
              "telemetry", _at_most(0), "== 0"),
    Criterion("promoted_evidence_precision", "promoted evidence precision (curated set)",
              "telemetry", _at_least(0.9), ">= 0.9"),
    Criterion("happy_path_report_tool_failures", "happy-path report tool failures",
              "telemetry", _at_most(0), "== 0"),
    Criterion("cancel_settlement_p95_s", "cancel settlement p95 from request", "telemetry",
              _at_most(5), "<= 5 s"),
    Criterion("live_eval_critical_all_pass", "full live eval: every critical task passes",
              "eval", lambda value: value is True, "true"),
    Criterion("live_eval_pass_rate", "full live eval pass rate", "eval",
              _at_least(0.9), ">= 0.9"),
    Criterion("sme_semantic_fixtures", "SME sign-off on semantic fixtures", "signoff",
              lambda value: bool(value), "signed"),
    Criterion("rollback_rehearsal", "rollback rehearsal performed", "signoff",
              lambda value: bool(value), "signed"),
    Criterion("runbooks", "runbooks reviewed", "signoff", lambda value: bool(value), "signed"),
    Criterion("fault_drills", "fault drills recorded (docs/runbooks/worker-drills.md)", "signoff",
              lambda value: bool(value), "signed"),
)


def _window_days(telemetry: Mapping[str, Any]) -> float | None:
    start, end = telemetry.get("window_start"), telemetry.get("window_end")
    if not start or not end:
        return None
    try:
        delta = datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))
    except ValueError:
        return None
    return delta.total_seconds() / 86_400


def _eval_values(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The live eval figures, from the newest manifest that ran a live runtime.

    A scripted manifest is not evidence about a live model and is ignored here;
    CI already gates on it.
    """
    live = [m for m in manifests if m.get("runtime_kind") not in (None, "scripted")]
    if not live:
        return {}
    newest = max(live, key=lambda m: str(m.get("generated_at") or ""))
    summary = newest.get("summary") or {}
    values: dict[str, Any] = {}
    if summary.get("executed"):
        values["live_eval_pass_rate"] = summary.get("pass_rate")
    if summary.get("critical_executed"):
        values["live_eval_critical_all_pass"] = summary.get("critical_all_pass")
    values["_manifest"] = {
        key: newest.get(key)
        for key in ("generated_at", "eval_suite_hash", "runtime_kind", "runtime_version",
                    "toxagent_commit", "toxpred_commit", "trial_count")
    }
    return values


def evaluate(
    telemetry: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]] = (),
    signoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    signoff = signoff or {}
    evals = _eval_values(manifests)
    values: dict[str, Any] = {**telemetry, **{k: v for k, v in evals.items() if not k.startswith("_")}}
    days = _window_days(telemetry)
    if days is not None:
        values["alpha_window_days"] = round(days, 3)
    for criterion in CRITERIA:
        if criterion.source == "signoff":
            entry = signoff.get(criterion.key)
            # A sign-off is a name and a date, not a boolean somebody typed.
            values[criterion.key] = bool(
                isinstance(entry, Mapping) and entry.get("by") and entry.get("at")
            ) or None

    results = []
    for criterion in CRITERIA:
        value = values.get(criterion.key)
        if value is None:
            verdict = "insufficient_data"
        else:
            try:
                verdict = "pass" if criterion.check(value) else "fail"
            except (TypeError, ValueError):
                verdict = "insufficient_data"
        entry = {
            "key": criterion.key,
            "description": criterion.description,
            "source": criterion.source,
            "target": criterion.target,
            "value": value,
            "verdict": verdict,
        }
        if criterion.key in BASELINE:
            entry["baseline"] = BASELINE[criterion.key]
        results.append(entry)

    counts = {v: sum(1 for r in results if r["verdict"] == v) for v in ("pass", "fail", "insufficient_data")}
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "go" if counts["pass"] == len(results) else "no_go",
        "counts": counts,
        "criteria": results,
        "telemetry_manifest": {
            key: telemetry.get(key)
            for key in ("window_start", "window_end", "model_id", "runtime_version",
                        "profile_hash", "hardware")
        },
        "eval_manifest": evals.get("_manifest"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    report = evaluate(
        json.loads(args.telemetry.read_text()),
        [json.loads(path.read_text()) for path in args.manifest],
        json.loads(args.signoff.read_text()) if args.signoff else None,
    )
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if report["decision"] == "go" else 1


if __name__ == "__main__":
    sys.exit(main())
