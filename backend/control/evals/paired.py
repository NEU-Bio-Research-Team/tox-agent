"""Paired comparison of two eval runs (Wave 4 cutover evidence).

A cutover decision — report_orchestrator_v2 on, router_v2 on, external workers,
trust envelopes — compares a baseline run with a candidate run of the same
suite. Aggregate pass rates hide the question that matters: *which tasks*
changed. Two runs at 80% can share no failures.

This reads two result sets (``results-*.json``) and their manifests, pairs
tasks by id, and reports:

* comparability: same suite hash, fixture mode and trial count; which
  effective-product fields differ (the change under test should be the only
  difference, and anything else is reported as a confound);
* per-task transitions: pass->fail (regressions), fail->pass (fixes), and
  tasks that were not graded in both runs;
* the discordant-pair counts and an exact McNemar p-value, so "better" means
  more than one lucky task;
* critical regressions separately — a single critical pass->fail blocks a
  cutover whatever the totals say;
* first-pass / fallback deltas from the per-trial answer outcomes.

    python -m evals.paired --baseline manifests/manifest-A.json \
        --candidate manifests/manifest-B.json --out paired.json
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "eval-paired-comparison-v1"

#: Effective-product sections that differ between runs by design of a
#: cutover; anything else differing is reported as a confound.
_VOLATILE = {"effective_product_hash", "capabilities"}


def _results_for(manifest_path: Path) -> list[dict[str, Any]]:
    stamp = manifest_path.name.removeprefix("manifest-")
    path = manifest_path.with_name(f"results-{stamp}")
    return json.loads(path.read_text())


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: value}


def product_diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    a = _flatten({k: v for k, v in (baseline or {}).items() if k not in _VOLATILE})
    b = _flatten({k: v for k, v in (candidate or {}).items() if k not in _VOLATILE})
    return {
        key: {"baseline": a.get(key), "candidate": b.get(key)}
        for key in sorted(set(a) | set(b))
        if a.get(key) != b.get(key)
    }


def mcnemar_exact(regressions: int, fixes: int) -> float | None:
    """Two-sided exact binomial test on the discordant pairs."""
    n = regressions + fixes
    if n == 0:
        return None
    k = min(regressions, fixes)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 6)


def _outcome_rates(results: list[dict[str, Any]]) -> dict[str, Any]:
    answered = first = fallback = 0
    for row in results:
        for trial in row.get("trials") or ():
            outcome = trial.get("answer_outcome")
            if outcome in (None, "none"):
                continue
            answered += 1
            first += outcome == "first_pass"
            fallback += outcome == "fallback"
    return {
        "answered_trials": answered,
        "first_pass_rate": round(first / answered, 4) if answered else None,
        "fallback_rate": round(fallback / answered, 4) if answered else None,
    }


def compare(
    baseline_manifest: dict[str, Any], baseline_results: list[dict[str, Any]],
    candidate_manifest: dict[str, Any], candidate_results: list[dict[str, Any]],
    *, expected_change: list[str] | None = None,
) -> dict[str, Any]:
    comparability: list[str] = []
    for key in ("eval_suite_hash", "fixture_mode", "trial_count", "runtime_kind"):
        if baseline_manifest.get(key) != candidate_manifest.get(key):
            comparability.append(
                f"{key} differs: {baseline_manifest.get(key)!r} vs {candidate_manifest.get(key)!r}"
            )
    diff = product_diff(
        baseline_manifest.get("effective_product") or {},
        candidate_manifest.get("effective_product") or {},
    )
    expected = tuple(expected_change or ())
    confounds = sorted(k for k in diff if not any(k.startswith(e) for e in expected))

    base = {r["task_id"]: r for r in baseline_results if r.get("status") != "invalid"}
    cand = {r["task_id"]: r for r in candidate_results if r.get("status") != "invalid"}
    graded = ("pass", "fail")
    transitions: dict[str, list[str]] = {
        "regressions": [], "fixes": [], "still_pass": [], "still_fail": [], "not_paired": [],
    }
    critical_regressions: list[str] = []
    for task_id in sorted(set(base) | set(cand)):
        a, b = base.get(task_id), cand.get(task_id)
        if not a or not b or a.get("status") not in graded or b.get("status") not in graded:
            transitions["not_paired"].append(task_id)
            continue
        before, after = a["status"] == "pass", b["status"] == "pass"
        if before and not after:
            transitions["regressions"].append(task_id)
            if a.get("critical") or b.get("critical"):
                critical_regressions.append(task_id)
        elif after and not before:
            transitions["fixes"].append(task_id)
        elif after:
            transitions["still_pass"].append(task_id)
        else:
            transitions["still_fail"].append(task_id)

    paired = len(transitions["regressions"]) + len(transitions["fixes"]) + len(
        transitions["still_pass"]) + len(transitions["still_fail"])
    blockers = list(comparability)
    if confounds:
        blockers.append(f"effective product differs outside the change under test: {confounds[:10]}")
    if critical_regressions:
        blockers.append(f"critical regressions: {critical_regressions}")
    return {
        "schema_version": SCHEMA_VERSION,
        "comparability": comparability,
        "product_diff": diff,
        "expected_change": list(expected),
        "confounds": confounds,
        "paired_tasks": paired,
        "transitions": {k: v for k, v in transitions.items()},
        "counts": {k: len(v) for k, v in transitions.items()},
        "mcnemar_p": mcnemar_exact(len(transitions["regressions"]), len(transitions["fixes"])),
        "critical_regressions": critical_regressions,
        "answer_outcomes": {
            "baseline": _outcome_rates(baseline_results),
            "candidate": _outcome_rates(candidate_results),
        },
        "cutover_blockers": blockers,
        "cutover_supported": not blockers and len(transitions["fixes"]) >= len(transitions["regressions"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--expect-change", action="append", default=[],
        help="effective-product key prefix the comparison is about, e.g. "
             "flags.report_orchestrator_v2 (repeatable). Other differences are confounds.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = compare(
        json.loads(args.baseline.read_text()), _results_for(args.baseline),
        json.loads(args.candidate.read_text()), _results_for(args.candidate),
        expected_change=args.expect_change,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0 if report["cutover_supported"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
