"""Judge calibration against a dual-SME adjudicated set (P0-04).

A semantic judge becomes a gate only after its agreement with adjudicated
human verdicts is measured and published, per dimension and broken down by
language and risk tier — an aggregate agreement figure can hide a judge that
is reliable in English and guesses in Vietnamese, or that is lenient exactly
on the critical tier.

Input, one JSON object per line (``sme-adjudicated-v1``)::

    {"item_id": "...", "rubric": "evidence-synthesis@1", "dimension": "claim_support",
     "language": "vi", "risk_tier": "high",
     "sme_a": "pass", "sme_b": "fail", "adjudicated": "fail"}

Judge verdicts, one per line::

    {"item_id": "...", "rubric": "...", "dimension": "...", "verdict": "pass|fail|abstain"}

The report per rubric/dimension: confusion matrix (adjudicated x judge),
agreement and Cohen's kappa over non-abstained items, abstention rate,
false-pass rate (judge pass where the SMEs said fail — the error that lets a
bad answer through), inter-SME kappa as the ceiling, and a ``calibrated``
decision against ``THRESHOLDS``.

    python -m evals.calibration --sme gold.jsonl --judge verdicts.jsonl --out calibration.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "judge-calibration-v1"
LABELS = ("pass", "fail")

#: A rubric is gating only when every blocking dimension clears these.
THRESHOLDS = {
    "min_items": 30,
    "min_kappa": 0.6,
    "max_false_pass_rate": 0.05,
    "max_abstention_rate": 0.2,
}


def _kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    expected = sum(
        (sum(a == label for a, _ in pairs) / n) * (sum(b == label for _, b in pairs) / n)
        for label in LABELS
    )
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def _cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {gold: {j: 0 for j in (*LABELS, "abstain")} for gold in LABELS}
    decided: list[tuple[str, str]] = []
    abstained = false_pass = gold_fail = 0
    for row in rows:
        gold, got = row["adjudicated"], row.get("verdict", "abstain")
        if gold not in LABELS:
            continue
        matrix[gold][got] += 1
        if got == "abstain":
            abstained += 1
        else:
            decided.append((gold, got))
        if gold == "fail":
            gold_fail += 1
            false_pass += got == "pass"
    total = sum(sum(r.values()) for r in matrix.values())
    sme_pairs = [
        (r["sme_a"], r["sme_b"]) for r in rows
        if r.get("sme_a") in LABELS and r.get("sme_b") in LABELS
    ]
    return {
        "items": total,
        "confusion": matrix,
        "agreement": round(sum(a == b for a, b in decided) / len(decided), 4) if decided else None,
        "kappa": _kappa(decided),
        "abstention_rate": round(abstained / total, 4) if total else None,
        "false_pass_rate": round(false_pass / gold_fail, 4) if gold_fail else None,
        "inter_sme_kappa": _kappa(sme_pairs),
    }


def _clears(cell: dict[str, Any]) -> list[str]:
    reasons = []
    if cell["items"] < THRESHOLDS["min_items"]:
        reasons.append(f"{cell['items']} items < {THRESHOLDS['min_items']}")
    if cell["kappa"] is None or cell["kappa"] < THRESHOLDS["min_kappa"]:
        reasons.append(f"kappa {cell['kappa']} < {THRESHOLDS['min_kappa']}")
    if cell["false_pass_rate"] is not None and cell["false_pass_rate"] > THRESHOLDS["max_false_pass_rate"]:
        reasons.append(f"false-pass rate {cell['false_pass_rate']} > {THRESHOLDS['max_false_pass_rate']}")
    if cell["abstention_rate"] is not None and cell["abstention_rate"] > THRESHOLDS["max_abstention_rate"]:
        reasons.append(f"abstention {cell['abstention_rate']} > {THRESHOLDS['max_abstention_rate']}")
    return reasons


def calibrate(gold: Iterable[dict[str, Any]], verdicts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    from evals.graders.semantic import RUBRICS

    by_key = {(v["item_id"], v["rubric"], v["dimension"]): v["verdict"] for v in verdicts}
    joined: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gold:
        key = (row["item_id"], row["rubric"], row["dimension"])
        joined[(row["rubric"], row["dimension"])].append(
            {**row, "verdict": by_key.get(key, "abstain")}
        )

    rubrics: dict[str, Any] = {}
    for rubric_key, rubric in RUBRICS.items():
        dims: dict[str, Any] = {}
        blockers: list[str] = []
        for dimension in rubric.dimensions:
            rows = joined.get((rubric_key, dimension.name), [])
            cell = _cell(rows)
            cell["by_language"] = {
                lang: _cell([r for r in rows if r.get("language") == lang])
                for lang in sorted({r.get("language") for r in rows if r.get("language")})
            }
            cell["by_risk_tier"] = {
                tier: _cell([r for r in rows if r.get("risk_tier") == tier])
                for tier in sorted({r.get("risk_tier") for r in rows if r.get("risk_tier")})
            }
            dims[dimension.name] = cell
            if dimension.blocking:
                blockers.extend(f"{dimension.name}: {reason}" for reason in _clears(cell))
                for tier, tier_cell in cell["by_risk_tier"].items():
                    if tier == "critical" and (tier_cell["false_pass_rate"] or 0) > 0:
                        blockers.append(f"{dimension.name}: a false pass on the critical tier")
        rubrics[rubric_key] = {"dimensions": dims, "calibrated": not blockers, "blockers": blockers}
    return {"schema_version": SCHEMA_VERSION, "thresholds": THRESHOLDS, "rubrics": rubrics}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sme", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = calibrate(_read_jsonl(args.sme), _read_jsonl(args.judge))
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for key, rubric in report["rubrics"].items():
        state = "calibrated" if rubric["calibrated"] else "NOT calibrated"
        print(f"{key}: {state}" + ("" if rubric["calibrated"] else f" — {rubric['blockers'][:3]}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
