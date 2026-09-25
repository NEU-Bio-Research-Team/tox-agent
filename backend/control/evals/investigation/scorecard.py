"""Turn the lab's grading sheets into a scorecard. Never a single score.

    python -m evals.investigation.scorecard --study pilot-2026-09 --packet-id lab-1 \\
        --grades grader_a.csv grader_b.csv \\
        --compare D_toxagent_investigator:C_toxagent_current

Per system and dimension: the mean over cases (graders and trials averaged
within a case first, because the case is the unit — RETHINK §5.3 — and trials
of one case are not independent samples), with a bootstrap interval that
resamples cases. Critical errors: the share of responses at least one grader
flagged. Paired comparisons use only cases both systems have. Inter-rater
agreement is reported when two or more graders graded the same cells.
Ungraded cells are counted, never imputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.investigation.packet import RUBRIC_PATH
from evals.investigation.run import DEFAULT_ROOT

SCHEMA_VERSION = "investigation-scorecard-v1"


class GradeError(ValueError):
    pass


def read_grades(paths: list[Path], rubric: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = {d["id"]: d for d in rubric["dimensions"]}
    criticals = [c["id"] for c in rubric["critical_errors"]]
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for number, row in enumerate(csv.DictReader(handle), 2):
                where = f"{path.name}:{number}"
                missing = [c for c in ("grader_id", "case_id", "response_id", *dimensions, *criticals)
                           if c not in row]
                if missing:
                    raise GradeError(f"{where}: missing columns {missing}")
                grade: dict[str, Any] = {
                    "grader_id": (row["grader_id"] or "").strip() or f"{path.stem}",
                    "case_id": row["case_id"], "response_id": row["response_id"],
                    "scores": {}, "critical": {}, "comments": row.get("comments", ""),
                }
                for dim_id, dim in dimensions.items():
                    raw = (row[dim_id] or "").strip().upper()
                    if raw == "":
                        continue
                    if raw == "NA":
                        if not dim["na_allowed"]:
                            raise GradeError(f"{where}: {dim_id} does not allow NA")
                        grade["scores"][dim_id] = None
                        continue
                    if raw not in {"0", "1", "2", "3"}:
                        raise GradeError(f"{where}: {dim_id} must be 0-3 or NA, got {raw!r}")
                    grade["scores"][dim_id] = int(raw)
                for crit in criticals:
                    raw = (row[crit] or "").strip()
                    if raw == "":
                        continue
                    if raw not in {"0", "1"}:
                        raise GradeError(f"{where}: {crit} must be 0 or 1, got {raw!r}")
                    grade["critical"][crit] = raw == "1"
                rows.append(grade)
    return rows


def _bootstrap(values_by_case: dict[str, float], *, reps: int, seed: int) -> tuple[float, float] | None:
    cases = sorted(values_by_case)
    if len(cases) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        sample = [values_by_case[rng.choice(cases)] for _ in cases]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def _weighted_kappa(pairs: list[tuple[int, int]], categories: int = 4) -> float | None:
    """Quadratic-weighted Cohen's kappa for two graders on a 0..3 scale."""
    if len(pairs) < 2:
        return None
    n = len(pairs)
    observed = [[0.0] * categories for _ in range(categories)]
    for a, b in pairs:
        observed[a][b] += 1
    row = [sum(observed[i]) for i in range(categories)]
    col = [sum(observed[i][j] for i in range(categories)) for j in range(categories)]
    weight = lambda i, j: ((i - j) ** 2) / ((categories - 1) ** 2)  # noqa: E731
    num = sum(weight(i, j) * observed[i][j] for i in range(categories) for j in range(categories))
    den = sum(weight(i, j) * row[i] * col[j] / n for i in range(categories) for j in range(categories))
    return None if den == 0 else round(1 - num / den, 4)


def build_scorecard(*, key: dict[str, Any], grades: list[dict[str, Any]], rubric: dict[str, Any],
                    compare: list[tuple[str, str]] = (), reps: int = 2000,
                    seed: int = 20260925) -> dict[str, Any]:
    identity = {(r["case_id"], r["response_id"]): r for r in key["responses"] if r.get("response_id")}
    unknown = sorted({(g["case_id"], g["response_id"]) for g in grades} - set(identity))
    if unknown:
        raise GradeError(f"grades name responses the key does not have: {unknown[:5]}")
    dimensions = [d["id"] for d in rubric["dimensions"]]
    criticals = [c["id"] for c in rubric["critical_errors"]]

    # system -> dimension -> case -> list of scores (graders x trials)
    cells: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    na: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    flagged: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    responses_by_system: dict[str, set] = defaultdict(set)
    for grade in grades:
        who = identity[(grade["case_id"], grade["response_id"])]
        system = who["system_id"]
        response = (grade["case_id"], grade["response_id"])
        responses_by_system[system].add(response)
        for dim, value in grade["scores"].items():
            if value is None:
                na[system][dim] += 1
            else:
                cells[system][dim][grade["case_id"]].append(value)
        for crit, present in grade["critical"].items():
            if present:
                flagged[system][crit].add(response)

    included = defaultdict(set)
    for r in key["responses"]:
        if r.get("response_id"):
            included[r["system_id"]].add((r["case_id"], r["response_id"]))
    missing = defaultdict(int)
    for r in key["responses"]:
        if not r.get("response_id"):
            missing[r["system_id"]] += 1

    systems: dict[str, Any] = {}
    for system in sorted(set(included) | set(missing)):
        dims: dict[str, Any] = {}
        for dim in dimensions:
            by_case = {case: statistics.fmean(v) for case, v in cells[system][dim].items() if v}
            dims[dim] = {
                "cases_graded": len(by_case),
                "na": na[system][dim],
                "mean": round(statistics.fmean(by_case.values()), 4) if by_case else None,
                "ci95_case_bootstrap": _bootstrap(by_case, reps=reps, seed=seed),
            }
        graded = responses_by_system[system]
        crit_rates = {
            crit: {"responses_flagged": len(flagged[system][crit]), "responses_graded": len(graded),
                   "rate": round(len(flagged[system][crit]) / len(graded), 4) if graded else None}
            for crit in criticals
        }
        any_flag = set().union(*flagged[system].values()) if flagged[system] else set()
        systems[system] = {
            "responses_in_packet": len(included[system]), "responses_missing": missing[system],
            "responses_graded": len(graded), "dimensions": dims, "critical_errors": crit_rates,
            "any_critical_error": {"responses": len(any_flag),
                                   "rate": round(len(any_flag) / len(graded), 4) if graded else None},
        }

    comparisons = []
    for left, right in compare:
        entry: dict[str, Any] = {"system": left, "baseline": right, "dimensions": {}}
        for dim in dimensions:
            a = {c: statistics.fmean(v) for c, v in cells[left][dim].items() if v}
            b = {c: statistics.fmean(v) for c, v in cells[right][dim].items() if v}
            shared = sorted(set(a) & set(b))
            diffs = {c: a[c] - b[c] for c in shared}
            entry["dimensions"][dim] = {
                "shared_cases": len(shared),
                "mean_difference": round(statistics.fmean(diffs.values()), 4) if diffs else None,
                "ci95_case_bootstrap": _bootstrap(diffs, reps=reps, seed=seed),
            }
        comparisons.append(entry)

    agreement: dict[str, Any] = {}
    by_cell: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for grade in grades:
        for dim, value in grade["scores"].items():
            if value is not None:
                by_cell[(grade["case_id"], grade["response_id"], dim)][grade["grader_id"]] = value
    graders = sorted({g["grader_id"] for g in grades})
    if len(graders) >= 2:
        first, second = graders[0], graders[1]
        for dim in dimensions:
            pairs = [(v[first], v[second]) for (c, r, d), v in by_cell.items()
                     if d == dim and first in v and second in v]
            agreement[dim] = {
                "graders": [first, second], "cells": len(pairs),
                "exact_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
                "quadratic_weighted_kappa": _weighted_kappa(pairs),
            }

    return {
        "schema_version": SCHEMA_VERSION, "study_id": key.get("study_id"),
        "packet_id": key.get("packet_id"), "rubric_version": rubric["schema_version"],
        "graders": graders, "unit_of_analysis": "case (graders and trials averaged within a case)",
        "interval": f"95% percentile bootstrap over cases, {reps} resamples, seed {seed}",
        "systems": systems, "comparisons": comparisons, "inter_rater": agreement,
        "note": "Dimensions are separate; there is no total score.",
    }


def render_markdown(card: dict[str, Any]) -> str:
    lines = [f"# Scorecard — {card['study_id']} / {card['packet_id']}", "",
             f"Graders: {', '.join(card['graders']) or 'none'}. Unit: {card['unit_of_analysis']}. "
             f"Interval: {card['interval']}. {card['note']}", ""]
    systems = card["systems"]
    dims = next(iter(systems.values()))["dimensions"].keys() if systems else []
    lines.append("| System | Graded / in packet / missing | " + " | ".join(dims) + " | Any critical error |")
    lines.append("|---|---|" + "---|" * len(dims) + "---|")
    for name, s in systems.items():
        cells = []
        for dim in dims:
            d = s["dimensions"][dim]
            ci = d["ci95_case_bootstrap"]
            cells.append("—" if d["mean"] is None else
                         f"{d['mean']:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
                         + f" (n={d['cases_graded']})")
        crit = s["any_critical_error"]
        lines.append(f"| `{name}` | {s['responses_graded']} / {s['responses_in_packet']} / "
                     f"{s['responses_missing']} | " + " | ".join(cells) + " | "
                     + ("—" if crit["rate"] is None else f"{crit['responses']} ({crit['rate']:.0%})") + " |")
    for comp in card["comparisons"]:
        lines += ["", f"## `{comp['system']}` minus `{comp['baseline']}` (shared cases)", "",
                  "| Dimension | Shared cases | Mean difference [95% CI] |", "|---|---|---|"]
        for dim, d in comp["dimensions"].items():
            ci = d["ci95_case_bootstrap"]
            lines.append(f"| {dim} | {d['shared_cases']} | "
                         + ("—" if d["mean_difference"] is None else f"{d['mean_difference']:+.2f}"
                            + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "")) + " |")
    if card["inter_rater"]:
        lines += ["", "## Inter-rater agreement", "", "| Dimension | Cells | Exact | Weighted κ |",
                  "|---|---|---|---|"]
        for dim, a in card["inter_rater"].items():
            lines.append(f"| {dim} | {a['cells']} | {a['exact_agreement']} | {a['quadratic_weighted_kappa']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--grades", type=Path, nargs="+", required=True)
    parser.add_argument("--compare", action="append", default=[], metavar="SYSTEM:BASELINE")
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args(argv)
    study_dir = args.root / args.study
    key = json.loads((study_dir / "keys" / f"{args.packet_id}.unblinding.json").read_text())
    rubric = json.loads(RUBRIC_PATH.read_text())
    card = build_scorecard(key=key, grades=read_grades(args.grades, rubric), rubric=rubric,
                           compare=[tuple(c.split(":", 1)) for c in args.compare], reps=args.reps)
    out = study_dir / "scorecards" / args.packet_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    (out / "scorecard.md").write_text(render_markdown(card))
    print(render_markdown(card))
    return 0


if __name__ == "__main__":
    sys.exit(main())
