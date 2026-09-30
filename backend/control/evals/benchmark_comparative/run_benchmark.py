#!/usr/bin/env python3
"""ToxBench one-shot runner — pretty-prints a scorecard table.

Usage::

    # From backend/control/ — needs a running stack on port 8000
    python -m evals.benchmark_comparative.run_benchmark

    # Custom URL / token / trial count
    python -m evals.benchmark_comparative.run_benchmark \\
        --base-url http://localhost:8000 \\
        --token dev-local \\
        --trials 1 \\
        --systems toxagent

    # Full three-way comparison (needs OPENAI_API_KEY + GOOGLE_API_KEY)
    python -m evals.benchmark_comparative.run_benchmark --systems toxagent,gpt,gemini
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _bar(value: float | None, width: int = 20) -> str:
    if value is None:
        return "  N/A  "
    filled = int(round((value or 0) * width))
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{(value or 0)*100:5.1f}%"
    return f"[{bar}] {pct}"


def _fmt(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "   N/A"
    return f"{value:.{decimals}f}"


def print_scorecard(report: dict) -> None:
    systems = report.get("systems", [])
    dims = report.get("dimensions", {})
    winners = report.get("winners", {})
    ts = report.get("timestamp", "")
    trials = report.get("trials_per_case", "?")

    print()
    print("=" * 72)
    print(f"  ToxBench Comparative Scorecard   [{ts}]   trials/case={trials}")
    print("=" * 72)

    # ── Hallucination ──────────────────────────────────────────────────
    halluc = dims.get("hallucination", {})
    print("\n── 1. Hallucination (lower = better) " + "─" * 35)
    for sys_name in systems:
        h = halluc.get(sys_name, {})
        rate = h.get("hallucination_rate")
        density = h.get("mean_hallucination_density")
        severity = h.get("mean_severity")
        w = "  ← winner" if winners.get("hallucination") == sys_name else ""
        print(f"  {sys_name}")
        print(f"    {'Hallucination rate':<33} {_fmt(rate)}   {_bar(rate)}{w}")
        print(f"    {'Mean density (spans/resp)':<33} {_fmt(density)}")
        print(f"    {'Mean severity (1-3)':<33} {_fmt(severity)}")

    # ── Predictive Accuracy ───────────────────────────────────────────
    pred = dims.get("predictive_accuracy", {})
    print("\n── 2. Predictive Accuracy (higher = better) " + "─" * 28)
    for sys_name in systems:
        p = pred.get(sys_name, {})
        herg_acc = p.get("herg_classification_accuracy")
        f1 = p.get("tox21_f1")
        prec = p.get("tox21_precision")
        rec = p.get("tox21_recall")
        lim = p.get("limitation_awareness_rate")
        abs_rate = p.get("abstention_rate")
        w = "  ← winner" if winners.get("predictive_accuracy") == sys_name else ""
        print(f"  {sys_name}")
        print(f"    {'hERG classification accuracy':<33} {_fmt(herg_acc)}   {_bar(herg_acc)}{w}")
        print(f"    {'Tox21 F1':<33} {_fmt(f1)}   {_bar(f1)}")
        print(f"    {'Tox21 Precision':<33} {_fmt(prec)}")
        print(f"    {'Tox21 Recall':<33} {_fmt(rec)}")
        print(f"    {'Limitation awareness rate':<33} {_fmt(lim)}")
        print(f"    {'Abstention rate':<33} {_fmt(abs_rate)}")

    # ── Faithfulness ──────────────────────────────────────────────────
    faith = dims.get("faithfulness", {})
    print("\n── 3. Faithfulness / FActScore_tox (higher = better) " + "─" * 19)
    for sys_name in systems:
        f = faith.get(sys_name, {})
        score = f.get("mean_factscore_tox")
        total = f.get("total_claims", 0)
        sup = f.get("total_supported", 0)
        w = "  ← winner" if winners.get("faithfulness") == sys_name else ""
        print(f"  {sys_name}")
        print(f"    {'Mean FActScore_tox':<33} {_fmt(score)}   {_bar(score)}{w}")
        print(f"    {'Total claims / supported':<33} {total} / {sup}")

    # ── Safety ────────────────────────────────────────────────────────
    safety = dims.get("safety_compliance", {})
    print("\n── 4. Safety Compliance (higher = better) " + "─" * 30)
    for sys_name in systems:
        s = safety.get(sys_name, {})
        overall = s.get("overall_pass_rate")
        per_gate = s.get("per_gate_pass_rates", {})
        w = "  ← winner" if winners.get("safety_compliance") == sys_name else ""
        print(f"  {sys_name}")
        print(f"    {'Overall pass rate':<33} {_fmt(overall)}   {_bar(overall)}{w}")
        for gate, rate in per_gate.items():
            print(f"    {'  ' + gate:<33} {_fmt(rate)}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n── Summary " + "─" * 61)
    print(f"  {'Dimension':<30}  Winner")
    print("  " + "-" * 45)
    for dim, winner in winners.items():
        print(f"  {dim:<30}  {winner}")
    print()
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="ToxBench Runner — pretty scorecard")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--systems", default="toxagent",
                        help="Comma-separated: toxagent,gpt,gemini")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--dataset", type=Path,
                        default=HERE / "dataset" / "toxbench_dataset.json")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--json-only", action="store_true",
                        help="Skip pretty-print, just save JSON")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        print("Run: python -m evals.benchmark_comparative.build_dataset")
        return 1

    # Inject token into env so driver picks it up
    if args.token:
        os.environ["TOXAGENT_STATIC_TOKENS"] = args.token

    # Import here (after possible env inject)
    from .runner import run_benchmark

    system_names = [s.strip() for s in args.systems.split(",") if s.strip()]
    print(f"\nRunning ToxBench: systems={system_names}  trials={args.trials}")
    print(f"Dataset : {args.dataset}")
    print(f"Backend : {args.base_url}")

    report = run_benchmark(
        dataset_path=args.dataset,
        system_names=system_names,
        trials=args.trials,
        out_dir=args.out,
        base_url=args.base_url,
    )

    if not args.json_only:
        print_scorecard(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
