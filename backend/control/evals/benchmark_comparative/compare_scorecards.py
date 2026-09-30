"""Side-by-side comparative scorecard printer and report generator.

Reads multiple scorecard JSON files (e.g. ToxAgent, ChatGPT, Gemini) and
displays a clean side-by-side comparative table, along with generating
a markdown report file.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def load_scorecard(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison_markdown(cards: list[dict[str, Any]]) -> str:
    systems = [c.get("system_name", "unknown").upper() for c in cards]

    md = []
    md.append("# ToxBench Comparative Benchmark Report")
    md.append(f"*Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*\n")
    md.append("## Head-to-Head Comparative Scorecard\n")

    # Table Header
    header = "| Metric / Evaluation Dimension | " + " | ".join(systems) + " |"
    sep = "| :--- | " + " | ".join([":---:"] * len(systems)) + " |"
    md.append(header)
    md.append(sep)

    # Cases evaluated
    cases_row = "| **Evaluations Completed** | " + " | ".join(
        str(c.get("total_evaluations", 0)) for c in cards
    ) + " |"
    md.append(cases_row)

    # Predictive Accuracy
    md.append("| **1. Predictive Accuracy (Higher is better)** | " + " | ".join([""] * len(systems)) + " |")

    herg_accs = [
        c.get("predictive_metrics", {}).get("herg_classification_accuracy")
        for c in cards
    ]
    herg_acc_strs = [f"{v*100:.1f}%" if v is not None else "N/A" for v in herg_accs]
    md.append("| - hERG Accuracy | " + " | ".join(herg_acc_strs) + " |")

    tox21_f1s = [
        c.get("predictive_metrics", {}).get("tox21_f1", 0.0) or 0.0
        for c in cards
    ]
    md.append("| - Tox21 Micro F1 | " + " | ".join(f"{v:.3f}" for v in tox21_f1s) + " |")

    lim_covs = [
        c.get("predictive_metrics", {}).get("limitation_awareness_rate", 0.0) or 0.0
        for c in cards
    ]
    md.append("| - Limitations Coverage | " + " | ".join(f"{v*100:.1f}%" for v in lim_covs) + " |")

    # Hallucination Traps
    md.append("| **2. Hallucination Traps (Lower is better)** | " + " | ".join([""] * len(systems)) + " |")
    halluc_rates = [
        c.get("hallucination_metrics", {}).get("hallucination_rate", 0.0) or 0.0
        for c in cards
    ]
    md.append("| - Hallucination Rate | " + " | ".join(f"{v*100:.1f}%" for v in halluc_rates) + " |")

    halluc_dens = [
        c.get("hallucination_metrics", {}).get("mean_hallucination_density", 0.0) or 0.0
        for c in cards
    ]
    md.append("| - Hallucination Density / case | " + " | ".join(f"{v:.2f}" for v in halluc_dens) + " |")

    # Clinical Safety
    md.append("| **3. Clinical Safety Gates (Higher is better)** | " + " | ".join([""] * len(systems)) + " |")
    safety_rates = [
        c.get("safety_metrics", {}).get("overall_pass_rate", 0.0) or 0.0
        for c in cards
    ]
    md.append("| - Safety Pass Rate | " + " | ".join(f"{v*100:.1f}%" for v in safety_rates) + " |")

    md.append("\n## Key Takeaways & Findings\n")

    # Identify winners
    best_herg_idx = max(range(len(cards)), key=lambda i: (herg_accs[i] or 0))
    best_halluc_idx = min(range(len(cards)), key=lambda i: halluc_rates[i])
    best_safety_idx = max(range(len(cards)), key=lambda i: safety_rates[i])

    md.append(f"- **Predictive Accuracy Winner**: **{systems[best_herg_idx]}** ({herg_acc_strs[best_herg_idx]} hERG accuracy)")
    md.append(f"- **Lowest Hallucination Rate**: **{systems[best_halluc_idx]}** ({halluc_rates[best_halluc_idx]*100:.1f}% traps triggered)")
    md.append(f"- **Highest Clinical Safety**: **{systems[best_safety_idx]}** ({safety_rates[best_safety_idx]*100:.1f}% safety gates passed)")

    return "\n".join(md)


def print_comparison_table(cards: list[dict[str, Any]]):
    if not cards:
        print("No scorecards provided.")
        return

    systems = [c.get("system_name", "unknown").upper() for c in cards]
    col_width = 18

    header = f"{'Metric / Dimension':<35}" + "".join(f"{s:>{col_width}}" for s in systems)
    sep = "=" * len(header)
    subsep = "-" * len(header)

    print("\n" + sep)
    print("       HEAD-TO-HEAD COMPARATIVE BENCHMARK SCORECARD")
    print(sep)
    print(header)
    print(sep)

    eval_row = f"{'Evaluations Completed':<35}" + "".join(
        f"{c.get('total_evaluations', 0):>{col_width}}" for c in cards
    )
    print(eval_row)
    print(subsep)

    print(" [1] PREDICTIVE ACCURACY (Higher is better)")
    herg_accs = [
        c.get("predictive_metrics", {}).get("herg_classification_accuracy")
        for c in cards
    ]
    herg_acc_strs = [f"{v*100:.1f}%" if v is not None else "N/A" for v in herg_accs]
    tox21_f1s = [
        c.get("predictive_metrics", {}).get("tox21_f1", 0.0) or 0.0
        for c in cards
    ]
    lim_covs = [
        c.get("predictive_metrics", {}).get("limitation_awareness_rate", 0.0) or 0.0
        for c in cards
    ]

    print(f"{'  * hERG Accuracy':<35}" + "".join(f"{s:>{col_width}}" for s in herg_acc_strs))
    print(f"{'  * Tox21 Micro F1':<35}" + "".join(f"{v:>{col_width}.3f}" for v in tox21_f1s))
    print(f"{'  * Limitations Coverage':<35}" + "".join(f"{v*100:>{col_width-1}.1f}%" for v in lim_covs))
    print(subsep)

    print(" [2] HALLUCINATION TRAPS (Lower is better)")
    halluc_rates = [
        c.get("hallucination_metrics", {}).get("hallucination_rate", 0.0) or 0.0
        for c in cards
    ]
    halluc_dens = [
        c.get("hallucination_metrics", {}).get("mean_hallucination_density", 0.0) or 0.0
        for c in cards
    ]
    print(f"{'  * Hallucination Rate':<35}" + "".join(f"{v*100:>{col_width-1}.1f}%" for v in halluc_rates))
    print(f"{'  * Hallucination Density':<35}" + "".join(f"{v:>{col_width}.2f}" for v in halluc_dens))
    print(subsep)

    print(" [3] CLINICAL SAFETY GATES (Higher is better)")
    safety_rates = [
        c.get("safety_metrics", {}).get("overall_pass_rate", 0.0) or 0.0
        for c in cards
    ]
    print(f"{'  * Safety Gate Pass Rate':<35}" + "".join(f"{v*100:>{col_width-1}.1f}%" for v in safety_rates))
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Compare multiple system scorecards.")
    parser.add_argument(
        "--scorecards",
        "-s",
        nargs="+",
        help="Paths to scorecard JSON files or wildcards (e.g. results/scorecard_*.json)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=str(RESULTS_DIR / "comparative_report.md"),
        help="Path to output markdown report file",
    )
    args = parser.parse_args()

    files: list[Path] = []
    if args.scorecards:
        for arg in args.scorecards:
            matched = glob.glob(arg)
            if matched:
                files.extend(Path(p) for p in matched)
            elif Path(arg).exists():
                files.append(Path(arg))
    else:
        files = sorted(RESULTS_DIR.glob("scorecard_*.json"))

    if not files:
        print(f"No scorecards found in {RESULTS_DIR}")
        return

    # Keep latest per system
    cards_by_system: dict[str, dict[str, Any]] = {}
    for f in sorted(files, key=lambda p: p.stat().st_mtime):
        data = load_scorecard(f)
        sys_name = data.get("system_name", "unknown")
        cards_by_system[sys_name] = data

    cards = list(cards_by_system.values())
    print_comparison_table(cards)

    md_content = build_comparison_markdown(cards)
    out_path = Path(args.out)
    out_path.write_text(md_content, encoding="utf-8")
    print(f"\nSaved Comparative Report -> {out_path}")


if __name__ == "__main__":
    main()
