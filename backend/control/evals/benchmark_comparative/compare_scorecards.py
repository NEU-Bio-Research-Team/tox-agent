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


def _get(card: dict[str, Any], section: str, key: str) -> Any:
    return card.get(section, {}).get(key)


def _pct(v: Any) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _num(v: Any, digits: int = 3) -> str:
    return "N/A" if v is None else f"{v:.{digits}f}"


def _count(v: Any) -> str:
    return "N/A" if v is None else str(v)


#: (group or None, label, section, key, formatter). A group row is a heading.
ROWS: list[tuple[str | None, str, str, str, Any]] = [
    (None, "Evaluations completed", "", "total_evaluations", _count),
    ("1. Predictive accuracy (higher is better)", "", "", "", None),
    (None, "hERG accuracy, committed calls", "predictive_metrics", "herg_classification_accuracy", _pct),
    (None, "hERG coverage (committed / labelled)", "predictive_metrics", "herg_coverage", _pct),
    (None, "hERG strict accuracy (uncertain = miss)", "predictive_metrics", "herg_strict_accuracy", _pct),
    (None, "hERG labelled cases", "predictive_metrics", "herg_labelled_count", _count),
    (None, "Tox21 micro F1", "predictive_metrics", "tox21_f1", _num),
    (None, "Tox21 precision", "predictive_metrics", "tox21_precision", _num),
    (None, "Tox21 recall", "predictive_metrics", "tox21_recall", _num),
    (None, "Tox21 labelled cases scored", "predictive_metrics", "tox21_evaluated_count", _count),
    (None, "Limitations coverage (lexical proxy)", "predictive_metrics", "limitation_awareness_rate", _pct),
    (None, "Cases with expected limitations", "predictive_metrics", "limitation_evaluated_count", _count),
    ("2. Hallucination traps (lower is better)", "", "", "", None),
    (None, "Hallucination rate", "hallucination_metrics", "hallucination_rate", _pct),
    (None, "Hallucination density / case", "hallucination_metrics", "mean_hallucination_density", lambda v: _num(v, 2)),
    ("3. Safety gates (higher is better)", "", "", "", None),
    (None, "Safety gate pass rate", "safety_metrics", "overall_pass_rate", _pct),
    ("4. Faithfulness", "", "", "", None),
    (None, "Mean FActScore_tox (N/A: no claims)", "faithfulness_metrics", "mean_factscore_tox", _num),
]


def _cells(card: dict[str, Any], section: str, key: str) -> Any:
    return card.get(key) if not section else _get(card, section, key)


def _per_code(cards: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    codes = sorted({
        code for c in cards
        for code in (_get(c, "predictive_metrics", "limitation_per_code_rate") or {})
    })
    return [
        (code, [_pct((_get(c, "predictive_metrics", "limitation_per_code_rate") or {}).get(code))
                for c in cards])
        for code in codes
    ]


def build_comparison_markdown(cards: list[dict[str, Any]]) -> str:
    systems = [c.get("system_name", "unknown").upper() for c in cards]

    md = []
    md.append("# ToxBench Comparative Benchmark Report")
    md.append(f"*Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*\n")
    md.append("## Head-to-Head Comparative Scorecard\n")
    md.append("| Metric | " + " | ".join(systems) + " |")
    md.append("| :--- | " + " | ".join([":---:"] * len(systems)) + " |")
    for group, label, section, key, fmt in ROWS:
        if group:
            md.append(f"| **{group}** | " + " | ".join([""] * len(systems)) + " |")
            continue
        md.append(f"| {label} | " + " | ".join(fmt(_cells(c, section, key)) for c in cards) + " |")
        if key == "limitation_awareness_rate":
            for code, values in _per_code(cards):
                md.append(f"| &nbsp;&nbsp;{code} | " + " | ".join(values) + " |")

    md.append("\n## Reading these numbers\n")
    md.append("- hERG accuracy counts only committed blocker/non-blocker calls; "
              "`uncertain` is an abstention, reported through coverage and strict accuracy.")
    md.append("- Tox21 is scored only on cases that carry Tox21 labels; a predicted active "
              "on an unlabelled case is unverifiable, not a false positive.")
    md.append("- Limitations coverage is a lexical proxy (codes or patterns in "
              "`metrics.LIMITATION_PATTERNS`), not a graded judgement.")
    md.append("- Hallucination traps and safety gates are regex checks over the response "
              "text. A system that returns little text trivially passes them.")
    md.append("- None of this is the SME grade; that comes from the lab's blind grading.")
    return "\n".join(md)


def print_comparison_table(cards: list[dict[str, Any]]):
    if not cards:
        print("No scorecards provided.")
        return

    systems = [c.get("system_name", "unknown").upper() for c in cards]
    col_width = 18
    label_width = 42

    header = f"{'Metric':<{label_width}}" + "".join(f"{s:>{col_width}}" for s in systems)
    sep = "=" * len(header)

    print("\n" + sep)
    print("       HEAD-TO-HEAD COMPARATIVE BENCHMARK SCORECARD")
    print(sep)
    print(header)
    print(sep)
    for group, label, section, key, fmt in ROWS:
        if group:
            print("-" * len(header))
            print(f" {group}")
            continue
        print(f"{'  ' + label:<{label_width}}"
              + "".join(f"{fmt(_cells(c, section, key)):>{col_width}}" for c in cards))
        if key == "limitation_awareness_rate":
            for code, values in _per_code(cards):
                print(f"{'    ' + code:<{label_width}}" + "".join(f"{v:>{col_width}}" for v in values))
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

    cards = [cards_by_system[name] for name in sorted(cards_by_system)]
    print_comparison_table(cards)

    md_content = build_comparison_markdown(cards)
    out_path = Path(args.out)
    out_path.write_text(md_content, encoding="utf-8")
    print(f"\nSaved Comparative Report -> {out_path}")


if __name__ == "__main__":
    main()
