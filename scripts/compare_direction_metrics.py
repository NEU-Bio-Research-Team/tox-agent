#!/usr/bin/env python3
"""Aggregate experiment metrics for Direction 1, 2, and 3 into a single CSV summary."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def _read_metrics_txt(path: Path) -> Dict[str, float | str]:
    payload: Dict[str, float | str] = {}
    if not path.exists():
        return payload

    pattern = re.compile(r"^([^:]+):\s*(.+)$")
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("="):
                continue
            m = pattern.match(line)
            if not m:
                continue
            key = m.group(1).strip()
            val = m.group(2).strip()
            try:
                payload[key] = float(val)
            except ValueError:
                payload[key] = val
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare metrics across 3 experiment directions")
    parser.add_argument(
        "--direction1-json",
        type=str,
        default="models/tox21_clinical_proxy/clinical_proxy_metrics.json",
    )
    parser.add_argument(
        "--direction2-txt",
        type=str,
        default="models/tox21_gatv2_model/tox21_gatv2_metrics.txt",
    )
    parser.add_argument(
        "--direction3-json",
        type=str,
        default="models/clinical_head_model/clinical_head_metrics.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/direction_comparison.csv",
    )
    args = parser.parse_args()

    d1_path = project_root / args.direction1_json
    d2_path = project_root / args.direction2_txt
    d3_path = project_root / args.direction3_json

    rows = []

    if d1_path.exists():
        with open(d1_path, "r") as f:
            d1 = json.load(f)
        test = ((d1.get("metrics") or {}).get("test") or {})
        rows.append(
            {
                "direction": "direction_1_tox21_proxy",
                "benchmark_domain": "clinical_binary",
                "test_auc_roc": test.get("auc_roc"),
                "test_pr_auc": test.get("pr_auc"),
                "test_f1": test.get("f1"),
                "test_accuracy": test.get("accuracy"),
                "test_sensitivity": test.get("sensitivity"),
                "test_specificity": test.get("specificity"),
                "threshold": ((d1.get("threshold") or {}).get("threshold")),
            }
        )

    if d2_path.exists():
        d2 = _read_metrics_txt(d2_path)
        rows.append(
            {
                "direction": "direction_2_retrain_tox21",
                "benchmark_domain": "tox21_multitask",
                "test_auc_roc": d2.get("test_macro_auc_roc"),
                "test_pr_auc": d2.get("test_macro_pr_auc"),
                "test_f1": d2.get("test_macro_f1"),
                "test_accuracy": None,
                "test_sensitivity": None,
                "test_specificity": None,
                "threshold": None,
            }
        )

    if d3_path.exists():
        with open(d3_path, "r") as f:
            d3 = json.load(f)
        test = ((d3.get("metrics") or {}).get("test") or {})
        rows.append(
            {
                "direction": "direction_3_clinical_head",
                "benchmark_domain": "clinical_binary",
                "test_auc_roc": test.get("auc_roc"),
                "test_pr_auc": test.get("pr_auc"),
                "test_f1": test.get("f1"),
                "test_accuracy": test.get("accuracy"),
                "test_sensitivity": test.get("sensitivity"),
                "test_specificity": test.get("specificity"),
                "threshold": ((d3.get("threshold") or {}).get("threshold")),
            }
        )

    if not rows:
        raise FileNotFoundError(
            "No metrics files found. Run direction scripts before comparison."
        )

    df = pd.DataFrame(rows)
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("=" * 80)
    print("Direction Metrics Summary")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"Saved comparison table to: {output_path}")


if __name__ == "__main__":
    main()
