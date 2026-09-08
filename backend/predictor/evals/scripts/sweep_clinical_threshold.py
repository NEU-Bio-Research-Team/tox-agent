#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, List

import httpx
import numpy as np
import pandas as pd


@dataclass
class SweepRow:
    threshold: float
    recall: float
    precision: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def _predict_scores(model_server_url: str, smiles: List[str], batch_size: int) -> List[float]:
    url = f"{model_server_url.rstrip('/')}/predict/batch"
    client = httpx.Client(timeout=120.0)
    scores: List[float] = []

    try:
        for start in range(0, len(smiles), batch_size):
            chunk = smiles[start : start + batch_size]
            response = client.post(url, json={"smiles_list": chunk, "threshold": 0.35})
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if len(results) != len(chunk):
                raise RuntimeError(
                    f"Unexpected result count for batch {start // batch_size}: expected {len(chunk)}, got {len(results)}"
                )
            for item in results:
                scores.append(float(item.get("p_toxic", 0.0) or 0.0))
    finally:
        client.close()

    return scores


def _iter_thresholds(start: float, stop: float, step: float) -> Iterable[float]:
    value = start
    while value <= stop + 1e-9:
        yield round(value, 6)
        value += step


def _compute_rows(labels: np.ndarray, scores: np.ndarray, thresholds: Iterable[float]) -> List[SweepRow]:
    rows: List[SweepRow] = []

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())

        recall = tp / (tp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        rows.append(
            SweepRow(
                threshold=float(threshold),
                recall=float(recall),
                precision=float(precision),
                f1=float(f1),
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
            )
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep clinical decision threshold without retraining and report recall/precision trade-off."
    )
    parser.add_argument(
        "--input-csv",
        default="test_data/full_test_set.csv",
        help="CSV containing at least columns: smiles, label",
    )
    parser.add_argument(
        "--scores-csv",
        default="",
        help="Optional CSV with p_toxic column. If absent, scores are fetched from model server.",
    )
    parser.add_argument(
        "--model-server-url",
        default="http://127.0.0.1:8000",
        help="Model server URL used when --scores-csv is not provided.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--t-start", type=float, default=0.05)
    parser.add_argument("--t-stop", type=float, default=0.55)
    parser.add_argument("--t-step", type=float, default=0.05)
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional output path for saving sweep metrics.",
    )
    parser.add_argument(
        "--output-summary-json",
        default="",
        help="Optional output path for dashboard summary JSON.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "smiles" not in df.columns or "label" not in df.columns:
        raise ValueError("input-csv must contain columns: smiles, label")

    if args.scores_csv:
        score_df = pd.read_csv(args.scores_csv)
        if "p_toxic" not in score_df.columns:
            raise ValueError("scores-csv must contain p_toxic column")
        if len(score_df) != len(df):
            raise ValueError("scores-csv row count must match input-csv row count")
        scores = score_df["p_toxic"].astype(float).to_numpy()
    else:
        scores = np.array(
            _predict_scores(
                model_server_url=args.model_server_url,
                smiles=df["smiles"].astype(str).tolist(),
                batch_size=max(1, int(args.batch_size)),
            )
        )

    labels = (df["label"].astype(float) >= 0.5).astype(int).to_numpy()
    thresholds = _iter_thresholds(args.t_start, args.t_stop, args.t_step)
    rows = _compute_rows(labels=labels, scores=scores, thresholds=thresholds)

    out = pd.DataFrame(
        [
            {
                "threshold": row.threshold,
                "recall": row.recall,
                "precision": row.precision,
                "f1": row.f1,
                "tp": row.tp,
                "fp": row.fp,
                "fn": row.fn,
                "tn": row.tn,
            }
            for row in rows
        ]
    )

    print(out.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_by_recall = out.sort_values(["recall", "precision", "f1"], ascending=[False, False, False]).iloc[0]
    print("\nBest threshold by recall:")
    print(best_by_recall.to_string())

    if args.output_csv:
        out.to_csv(args.output_csv, index=False)

    if args.output_summary_json:
        row_050 = out.loc[(out["threshold"] - 0.50).abs().idxmin()]
        row_035 = out.loc[(out["threshold"] - 0.35).abs().idxmin()]
        summary = {
            "toxic_recall_t_0_50": float(row_050["recall"]),
            "toxic_precision_t_0_50": float(row_050["precision"]),
            "false_negatives_t_0_50": int(row_050["fn"]),
            "toxic_recall_t_0_35": float(row_035["recall"]),
            "toxic_precision_t_0_35": float(row_035["precision"]),
            "false_negatives_t_0_35": int(row_035["fn"]),
            "best_threshold_by_recall": float(best_by_recall["threshold"]),
            "best_recall": float(best_by_recall["recall"]),
            "best_precision": float(best_by_recall["precision"]),
            "n_total_samples": int(len(labels)),
            "n_positive_samples": int(labels.sum()),
        }
        with open(args.output_summary_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
