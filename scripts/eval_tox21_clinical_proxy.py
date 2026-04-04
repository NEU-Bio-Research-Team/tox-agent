#!/usr/bin/env python3
"""Evaluate Tox21-derived clinical proxy on a binary clinical dataset (ClinTox)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.clinical_head import calibrate_threshold_youden, compute_binary_metrics
from backend.data import load_clintox
from backend.inference import (
    DEFAULT_CLINICAL_PROXY_TASK_WEIGHTS,
    clinical_score_from_tox21,
    load_tox21_gatv2_model,
)
from backend.graph_data import smiles_to_pyg_data
from backend.utils import set_seed


def _collate_graph(batch):
    return Batch.from_data_list(batch)


def _predict_tox21_prob_matrix(
    smiles_list,
    model,
    task_names,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Predict per-task probabilities aligned with input order."""
    num_tasks = len(task_names)
    probs = np.full((len(smiles_list), num_tasks), np.nan, dtype=np.float32)

    valid_indices = []
    valid_graphs = []
    for idx, smiles in enumerate(smiles_list):
        try:
            data = smiles_to_pyg_data(smiles, label=None)
        except Exception:
            data = None

        if data is None:
            continue

        valid_indices.append(idx)
        valid_graphs.append(data)

    if not valid_graphs:
        return probs

    loader = DataLoader(
        valid_graphs,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=_collate_graph,
        num_workers=0,
    )

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(batch_probs)

    pred_matrix = np.vstack(all_probs)
    for row_idx, sample_idx in enumerate(valid_indices):
        probs[sample_idx, :] = pred_matrix[row_idx, :]

    return probs


def _proxy_probabilities(
    prob_matrix: np.ndarray,
    task_names,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute proxy p_toxic and coverage per sample from Tox21 probabilities."""
    p_proxy = np.zeros(prob_matrix.shape[0], dtype=np.float32)
    coverage = np.zeros(prob_matrix.shape[0], dtype=np.float32)

    for i in range(prob_matrix.shape[0]):
        scores = {task: float(prob_matrix[i, j]) for j, task in enumerate(task_names)}
        proxy = clinical_score_from_tox21(scores)
        p_proxy[i] = float(proxy.get("score", 0.0))
        coverage[i] = float(proxy.get("coverage", 0.0))

    return p_proxy, coverage


def _evaluate_split(
    df: pd.DataFrame,
    model,
    task_names,
    device: str,
    batch_size: int,
    threshold: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    smiles = df["smiles"].astype(str).tolist()
    labels = df["CT_TOX"].astype(int).to_numpy()

    prob_matrix = _predict_tox21_prob_matrix(
        smiles_list=smiles,
        model=model,
        task_names=task_names,
        device=device,
        batch_size=batch_size,
    )
    p_proxy, coverage = _proxy_probabilities(prob_matrix=prob_matrix, task_names=task_names)

    valid_mask = np.isfinite(labels) & np.isfinite(p_proxy)
    metrics = compute_binary_metrics(
        y_true=labels[valid_mask],
        y_prob=p_proxy[valid_mask],
        threshold=float(threshold),
    )
    metrics["n_parse_error"] = int((~valid_mask).sum())
    metrics["mean_proxy_coverage"] = float(np.nanmean(coverage)) if coverage.size else float("nan")

    pred_df = pd.DataFrame(
        {
            "smiles": smiles,
            "label": labels,
            "p_clinical_proxy": p_proxy,
            "proxy_coverage": coverage,
            "pred_label": (p_proxy >= float(threshold)).astype(int),
            "valid": valid_mask.astype(int),
        }
    )

    return metrics, pred_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate weighted Tox21 clinical proxy against ClinTox labels"
    )
    parser.add_argument("--model-dir", type=str, default="models/tox21_gatv2_model")
    parser.add_argument("--config", type=str, default="config/tox21_gatv2_config.yaml")
    parser.add_argument("--cache-dir", type=str, default="data")
    parser.add_argument("--split-type", type=str, default="scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clinical-threshold", type=float, default=None)
    parser.add_argument("--no-calibrate-threshold", action="store_true")
    parser.add_argument("--enforce-workspace-mode", action="store_true")
    parser.add_argument("--output-dir", type=str, default="models/tox21_clinical_proxy")
    args = parser.parse_args()

    set_seed(int(args.seed))

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; fallback to CPU")
        device = "cpu"

    model_dir = project_root / args.model_dir
    config_path = project_root / args.config
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, task_names = load_tox21_gatv2_model(
        model_dir=model_dir,
        config_path=config_path,
        device=device,
    )

    train_df, val_df, test_df = load_clintox(
        cache_dir=str(project_root / args.cache_dir),
        split_type=str(args.split_type),
        seed=int(args.seed),
        enforce_workspace_mode=bool(args.enforce_workspace_mode),
    )

    # Calibrate threshold on validation split if not provided.
    val_prob_matrix = _predict_tox21_prob_matrix(
        smiles_list=val_df["smiles"].astype(str).tolist(),
        model=model,
        task_names=task_names,
        device=device,
        batch_size=int(args.batch_size),
    )
    val_probs, _ = _proxy_probabilities(val_prob_matrix, task_names=task_names)
    val_labels = val_df["CT_TOX"].astype(int).to_numpy()

    if args.clinical_threshold is not None:
        threshold_info = {
            "threshold": float(args.clinical_threshold),
            "youden_j": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "reason": "provided_by_user",
        }
    elif args.no_calibrate_threshold:
        threshold_info = {
            "threshold": 0.35,
            "youden_j": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "reason": "default_0_35",
        }
    else:
        threshold_info = calibrate_threshold_youden(
            y_true=val_labels,
            y_prob=val_probs,
            default_threshold=0.35,
        )

    threshold = float(threshold_info["threshold"])

    train_metrics, train_pred = _evaluate_split(
        df=train_df,
        model=model,
        task_names=task_names,
        device=device,
        batch_size=int(args.batch_size),
        threshold=threshold,
    )
    val_metrics, val_pred = _evaluate_split(
        df=val_df,
        model=model,
        task_names=task_names,
        device=device,
        batch_size=int(args.batch_size),
        threshold=threshold,
    )
    test_metrics, test_pred = _evaluate_split(
        df=test_df,
        model=model,
        task_names=task_names,
        device=device,
        batch_size=int(args.batch_size),
        threshold=threshold,
    )

    payload = {
        "direction": "direction_1_tox21_clinical_proxy",
        "weights": DEFAULT_CLINICAL_PROXY_TASK_WEIGHTS,
        "threshold": threshold_info,
        "config": {
            "model_dir": str(model_dir),
            "config_path": str(config_path),
            "split_type": str(args.split_type),
            "seed": int(args.seed),
            "device": device,
            "batch_size": int(args.batch_size),
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
    }

    metrics_json = output_dir / "clinical_proxy_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(payload, f, indent=2)

    train_pred.to_csv(output_dir / "clinical_proxy_train_predictions.csv", index=False)
    val_pred.to_csv(output_dir / "clinical_proxy_val_predictions.csv", index=False)
    test_pred.to_csv(output_dir / "clinical_proxy_test_predictions.csv", index=False)

    print("=" * 80)
    print("Direction 1 - Tox21 Clinical Proxy Evaluation")
    print("=" * 80)
    print(f"Threshold used: {threshold:.4f} ({threshold_info.get('reason')})")
    print(f"Train AUC/F1: {train_metrics['auc_roc']:.4f} / {train_metrics['f1']:.4f}")
    print(f"Val   AUC/F1: {val_metrics['auc_roc']:.4f} / {val_metrics['f1']:.4f}")
    print(f"Test  AUC/F1: {test_metrics['auc_roc']:.4f} / {test_metrics['f1']:.4f}")
    print(f"Saved metrics to: {metrics_json}")


if __name__ == "__main__":
    main()
