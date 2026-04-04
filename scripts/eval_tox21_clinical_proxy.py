#!/usr/bin/env python3
"""Evaluate Tox21-derived clinical proxy on ClinTox with robust calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.clinical_head import (
    calibrate_threshold_youden,
    calibrate_threshold_youden_cv,
    compute_binary_metrics,
)
from backend.data import load_clintox
from backend.graph_data import smiles_to_pyg_data
from backend.inference import (
    DEFAULT_CLINICAL_PROXY_TASK_WEIGHTS,
    clinical_score_from_tox21,
    load_tox21_gatv2_model,
)
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


def _load_task_weights(raw: Optional[str]) -> Optional[Dict[str, float]]:
    if raw is None:
        return None

    payload = None
    candidate = Path(raw)
    if candidate.exists() and candidate.is_file():
        with open(candidate, "r") as f:
            payload = json.load(f)
    else:
        payload = json.loads(raw)

    if not isinstance(payload, dict):
        raise ValueError("Task weights must be a JSON object mapping task name to weight.")

    out: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _impute_prob_matrix(prob_matrix: np.ndarray, fill_value: float) -> np.ndarray:
    matrix = np.where(np.isfinite(prob_matrix), prob_matrix, float(fill_value)).astype(np.float32)
    return np.clip(matrix, 0.0, 1.0)


def _coverage_from_prob_matrix(prob_matrix: np.ndarray) -> np.ndarray:
    if prob_matrix.size == 0:
        return np.array([], dtype=np.float32)
    return np.mean(np.isfinite(prob_matrix), axis=1).astype(np.float32)


def _weighted_proxy_probabilities(
    prob_matrix: np.ndarray,
    task_names,
    task_weights: Optional[Dict[str, float]],
    renormalize_missing: bool,
    missing_task_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    p_proxy = np.zeros(prob_matrix.shape[0], dtype=np.float32)
    coverage = np.zeros(prob_matrix.shape[0], dtype=np.float32)

    for i in range(prob_matrix.shape[0]):
        scores = {task: float(prob_matrix[i, j]) for j, task in enumerate(task_names)}
        proxy = clinical_score_from_tox21(
            task_scores=scores,
            task_weights=task_weights,
            renormalize_missing=bool(renormalize_missing),
            missing_task_value=float(missing_task_value),
        )
        p_proxy[i] = float(proxy.get("score", 0.0))
        coverage[i] = float(proxy.get("coverage", 0.0))

    return p_proxy, coverage


def _fit_learned_proxy(
    train_prob_matrix: np.ndarray,
    train_labels: np.ndarray,
    task_names,
    impute_value: float,
    cv_folds: int,
    seed: int,
):
    x_train = _impute_prob_matrix(train_prob_matrix, fill_value=float(impute_value))
    y_train = np.asarray(train_labels).reshape(-1).astype(int)

    if x_train.shape[0] != y_train.shape[0]:
        raise RuntimeError("Learned proxy fit failed: x/y shape mismatch")

    valid = np.isfinite(y_train)
    x_train = x_train[valid]
    y_train = y_train[valid]

    labels, counts = np.unique(y_train, return_counts=True)
    if labels.size < 2:
        raise RuntimeError("Learned proxy fit failed: need both positive and negative labels")

    max_cv = int(np.min(counts))
    effective_cv = min(int(cv_folds), max_cv)

    if effective_cv >= 2:
        model = LogisticRegressionCV(
            cv=int(effective_cv),
            class_weight="balanced",
            max_iter=2000,
            scoring="roc_auc",
            solver="liblinear",
            random_state=int(seed),
        )
        reason = "logistic_regression_cv"
    else:
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=int(seed),
        )
        reason = "logistic_regression_fallback_no_cv"

    model.fit(x_train, y_train)

    coef = model.coef_.reshape(-1)
    coef_map = {
        str(task): float(coef[idx])
        for idx, task in enumerate(task_names)
    }

    payload = {
        "fit_reason": reason,
        "cv_folds_effective": int(effective_cv) if effective_cv >= 2 else 0,
        "intercept": float(model.intercept_.reshape(-1)[0]),
        "coefficients": coef_map,
    }
    return model, payload


def _learned_proxy_probabilities(
    prob_matrix: np.ndarray,
    model,
    impute_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = _impute_prob_matrix(prob_matrix, fill_value=float(impute_value))
    p_proxy = model.predict_proba(x)[:, 1].astype(np.float32)
    coverage = _coverage_from_prob_matrix(prob_matrix)
    return p_proxy, coverage


def _coverage_summary(coverage: np.ndarray) -> Dict[str, float]:
    if coverage.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
            "p_low_coverage_lt_0_8": float("nan"),
            "p_low_coverage_lt_0_5": float("nan"),
            "p_zero_coverage": float("nan"),
        }

    cov = np.asarray(coverage).reshape(-1).astype(np.float32)
    return {
        "mean": float(np.nanmean(cov)),
        "std": float(np.nanstd(cov)),
        "min": float(np.nanmin(cov)),
        "p25": float(np.nanpercentile(cov, 25)),
        "p50": float(np.nanpercentile(cov, 50)),
        "p75": float(np.nanpercentile(cov, 75)),
        "max": float(np.nanmax(cov)),
        "p_low_coverage_lt_0_8": float(np.nanmean(cov < 0.8)),
        "p_low_coverage_lt_0_5": float(np.nanmean(cov < 0.5)),
        "p_zero_coverage": float(np.nanmean(cov <= 0.0)),
    }


def _evaluate_split(
    smiles: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    coverage: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, float]]:
    valid_mask = np.isfinite(labels) & np.isfinite(probs)
    metrics = compute_binary_metrics(
        y_true=labels[valid_mask],
        y_prob=probs[valid_mask],
        threshold=float(threshold),
    )
    metrics["n_parse_error"] = int(np.sum(coverage <= 0.0))
    metrics["n_invalid_prob"] = int(np.sum(~valid_mask))
    metrics["mean_proxy_coverage"] = float(np.nanmean(coverage)) if coverage.size else float("nan")

    pred_df = pd.DataFrame(
        {
            "smiles": smiles,
            "label": labels,
            "p_clinical_proxy": probs,
            "proxy_coverage": coverage,
            "pred_label": (probs >= float(threshold)).astype(int),
            "valid": valid_mask.astype(int),
        }
    )

    coverage_stats = _coverage_summary(coverage)
    return metrics, pred_df, coverage_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Tox21-derived clinical proxy against ClinTox labels"
    )
    parser.add_argument("--model-dir", type=str, default="models/tox21_gatv2_model")
    parser.add_argument("--config", type=str, default="config/tox21_gatv2_config.yaml")
    parser.add_argument("--cache-dir", type=str, default="data")
    parser.add_argument("--split-type", type=str, default="scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--proxy-mode",
        choices=["weighted", "learned_lr_cv"],
        default="learned_lr_cv",
        help="weighted: domain-weighted proxy, learned_lr_cv: learned linear proxy",
    )
    parser.add_argument("--task-weights", type=str, default=None)
    parser.add_argument("--renormalize-missing", action="store_true")
    parser.add_argument("--missing-task-value", type=float, default=0.5)
    parser.add_argument("--learned-impute-value", type=float, default=0.5)
    parser.add_argument("--lr-cv-folds", type=int, default=5)
    parser.add_argument("--clinical-threshold", type=float, default=None)
    parser.add_argument("--no-calibrate-threshold", action="store_true")
    parser.add_argument(
        "--threshold-calibration",
        choices=["val", "cv"],
        default="cv",
        help="Threshold calibration strategy when threshold is not provided.",
    )
    parser.add_argument("--threshold-cv-folds", type=int, default=5)
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

    train_smiles = train_df["smiles"].astype(str).to_numpy()
    val_smiles = val_df["smiles"].astype(str).to_numpy()
    test_smiles = test_df["smiles"].astype(str).to_numpy()

    train_labels = train_df["CT_TOX"].astype(int).to_numpy()
    val_labels = val_df["CT_TOX"].astype(int).to_numpy()
    test_labels = test_df["CT_TOX"].astype(int).to_numpy()

    print("Predicting frozen Tox21 probabilities for ClinTox splits...")
    train_prob_matrix = _predict_tox21_prob_matrix(
        smiles_list=train_smiles.tolist(),
        model=model,
        task_names=task_names,
        device=device,
        batch_size=int(args.batch_size),
    )
    val_prob_matrix = _predict_tox21_prob_matrix(
        smiles_list=val_smiles.tolist(),
        model=model,
        task_names=task_names,
        device=device,
        batch_size=int(args.batch_size),
    )
    test_prob_matrix = _predict_tox21_prob_matrix(
        smiles_list=test_smiles.tolist(),
        model=model,
        task_names=task_names,
        device=device,
        batch_size=int(args.batch_size),
    )

    proxy_config: Dict[str, object]
    if str(args.proxy_mode) == "weighted":
        task_weights = _load_task_weights(args.task_weights)
        if task_weights is None:
            task_weights = dict(DEFAULT_CLINICAL_PROXY_TASK_WEIGHTS)

        train_probs, train_cov = _weighted_proxy_probabilities(
            train_prob_matrix,
            task_names=task_names,
            task_weights=task_weights,
            renormalize_missing=bool(args.renormalize_missing),
            missing_task_value=float(args.missing_task_value),
        )
        val_probs, val_cov = _weighted_proxy_probabilities(
            val_prob_matrix,
            task_names=task_names,
            task_weights=task_weights,
            renormalize_missing=bool(args.renormalize_missing),
            missing_task_value=float(args.missing_task_value),
        )
        test_probs, test_cov = _weighted_proxy_probabilities(
            test_prob_matrix,
            task_names=task_names,
            task_weights=task_weights,
            renormalize_missing=bool(args.renormalize_missing),
            missing_task_value=float(args.missing_task_value),
        )
        proxy_config = {
            "proxy_mode": "weighted",
            "task_weights": task_weights,
            "renormalize_missing": bool(args.renormalize_missing),
            "missing_task_value": float(args.missing_task_value),
        }
    else:
        learned_model, learned_info = _fit_learned_proxy(
            train_prob_matrix=train_prob_matrix,
            train_labels=train_labels,
            task_names=task_names,
            impute_value=float(args.learned_impute_value),
            cv_folds=int(args.lr_cv_folds),
            seed=int(args.seed),
        )

        train_probs, train_cov = _learned_proxy_probabilities(
            train_prob_matrix,
            model=learned_model,
            impute_value=float(args.learned_impute_value),
        )
        val_probs, val_cov = _learned_proxy_probabilities(
            val_prob_matrix,
            model=learned_model,
            impute_value=float(args.learned_impute_value),
        )
        test_probs, test_cov = _learned_proxy_probabilities(
            test_prob_matrix,
            model=learned_model,
            impute_value=float(args.learned_impute_value),
        )
        proxy_config = {
            "proxy_mode": "learned_lr_cv",
            "learned_impute_value": float(args.learned_impute_value),
            "lr_cv_folds": int(args.lr_cv_folds),
            "learned_model": learned_info,
        }

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
    elif str(args.threshold_calibration) == "cv":
        threshold_info = calibrate_threshold_youden_cv(
            y_true=val_labels,
            y_prob=val_probs,
            n_splits=int(args.threshold_cv_folds),
            seed=int(args.seed),
            default_threshold=0.35,
        )
    else:
        threshold_info = calibrate_threshold_youden(
            y_true=val_labels,
            y_prob=val_probs,
            default_threshold=0.35,
        )

    threshold = float(threshold_info["threshold"])

    train_metrics, train_pred, train_cov_stats = _evaluate_split(
        smiles=train_smiles,
        labels=train_labels,
        probs=train_probs,
        coverage=train_cov,
        threshold=threshold,
    )
    val_metrics, val_pred, val_cov_stats = _evaluate_split(
        smiles=val_smiles,
        labels=val_labels,
        probs=val_probs,
        coverage=val_cov,
        threshold=threshold,
    )
    test_metrics, test_pred, test_cov_stats = _evaluate_split(
        smiles=test_smiles,
        labels=test_labels,
        probs=test_probs,
        coverage=test_cov,
        threshold=threshold,
    )

    payload = {
        "direction": "direction_1_tox21_clinical_proxy",
        "threshold": threshold_info,
        "proxy": proxy_config,
        "config": {
            "model_dir": str(model_dir),
            "config_path": str(config_path),
            "split_type": str(args.split_type),
            "seed": int(args.seed),
            "device": device,
            "batch_size": int(args.batch_size),
            "threshold_calibration": str(args.threshold_calibration),
            "threshold_cv_folds": int(args.threshold_cv_folds),
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "coverage": {
            "train": train_cov_stats,
            "val": val_cov_stats,
            "test": test_cov_stats,
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
    print(f"Proxy mode: {proxy_config['proxy_mode']}")
    print(f"Threshold used: {threshold:.4f} ({threshold_info.get('reason')})")
    print(f"Train AUC/F1: {train_metrics['auc_roc']:.4f} / {train_metrics['f1']:.4f}")
    print(f"Val   AUC/F1: {val_metrics['auc_roc']:.4f} / {val_metrics['f1']:.4f}")
    print(f"Test  AUC/F1: {test_metrics['auc_roc']:.4f} / {test_metrics['f1']:.4f}")
    print(
        "Coverage (mean train/val/test): "
        f"{train_cov_stats['mean']:.4f} / {val_cov_stats['mean']:.4f} / {test_cov_stats['mean']:.4f}"
    )
    print(f"Saved metrics to: {metrics_json}")


if __name__ == "__main__":
    main()
