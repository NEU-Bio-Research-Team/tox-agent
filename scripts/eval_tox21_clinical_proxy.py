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
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
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
from backend.featurization import featurize_batch
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


def _build_learned_feature_matrix(
    prob_matrix: np.ndarray,
    smiles: np.ndarray,
    impute_value: float,
    use_ecfp4_features: bool,
    ecfp_radius: int,
    ecfp_bits: int,
) -> np.ndarray:
    tox21_features = _impute_prob_matrix(prob_matrix, fill_value=float(impute_value))
    if not use_ecfp4_features:
        return tox21_features

    fp_matrix = featurize_batch(
        smiles_list=[str(s) for s in smiles.tolist()],
        mode="fingerprint",
        radius=int(ecfp_radius),
        n_bits=int(ecfp_bits),
    ).astype(np.float32)
    return np.concatenate([tox21_features, fp_matrix], axis=1).astype(np.float32)


def _parse_c_grid(raw: str) -> list[float]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if np.isfinite(value) and value > 0:
            values.append(value)
    if not values:
        return [0.01, 0.03, 0.1, 0.3, 1.0]
    values = sorted(set(values))
    return [float(v) for v in values]


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


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


def _fit_learned_proxy_lr(
    train_feature_matrix: np.ndarray,
    train_labels: np.ndarray,
    task_names,
    cv_folds: int,
    seed: int,
    lr_max_iter: int,
    lr_regularization: str,
    lr_c_grid: list[float],
):
    x_train = np.asarray(train_feature_matrix, dtype=np.float32)
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
            Cs=[float(c) for c in lr_c_grid],
            cv=int(effective_cv),
            penalty=str(lr_regularization),
            class_weight="balanced",
            max_iter=int(lr_max_iter),
            scoring="roc_auc",
            solver="liblinear",
            random_state=int(seed),
        )
        reason = "logistic_regression_cv"
    else:
        model = LogisticRegression(
            C=float(min(lr_c_grid)),
            penalty=str(lr_regularization),
            class_weight="balanced",
            max_iter=int(lr_max_iter),
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
        "model_type": "logistic_regression",
        "cv_folds_effective": int(effective_cv) if effective_cv >= 2 else 0,
        "lr_regularization": str(lr_regularization),
        "lr_c_grid": [float(c) for c in lr_c_grid],
        "lr_max_iter": int(lr_max_iter),
        "intercept": float(model.intercept_.reshape(-1)[0]),
        "coefficients": coef_map,
    }
    return model, payload


def _fit_learned_proxy_svm_rbf(
    train_feature_matrix: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
    svm_c: float,
    svm_gamma: str,
):
    x_train = np.asarray(train_feature_matrix, dtype=np.float32)
    y_train = np.asarray(train_labels).reshape(-1).astype(int)

    if x_train.shape[0] != y_train.shape[0]:
        raise RuntimeError("Learned proxy fit failed: x/y shape mismatch")

    valid = np.isfinite(y_train)
    x_train = x_train[valid]
    y_train = y_train[valid]

    labels = np.unique(y_train)
    if labels.size < 2:
        raise RuntimeError("Learned proxy fit failed: need both positive and negative labels")

    model = SVC(
        C=float(svm_c),
        kernel="rbf",
        gamma=str(svm_gamma),
        class_weight="balanced",
        probability=True,
        random_state=int(seed),
    )
    model.fit(x_train, y_train)

    payload = {
        "fit_reason": "svm_rbf",
        "model_type": "svm_rbf",
        "svm_c": float(svm_c),
        "svm_gamma": str(svm_gamma),
    }
    return model, payload


def _learned_proxy_probabilities(
    feature_matrix: np.ndarray,
    prob_matrix: np.ndarray,
    model,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(feature_matrix, dtype=np.float32)
    p_proxy = model.predict_proba(x)[:, 1].astype(np.float32)
    coverage = _coverage_from_prob_matrix(prob_matrix)
    return p_proxy, coverage


def _threshold_calibration_data(
    y_train: np.ndarray,
    train_probs: np.ndarray,
    y_val: np.ndarray,
    val_probs: np.ndarray,
    fit_split: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if str(fit_split) == "train_val":
        y_all = np.concatenate([y_train.reshape(-1), y_val.reshape(-1)], axis=0)
        p_all = np.concatenate([train_probs.reshape(-1), val_probs.reshape(-1)], axis=0)
        return y_all.astype(np.float32), p_all.astype(np.float32)
    return y_val.astype(np.float32), val_probs.astype(np.float32)


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
        choices=["weighted", "learned_lr_cv", "learned_svm_rbf"],
        default="learned_lr_cv",
        help="weighted: domain-weighted proxy, learned_*: learned proxy",
    )
    parser.add_argument("--task-weights", type=str, default=None)
    parser.add_argument("--renormalize-missing", action="store_true")
    parser.add_argument("--missing-task-value", type=float, default=0.5)
    parser.add_argument("--learned-impute-value", type=float, default=0.5)
    parser.add_argument("--use-ecfp4-features", action="store_true")
    parser.add_argument("--ecfp-radius", type=int, default=2)
    parser.add_argument("--ecfp-bits", type=int, default=256)
    parser.add_argument("--lr-cv-folds", type=int, default=5)
    parser.add_argument("--lr-max-iter", type=int, default=500)
    parser.add_argument("--lr-regularization", choices=["l1", "l2"], default="l2")
    parser.add_argument("--lr-c-grid", type=str, default="0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--learned-min-val-auc", type=float, default=0.55)
    parser.add_argument("--disable-learned-fallback", action="store_true")
    parser.add_argument("--clinical-threshold", type=float, default=None)
    parser.add_argument("--no-calibrate-threshold", action="store_true")
    parser.add_argument(
        "--threshold-calibration",
        choices=["val", "cv"],
        default="cv",
        help="Threshold calibration strategy when threshold is not provided.",
    )
    parser.add_argument("--threshold-cv-folds", type=int, default=3)
    parser.add_argument(
        "--threshold-fit-split",
        choices=["val", "train_val"],
        default="train_val",
        help="Data used to fit threshold when calibration is enabled.",
    )
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

    train_feature_matrix = _build_learned_feature_matrix(
        prob_matrix=train_prob_matrix,
        smiles=train_smiles,
        impute_value=float(args.learned_impute_value),
        use_ecfp4_features=bool(args.use_ecfp4_features),
        ecfp_radius=int(args.ecfp_radius),
        ecfp_bits=int(args.ecfp_bits),
    )
    val_feature_matrix = _build_learned_feature_matrix(
        prob_matrix=val_prob_matrix,
        smiles=val_smiles,
        impute_value=float(args.learned_impute_value),
        use_ecfp4_features=bool(args.use_ecfp4_features),
        ecfp_radius=int(args.ecfp_radius),
        ecfp_bits=int(args.ecfp_bits),
    )
    test_feature_matrix = _build_learned_feature_matrix(
        prob_matrix=test_prob_matrix,
        smiles=test_smiles,
        impute_value=float(args.learned_impute_value),
        use_ecfp4_features=bool(args.use_ecfp4_features),
        ecfp_radius=int(args.ecfp_radius),
        ecfp_bits=int(args.ecfp_bits),
    )

    task_weights = _load_task_weights(args.task_weights)
    if task_weights is None:
        task_weights = dict(DEFAULT_CLINICAL_PROXY_TASK_WEIGHTS)

    proxy_config: Dict[str, object]
    if str(args.proxy_mode) == "weighted":

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
        if str(args.proxy_mode) == "learned_svm_rbf":
            learned_model, learned_info = _fit_learned_proxy_svm_rbf(
                train_feature_matrix=train_feature_matrix,
                train_labels=train_labels,
                seed=int(args.seed),
                svm_c=float(args.svm_c),
                svm_gamma=str(args.svm_gamma),
            )
        else:
            learned_model, learned_info = _fit_learned_proxy_lr(
                train_feature_matrix=train_feature_matrix,
                train_labels=train_labels,
                task_names=task_names,
                cv_folds=int(args.lr_cv_folds),
                seed=int(args.seed),
                lr_max_iter=int(args.lr_max_iter),
                lr_regularization=str(args.lr_regularization),
                lr_c_grid=_parse_c_grid(args.lr_c_grid),
            )

        train_probs, train_cov = _learned_proxy_probabilities(
            feature_matrix=train_feature_matrix,
            prob_matrix=train_prob_matrix,
            model=learned_model,
        )
        val_probs, val_cov = _learned_proxy_probabilities(
            feature_matrix=val_feature_matrix,
            prob_matrix=val_prob_matrix,
            model=learned_model,
        )
        test_probs, test_cov = _learned_proxy_probabilities(
            feature_matrix=test_feature_matrix,
            prob_matrix=test_prob_matrix,
            model=learned_model,
        )

        learned_val_auc = _safe_auc(val_labels, val_probs)
        min_val_auc = float(args.learned_min_val_auc)
        should_fallback = (
            not bool(args.disable_learned_fallback)
            and (not np.isfinite(learned_val_auc) or learned_val_auc < min_val_auc)
        )

        if should_fallback:
            print(
                "Learned proxy val AUC is below threshold; fallback to weighted proxy: "
                f"val_auc={learned_val_auc:.4f}, min_required={min_val_auc:.4f}"
            )
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
                "fallback_from": str(args.proxy_mode),
                "fallback_reason": "low_validation_auc",
                "learned_val_auc": float(learned_val_auc),
                "learned_min_val_auc": float(min_val_auc),
                "task_weights": task_weights,
                "renormalize_missing": bool(args.renormalize_missing),
                "missing_task_value": float(args.missing_task_value),
                "learned_model": learned_info,
            }
        else:
            proxy_config = {
                "proxy_mode": str(args.proxy_mode),
                "learned_impute_value": float(args.learned_impute_value),
                "use_ecfp4_features": bool(args.use_ecfp4_features),
                "ecfp_radius": int(args.ecfp_radius) if bool(args.use_ecfp4_features) else 0,
                "ecfp_bits": int(args.ecfp_bits) if bool(args.use_ecfp4_features) else 0,
                "lr_cv_folds": int(args.lr_cv_folds),
                "learned_val_auc": float(learned_val_auc),
                "learned_min_val_auc": float(min_val_auc),
                "learned_model": learned_info,
            }

    threshold_y_true, threshold_y_prob = _threshold_calibration_data(
        y_train=train_labels,
        train_probs=train_probs,
        y_val=val_labels,
        val_probs=val_probs,
        fit_split=str(args.threshold_fit_split),
    )

    if args.clinical_threshold is not None:
        threshold_info = {
            "threshold": float(args.clinical_threshold),
            "youden_j": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "reason": "provided_by_user",
            "fit_split": "user_provided",
        }
    elif args.no_calibrate_threshold:
        threshold_info = {
            "threshold": 0.35,
            "youden_j": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "reason": "default_0_35",
            "fit_split": "disabled",
        }
    elif str(args.threshold_calibration) == "cv":
        threshold_info = calibrate_threshold_youden_cv(
            y_true=threshold_y_true,
            y_prob=threshold_y_prob,
            n_splits=int(args.threshold_cv_folds),
            seed=int(args.seed),
            default_threshold=0.35,
        )
        threshold_info["fit_split"] = str(args.threshold_fit_split)
    else:
        threshold_info = calibrate_threshold_youden(
            y_true=threshold_y_true,
            y_prob=threshold_y_prob,
            default_threshold=0.35,
        )
        threshold_info["fit_split"] = str(args.threshold_fit_split)

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
            "threshold_fit_split": str(args.threshold_fit_split),
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
