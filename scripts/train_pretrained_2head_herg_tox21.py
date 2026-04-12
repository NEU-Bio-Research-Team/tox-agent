#!/usr/bin/env python3
"""
Two-phase training for pretrained molecular backbone with dual heads:
- Tox21 multi-task head (12 tasks)
- hERG binary head

Phase 1:
- Train backbone + Tox21 head on Tox21 only.

Phase 2:
- Joint fine-tune backbone + both heads with weighted objective.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.clinical_head import calibrate_threshold_youden, calibrate_threshold_youden_cv, compute_binary_metrics
from backend.data import get_task_names, load_tox21
from backend.graph_train import (
    FocalLoss,
    MaskedBCEWithLogitsLoss,
    MaskedFocalLoss,
    WeightedBCELoss,
    compute_multitask_pos_weights,
    create_balanced_sampler,
)
from backend.pretrained_mol_model import (
    create_pretrained_dual_head_model,
    get_checkpoint_defaults,
)
from backend.utils import ensure_dir, set_seed
from backend.workspace_mode import assert_tox21_enabled

try:
    from tdc.single_pred import Tox
except ImportError:
    Tox = None


class Tox21TokenizedDataset(Dataset):
    """Tokenized SMILES dataset for multi-task Tox21 labels."""

    def __init__(
        self,
        smiles_list: List[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.encodings = tokenizer(
            smiles_list,
            padding="max_length",
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class BinaryTokenizedDataset(Dataset):
    """Tokenized SMILES dataset for binary labels."""

    def __init__(
        self,
        smiles_list: List[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.encodings = tokenizer(
            smiles_list,
            padding="max_length",
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_herg_karim(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if Tox is None:
        raise ImportError("pytdc is required for hERG_Karim. Install with: pip install pytdc")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    data = Tox(name="hERG_Karim", path=str(cache_path))
    split = data.get_split(method=split_type, seed=seed)

    train_df = split["train"].rename(columns={"Drug": "smiles", "Y": "label"})
    val_df = split["valid"].rename(columns={"Drug": "smiles", "Y": "label"})
    test_df = split["test"].rename(columns={"Drug": "smiles", "Y": "label"})

    return (
        train_df[["smiles", "label"]].copy(),
        val_df[["smiles", "label"]].copy(),
        test_df[["smiles", "label"]].copy(),
    )


def select_tox21_task_columns(df: pd.DataFrame) -> List[str]:
    expected = get_task_names("tox21")
    columns = [task for task in expected if task in df.columns]
    if columns:
        return columns

    fallback = [c for c in df.columns if c != "smiles"]
    if not fallback:
        raise ValueError("No Tox21 task columns found")
    return fallback


def _build_tox21_labels(df: pd.DataFrame, task_columns: List[str]) -> np.ndarray:
    return df[task_columns].astype(np.float32).values


def _build_binary_labels(df: pd.DataFrame, label_col: str = "label") -> np.ndarray:
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(np.float32).values
    labels = np.clip(labels, 0.0, 1.0)
    return labels


def _build_tox21_proxy_binary_labels(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 2:
        raise ValueError("labels must be shape (N, T)")
    proxy = np.nanmean(labels, axis=1)
    proxy = np.nan_to_num(proxy, nan=0.0)
    return (proxy >= 0.5).astype(np.int32)


def _to_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _safe_metric(metrics: Dict[str, float], key: str, default: float = 0.0) -> float:
    return _to_float(metrics.get(key), default=default)


def _next_batch(it, loader):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)
    return batch, it


def _create_tox21_criterion(phase_cfg: dict, pos_weight: Optional[torch.Tensor]) -> nn.Module:
    loss_type = str(phase_cfg.get("loss_type", "focal")).strip().lower()

    if loss_type == "focal":
        return MaskedFocalLoss(
            alpha=float(phase_cfg.get("focal_alpha", 0.25)),
            gamma=float(phase_cfg.get("focal_gamma", 2.0)),
            reduction="mean",
        )

    if loss_type == "weighted_bce":
        return MaskedBCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    return MaskedBCEWithLogitsLoss(reduction="mean")


def _create_herg_criterion(phase_cfg: dict, pos_weight_value: Optional[float]) -> nn.Module:
    loss_type = str(phase_cfg.get("herg_loss_type", "focal")).strip().lower()

    if loss_type == "focal":
        return FocalLoss(
            alpha=float(phase_cfg.get("herg_focal_alpha", 0.25)),
            gamma=float(phase_cfg.get("herg_focal_gamma", 2.0)),
            reduction="mean",
        )

    if loss_type == "weighted_bce":
        return WeightedBCELoss(pos_weight=pos_weight_value, reduction="mean")

    return nn.BCEWithLogitsLoss()


def _compute_binary_pos_weight(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    num_pos = float(np.sum(labels == 1.0))
    num_neg = float(np.sum(labels == 0.0))
    if num_pos <= 0 or num_neg <= 0:
        return 1.0
    return float(num_neg / num_pos)


def _resolve_beta(
    beta_mode: str,
    beta_herg: float,
    tox21_train_size: int,
    herg_train_size: int,
) -> float:
    mode = str(beta_mode).strip().lower()
    if mode == "auto_ratio":
        denom = max(1, int(herg_train_size))
        return float(max(1.0, tox21_train_size / denom))
    return float(beta_herg)


def _joint_selection_metric(
    selection_metric: str,
    tox21_metrics: Dict,
    herg_metrics: Dict,
    beta_herg_effective: float,
) -> float:
    mode = str(selection_metric).strip().lower()
    tox_auc = _safe_metric(tox21_metrics, "macro_auc_roc")
    herg_auc = _safe_metric(herg_metrics, "auc_roc")

    if mode == "tox21_macro_auc":
        return tox_auc
    if mode == "herg_auc":
        return herg_auc
    if mode in {"unweighted_joint_auc", "joint_auc_unweighted", "mean_joint_auc"}:
        return 0.5 * tox_auc + 0.5 * herg_auc

    beta = max(0.0, _to_float(beta_herg_effective, default=1.0))
    return (tox_auc + beta * herg_auc) / (1.0 + beta)


def _selection_formula_text(selection_metric: str, beta_herg_effective: float) -> str:
    mode = str(selection_metric).strip().lower()
    if mode == "tox21_macro_auc":
        return "tox21_macro_auc"
    if mode == "herg_auc":
        return "herg_auc"
    if mode in {"unweighted_joint_auc", "joint_auc_unweighted", "mean_joint_auc"}:
        return "0.5 * tox21_macro_auc + 0.5 * herg_auc"
    beta = max(0.0, _to_float(beta_herg_effective, default=1.0))
    return f"(tox21_macro_auc + {beta:.6f} * herg_auc) / (1 + {beta:.6f})"


def _evaluate_tox21(
    model: nn.Module,
    data_loader: DataLoader,
    task_names: List[str],
    device: str,
    criterion: Optional[nn.Module] = None,
    thresholds: Optional[Dict[str, float]] = None,
    return_predictions: bool = False,
) -> Dict:
    model.eval()

    all_logits_chunks: List[np.ndarray] = []
    all_labels_chunks: List[np.ndarray] = []
    losses: List[float] = []

    with torch.no_grad():
        for batch in data_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model.forward_tox21(ids, mask)
            if criterion is not None:
                losses.append(float(criterion(logits, labels).item()))

            all_logits_chunks.append(logits.detach().cpu().numpy())
            all_labels_chunks.append(labels.detach().cpu().numpy())

    if not all_logits_chunks:
        out = {
            "loss": float("nan"),
            "macro_auc_roc": 0.0,
            "macro_pr_auc": 0.0,
            "macro_accuracy": 0.0,
            "macro_f1": 0.0,
            "num_valid_tasks": 0,
            "task_metrics": {},
        }
        if return_predictions:
            out.update(
                {
                    "logits": np.zeros((0, len(task_names)), dtype=np.float32),
                    "probabilities": np.zeros((0, len(task_names)), dtype=np.float32),
                    "labels": np.zeros((0, len(task_names)), dtype=np.float32),
                }
            )
        return out

    all_logits = np.vstack(all_logits_chunks)
    all_labels = np.vstack(all_labels_chunks)
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    task_metrics: Dict[str, Dict[str, float]] = {}
    auc_vals: List[float] = []
    pr_vals: List[float] = []
    acc_vals: List[float] = []
    f1_vals: List[float] = []

    for idx, task_name in enumerate(task_names):
        y_true = all_labels[:, idx]
        y_prob = all_probs[:, idx]
        valid_mask = np.isfinite(y_true)

        thr = 0.5
        if thresholds and task_name in thresholds:
            thr = float(thresholds[task_name])

        if int(np.sum(valid_mask)) == 0:
            task_metrics[task_name] = {
                "n_valid": 0,
                "auc_roc": float("nan"),
                "pr_auc": float("nan"),
                "accuracy": float("nan"),
                "f1": float("nan"),
                "threshold": float(thr),
            }
            auc_vals.append(np.nan)
            pr_vals.append(np.nan)
            acc_vals.append(np.nan)
            f1_vals.append(np.nan)
            continue

        y_true_valid = y_true[valid_mask].astype(np.int32)
        y_prob_valid = y_prob[valid_mask].astype(np.float32)
        y_pred_valid = (y_prob_valid >= float(thr)).astype(np.int32)

        if len(np.unique(y_true_valid)) > 1:
            auc = float(roc_auc_score(y_true_valid, y_prob_valid))
            pr = float(average_precision_score(y_true_valid, y_prob_valid))
        else:
            auc = float("nan")
            pr = float("nan")

        acc = float(accuracy_score(y_true_valid, y_pred_valid))
        f1 = float(f1_score(y_true_valid, y_pred_valid, zero_division=0.0))

        task_metrics[task_name] = {
            "n_valid": int(y_true_valid.shape[0]),
            "auc_roc": auc,
            "pr_auc": pr,
            "accuracy": acc,
            "f1": f1,
            "threshold": float(thr),
        }

        auc_vals.append(auc)
        pr_vals.append(pr)
        acc_vals.append(acc)
        f1_vals.append(f1)

    macro_auc = float(np.nanmean(np.asarray(auc_vals, dtype=np.float32))) if auc_vals else 0.0
    macro_pr = float(np.nanmean(np.asarray(pr_vals, dtype=np.float32))) if pr_vals else 0.0
    macro_acc = float(np.nanmean(np.asarray(acc_vals, dtype=np.float32))) if acc_vals else 0.0
    macro_f1 = float(np.nanmean(np.asarray(f1_vals, dtype=np.float32))) if f1_vals else 0.0

    out = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "macro_auc_roc": macro_auc,
        "macro_pr_auc": macro_pr,
        "macro_accuracy": macro_acc,
        "macro_f1": macro_f1,
        "num_valid_tasks": int(np.sum(np.isfinite(np.asarray(auc_vals, dtype=np.float32)))),
        "task_metrics": task_metrics,
    }

    if return_predictions:
        out.update(
            {
                "logits": all_logits,
                "probabilities": all_probs,
                "labels": all_labels,
            }
        )

    return out


def _evaluate_herg(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    criterion: Optional[nn.Module] = None,
    threshold: float = 0.5,
    return_predictions: bool = False,
) -> Dict:
    model.eval()

    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    losses: List[float] = []

    with torch.no_grad():
        for batch in data_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask)
            if criterion is not None:
                losses.append(float(criterion(logits, labels).item()))

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    if not all_logits:
        out = {
            "loss": float("nan"),
            "auc_roc": float("nan"),
            "pr_auc": float("nan"),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "threshold": float(threshold),
            "n_samples": 0,
        }
        if return_predictions:
            out.update(
                {
                    "labels": np.zeros((0,), dtype=np.float32),
                    "predictions": np.zeros((0,), dtype=np.float32),
                    "logits": np.zeros((0,), dtype=np.float32),
                }
            )
        return out

    logits = np.concatenate(all_logits, axis=0).reshape(-1)
    labels = np.concatenate(all_labels, axis=0).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))

    metrics = compute_binary_metrics(y_true=labels, y_prob=probs, threshold=float(threshold))
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")

    if return_predictions:
        metrics.update(
            {
                "labels": labels.astype(np.float32),
                "predictions": probs.astype(np.float32),
                "logits": logits.astype(np.float32),
            }
        )

    return metrics


def _calibrate_tox21_task_thresholds(
    val_preds: Dict,
    task_names: List[str],
    calibration_cfg: dict,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    default_threshold = float(calibration_cfg.get("default_threshold", 0.5))
    enabled = bool(calibration_cfg.get("enabled", True))
    method = str(calibration_cfg.get("method", "val")).strip().lower()

    threshold_min = float(calibration_cfg.get("min", 0.05))
    threshold_max = float(calibration_cfg.get("max", 0.95))
    threshold_step = float(calibration_cfg.get("step", 0.01))
    cv_folds = int(calibration_cfg.get("cv_folds", 3))

    threshold_map = {task_name: float(default_threshold) for task_name in task_names}
    details: Dict[str, object] = {}

    if not enabled:
        for task_name in task_names:
            details[task_name] = {
                "threshold": float(default_threshold),
                "reason": "disabled",
                "n_valid": 0,
            }
        return threshold_map, {
            "enabled": False,
            "method": method,
            "default_threshold": float(default_threshold),
            "task_thresholds": threshold_map,
            "task_details": details,
        }

    labels = np.asarray(val_preds.get("labels", []), dtype=np.float32)
    probs = np.asarray(val_preds.get("probabilities", []), dtype=np.float32)

    if labels.ndim != 2 or probs.ndim != 2 or labels.shape != probs.shape:
        for task_name in task_names:
            details[task_name] = {
                "threshold": float(default_threshold),
                "reason": "invalid_predictions_shape",
                "n_valid": 0,
            }
        return threshold_map, {
            "enabled": True,
            "method": method,
            "default_threshold": float(default_threshold),
            "task_thresholds": threshold_map,
            "task_details": details,
        }

    for idx, task_name in enumerate(task_names):
        y_true = labels[:, idx]
        y_prob = probs[:, idx]
        valid_mask = np.isfinite(y_true) & np.isfinite(y_prob)

        y_true_valid = y_true[valid_mask].astype(np.int32)
        y_prob_valid = y_prob[valid_mask].astype(np.float32)
        n_valid = int(y_true_valid.shape[0])

        if n_valid == 0:
            details[task_name] = {
                "threshold": float(default_threshold),
                "reason": "no_valid_labels",
                "n_valid": n_valid,
            }
            continue

        if len(np.unique(y_true_valid)) < 2:
            details[task_name] = {
                "threshold": float(default_threshold),
                "reason": "insufficient_label_variation",
                "n_valid": n_valid,
            }
            continue

        if method == "cv":
            info = calibrate_threshold_youden_cv(
                y_true=y_true_valid,
                y_prob=y_prob_valid,
                n_splits=cv_folds,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                default_threshold=default_threshold,
            )
        else:
            info = calibrate_threshold_youden(
                y_true=y_true_valid,
                y_prob=y_prob_valid,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                default_threshold=default_threshold,
            )

        threshold_map[task_name] = float(info.get("threshold", default_threshold))
        details[task_name] = {
            **info,
            "n_valid": n_valid,
            "positive_rate": float(np.mean(y_true_valid == 1)),
        }

    return threshold_map, {
        "enabled": True,
        "method": method,
        "default_threshold": float(default_threshold),
        "min": float(threshold_min),
        "max": float(threshold_max),
        "step": float(threshold_step),
        "cv_folds": int(cv_folds),
        "task_thresholds": threshold_map,
        "task_details": details,
    }


def _calibrate_herg_threshold(
    train_preds: Dict,
    val_preds: Dict,
    calibration_cfg: dict,
) -> Dict:
    enabled = bool(calibration_cfg.get("enabled", True))
    if not enabled:
        default_threshold = float(calibration_cfg.get("default_threshold", 0.5))
        return {
            "threshold": float(default_threshold),
            "youden_j": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "reason": "disabled",
            "fit_split": "disabled",
        }

    fit_split = str(calibration_cfg.get("fit_split", "train_val")).strip().lower()
    method = str(calibration_cfg.get("method", "cv")).strip().lower()

    y_train = np.asarray(train_preds.get("labels", []), dtype=np.float32)
    p_train = np.asarray(train_preds.get("predictions", []), dtype=np.float32)
    y_val = np.asarray(val_preds.get("labels", []), dtype=np.float32)
    p_val = np.asarray(val_preds.get("predictions", []), dtype=np.float32)

    if fit_split == "train_val":
        y_fit = np.concatenate([y_train, y_val], axis=0)
        p_fit = np.concatenate([p_train, p_val], axis=0)
    else:
        y_fit = y_val
        p_fit = p_val

    threshold_min = float(calibration_cfg.get("min", 0.05))
    threshold_max = float(calibration_cfg.get("max", 0.95))
    threshold_step = float(calibration_cfg.get("step", 0.01))
    default_threshold = float(calibration_cfg.get("default_threshold", 0.5))

    if method == "cv":
        info = calibrate_threshold_youden_cv(
            y_true=y_fit,
            y_prob=p_fit,
            n_splits=int(calibration_cfg.get("cv_folds", 3)),
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            default_threshold=default_threshold,
        )
    else:
        info = calibrate_threshold_youden(
            y_true=y_fit,
            y_prob=p_fit,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            default_threshold=default_threshold,
        )

    info["fit_split"] = fit_split
    return info


def _run_phase1_pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_names: List[str],
    phase1_cfg: dict,
    device: str,
    tox21_pos_weight: Optional[torch.Tensor],
) -> Dict:
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    tox21_head_params = [p for p in model.tox21_head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": float(phase1_cfg.get("backbone_learning_rate", phase1_cfg.get("learning_rate", 1e-5))),
            },
            {
                "params": tox21_head_params,
                "lr": float(phase1_cfg.get("head_learning_rate", phase1_cfg.get("learning_rate", 5e-4))),
            },
        ],
        weight_decay=float(phase1_cfg.get("weight_decay", 1e-4)),
    )

    num_epochs = int(phase1_cfg.get("num_epochs", 30))
    warmup_ratio = float(phase1_cfg.get("warmup_ratio", 0.1))
    total_steps = max(1, len(train_loader) * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = _create_tox21_criterion(phase1_cfg, tox21_pos_weight)
    grad_clip = float(phase1_cfg.get("grad_clip_norm", 1.0))
    patience_limit = int(phase1_cfg.get("early_stopping_patience", 12))
    selection_metric = str(phase1_cfg.get("early_stopping_metric", "macro_auc_roc"))
    log_every = int(phase1_cfg.get("log_every_n_epochs", 1))
    log_every_steps = int(phase1_cfg.get("log_every_n_steps", 0))

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_auc": [],
        "val_macro_pr_auc": [],
    }

    best_state = None
    best_score = float("-inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.perf_counter()
        train_losses: List[float] = []
        skipped_batches = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model.forward_tox21(ids, mask)
            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                skipped_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

            if log_every_steps > 0 and step_idx % max(1, log_every_steps) == 0:
                lr_backbone = float(optimizer.param_groups[0]["lr"])
                lr_head = float(optimizer.param_groups[min(1, len(optimizer.param_groups) - 1)]["lr"])
                recent_avg = float(np.mean(train_losses[-max(1, log_every_steps):])) if train_losses else float("nan")
                _log(
                    f"[Phase1][Epoch {epoch + 1:03d}][Step {step_idx:04d}/{len(train_loader):04d}] "
                    f"loss={recent_avg:.4f} lr_backbone={lr_backbone:.2e} lr_head={lr_head:.2e}"
                )

        if not train_losses:
            raise RuntimeError(
                "[Phase1] All batches were skipped due to NaN/Inf loss. "
                "Check learning-rate/loss settings and input labels."
            )

        val_metrics = _evaluate_tox21(
            model=model,
            data_loader=val_loader,
            task_names=task_names,
            device=device,
            criterion=criterion,
            return_predictions=False,
        )

        history["train_loss"].append(float(np.mean(train_losses)) if train_losses else float("nan"))
        history["val_loss"].append(_safe_metric(val_metrics, "loss", default=float("nan")))
        history["val_macro_auc"].append(_safe_metric(val_metrics, "macro_auc_roc"))
        history["val_macro_pr_auc"].append(_safe_metric(val_metrics, "macro_pr_auc"))

        score = _safe_metric(val_metrics, selection_metric, default=float("-inf"))
        if not np.isfinite(score):
            score = float("-inf")

        if score > best_score:
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_seconds = float(time.perf_counter() - epoch_start)

        if (epoch + 1) % max(1, log_every) == 0 or epoch == 0:
            _log(
                f"[Phase1][Epoch {epoch + 1:03d}] "
                f"train_loss={history['train_loss'][-1]:.4f} "
                f"val_loss={history['val_loss'][-1]:.4f} "
                f"val_macro_auc={history['val_macro_auc'][-1]:.4f} "
                f"val_macro_pr_auc={history['val_macro_pr_auc'][-1]:.4f} "
                f"skipped_batches={skipped_batches} epoch_sec={epoch_seconds:.1f}"
            )

        if patience_counter >= patience_limit:
            _log(f"[Phase1] Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_metrics = _evaluate_tox21(
        model=model,
        data_loader=val_loader,
        task_names=task_names,
        device=device,
        criterion=criterion,
        return_predictions=False,
    )

    return {
        "history": history,
        "best_score": float(best_score),
        "selection_metric": selection_metric,
        "val_metrics": final_val_metrics,
    }


def _run_phase2_joint_finetune(
    model: nn.Module,
    tox21_train_loader: DataLoader,
    tox21_val_loader: DataLoader,
    herg_train_loader: DataLoader,
    herg_val_loader: DataLoader,
    task_names: List[str],
    phase2_cfg: dict,
    device: str,
    tox21_pos_weight: Optional[torch.Tensor],
    herg_pos_weight_value: Optional[float],
    tox21_train_size: int,
    herg_train_size: int,
) -> Dict:
    freeze_layers = int(phase2_cfg.get("freeze_layers", phase2_cfg.get("freeze_backbone_layers", 0)))
    freeze_embeddings = bool(phase2_cfg.get("freeze_embeddings", False))
    freeze_report = model.freeze_backbone_layers(
        freeze_layers=freeze_layers,
        freeze_embeddings=freeze_embeddings,
    )

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    tox21_head_params = [p for p in model.tox21_head.parameters() if p.requires_grad]
    herg_head_params = [p for p in model.herg_head.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(phase2_cfg.get("backbone_learning_rate", 1e-5)),
            }
        )
    param_groups.append(
        {
            "params": tox21_head_params,
            "lr": float(phase2_cfg.get("tox21_head_learning_rate", phase2_cfg.get("head_learning_rate", 3e-4))),
        }
    )
    param_groups.append(
        {
            "params": herg_head_params,
            "lr": float(phase2_cfg.get("herg_head_learning_rate", phase2_cfg.get("head_learning_rate", 6e-4))),
        }
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(phase2_cfg.get("weight_decay", 1e-4)),
    )

    num_epochs = int(phase2_cfg.get("num_epochs", 20))
    warmup_ratio = float(phase2_cfg.get("warmup_ratio", 0.1))
    total_steps = max(1, max(len(tox21_train_loader), len(herg_train_loader)) * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    tox21_criterion = _create_tox21_criterion(phase2_cfg, tox21_pos_weight)
    herg_criterion = _create_herg_criterion(phase2_cfg, herg_pos_weight_value)

    alpha_tox21 = float(phase2_cfg.get("alpha_tox21", 1.0))
    beta_herg = _resolve_beta(
        beta_mode=str(phase2_cfg.get("beta_mode", "fixed")),
        beta_herg=float(phase2_cfg.get("beta_herg", 3.0)),
        tox21_train_size=tox21_train_size,
        herg_train_size=herg_train_size,
    )

    grad_clip = float(phase2_cfg.get("grad_clip_norm", 1.0))
    patience_limit = int(phase2_cfg.get("early_stopping_patience", 12))
    selection_metric = str(phase2_cfg.get("selection_metric", "joint_auc"))
    log_every = int(phase2_cfg.get("log_every_n_epochs", 1))
    log_every_steps = int(phase2_cfg.get("log_every_n_steps", 0))

    history = {
        "train_joint_loss": [],
        "train_tox21_loss": [],
        "train_herg_loss": [],
        "val_joint_score": [],
        "val_tox21_macro_auc": [],
        "val_herg_auc": [],
        "val_herg_f1": [],
        "val_weighted_joint_auc": [],
        "val_unweighted_joint_auc": [],
    }

    best_state = None
    best_score = float("-inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.perf_counter()

        tox_it = iter(tox21_train_loader)
        herg_it = iter(herg_train_loader)
        steps = max(len(tox21_train_loader), len(herg_train_loader))

        train_joint_losses: List[float] = []
        train_tox_losses: List[float] = []
        train_herg_losses: List[float] = []
        skipped_steps = 0

        for step_idx in range(1, steps + 1):
            tox_batch, tox_it = _next_batch(tox_it, tox21_train_loader)
            herg_batch, herg_it = _next_batch(herg_it, herg_train_loader)

            tox_ids = tox_batch["input_ids"].to(device)
            tox_mask = tox_batch["attention_mask"].to(device)
            tox_labels = tox_batch["labels"].to(device)

            herg_ids = herg_batch["input_ids"].to(device)
            herg_mask = herg_batch["attention_mask"].to(device)
            herg_labels = herg_batch["labels"].to(device)

            optimizer.zero_grad()

            tox_logits = model.forward_tox21(tox_ids, tox_mask)
            tox_loss = tox21_criterion(tox_logits, tox_labels)

            herg_logits = model(herg_ids, herg_mask)
            herg_loss = herg_criterion(herg_logits, herg_labels)

            if torch.isnan(tox_loss) or torch.isinf(tox_loss):
                optimizer.zero_grad()
                skipped_steps += 1
                continue
            if torch.isnan(herg_loss) or torch.isinf(herg_loss):
                optimizer.zero_grad()
                skipped_steps += 1
                continue

            joint_loss = alpha_tox21 * tox_loss + beta_herg * herg_loss
            if torch.isnan(joint_loss) or torch.isinf(joint_loss):
                optimizer.zero_grad()
                skipped_steps += 1
                continue

            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            train_joint_losses.append(float(joint_loss.detach().cpu().item()))
            train_tox_losses.append(float(tox_loss.detach().cpu().item()))
            train_herg_losses.append(float(herg_loss.detach().cpu().item()))

            if log_every_steps > 0 and step_idx % max(1, log_every_steps) == 0:
                lr_backbone = float(optimizer.param_groups[0]["lr"])
                lr_tox = float(optimizer.param_groups[min(1, len(optimizer.param_groups) - 1)]["lr"])
                lr_herg = float(optimizer.param_groups[min(2, len(optimizer.param_groups) - 1)]["lr"])
                recent_joint = float(np.mean(train_joint_losses[-max(1, log_every_steps):])) if train_joint_losses else float("nan")
                _log(
                    f"[Phase2][Epoch {epoch + 1:03d}][Step {step_idx:04d}/{steps:04d}] "
                    f"joint={recent_joint:.4f} lr_backbone={lr_backbone:.2e} "
                    f"lr_tox={lr_tox:.2e} lr_herg={lr_herg:.2e}"
                )

        if not train_joint_losses:
            raise RuntimeError(
                "[Phase2] All steps were skipped due to NaN/Inf loss. "
                "Check phase2 loss/learning-rate settings."
            )

        history["train_joint_loss"].append(float(np.mean(train_joint_losses)) if train_joint_losses else float("nan"))
        history["train_tox21_loss"].append(float(np.mean(train_tox_losses)) if train_tox_losses else float("nan"))
        history["train_herg_loss"].append(float(np.mean(train_herg_losses)) if train_herg_losses else float("nan"))

        val_tox_metrics = _evaluate_tox21(
            model=model,
            data_loader=tox21_val_loader,
            task_names=task_names,
            device=device,
            criterion=tox21_criterion,
            return_predictions=False,
        )
        val_herg_metrics = _evaluate_herg(
            model=model,
            data_loader=herg_val_loader,
            device=device,
            criterion=herg_criterion,
            threshold=0.5,
            return_predictions=False,
        )

        score = _joint_selection_metric(
            selection_metric=selection_metric,
            tox21_metrics=val_tox_metrics,
            herg_metrics=val_herg_metrics,
            beta_herg_effective=beta_herg,
        )
        weighted_joint_score = _joint_selection_metric(
            selection_metric="joint_auc",
            tox21_metrics=val_tox_metrics,
            herg_metrics=val_herg_metrics,
            beta_herg_effective=beta_herg,
        )
        unweighted_joint_score = _joint_selection_metric(
            selection_metric="unweighted_joint_auc",
            tox21_metrics=val_tox_metrics,
            herg_metrics=val_herg_metrics,
            beta_herg_effective=beta_herg,
        )

        if not np.isfinite(score):
            score = float("-inf")

        history["val_joint_score"].append(float(score))
        history["val_tox21_macro_auc"].append(_safe_metric(val_tox_metrics, "macro_auc_roc"))
        history["val_herg_auc"].append(_safe_metric(val_herg_metrics, "auc_roc"))
        history["val_herg_f1"].append(_safe_metric(val_herg_metrics, "f1"))
        history["val_weighted_joint_auc"].append(float(weighted_joint_score))
        history["val_unweighted_joint_auc"].append(float(unweighted_joint_score))

        if score > best_score:
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_seconds = float(time.perf_counter() - epoch_start)

        if (epoch + 1) % max(1, log_every) == 0 or epoch == 0:
            _log(
                f"[Phase2][Epoch {epoch + 1:03d}] "
                f"train_joint={history['train_joint_loss'][-1]:.4f} "
                f"train_tox21={history['train_tox21_loss'][-1]:.4f} "
                f"train_herg={history['train_herg_loss'][-1]:.4f} "
                f"val_tox21_auc={history['val_tox21_macro_auc'][-1]:.4f} "
                f"val_herg_auc={history['val_herg_auc'][-1]:.4f} "
                f"joint_score={history['val_joint_score'][-1]:.4f} "
                f"skipped_steps={skipped_steps} epoch_sec={epoch_seconds:.1f}"
            )

        if patience_counter >= patience_limit:
            _log(f"[Phase2] Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_score": float(best_score),
        "beta_herg_effective": float(beta_herg),
        "selection_metric": selection_metric,
        "selection_formula": _selection_formula_text(selection_metric, beta_herg),
        "freeze_report": freeze_report,
    }


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _log(message: str) -> None:
    print(str(message), flush=True)


def main() -> None:
    assert_tox21_enabled("scripts/train_pretrained_2head_herg_tox21.py")

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Train pretrained dual-head model (Tox21 + hERG)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pretrained_2head_herg_chemberta_config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip phase-1 tox21 pretraining")
    parser.add_argument("--phase1-only", action="store_true", help="Run only phase-1 and stop")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output override")
    args = parser.parse_args()

    if args.skip_phase1 and args.phase1_only:
        raise ValueError("--skip-phase1 and --phase1-only cannot be used together")

    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    phase1_cfg = training_cfg.get("phase1", {})
    phase2_cfg = training_cfg.get("phase2", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    herg_thr_cfg = config.get("threshold_calibration", {})
    tox21_thr_cfg = config.get("tox21_threshold_calibration", {})

    seed = int(data_cfg.get("seed", 42))
    set_seed(seed)

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU")
        device = "cpu"

    model_dir_rel = args.output_dir or output_cfg.get("model_dir", "models/pretrained_2head_herg_tox21_model")
    model_dir = project_root / model_dir_rel
    ensure_dir(str(model_dir))

    checkpoint_name = str(model_cfg["pretrained_model"])
    ckpt_defaults = get_checkpoint_defaults(checkpoint_name)
    max_length = int(training_cfg.get("max_length", ckpt_defaults["max_length"]))

    _log("=" * 88)
    _log("Pretrained Dual-Head Training (Phase1: Tox21, Phase2: Joint Tox21+hERG)")
    _log("=" * 88)
    _log(f"Device: {device}")
    _log(f"Config: {config_path}")
    _log(f"Checkpoint: {checkpoint_name}")
    _log(f"Max length: {max_length}")
    _log(f"Output: {model_dir}")

    _log("\nLoading datasets...")
    cache_dir = str(project_root / data_cfg.get("cache_dir", "data"))
    split_type = str(data_cfg.get("split_type", "scaffold"))

    _log("- Loading Tox21 split...")
    train_tox_df, val_tox_df, test_tox_df = load_tox21(
        cache_dir=cache_dir,
        split_type=split_type,
        seed=seed,
    )
    _log("- Loading hERG_Karim split...")
    train_herg_df, val_herg_df, test_herg_df = load_herg_karim(
        cache_dir=cache_dir,
        split_type=split_type,
        seed=seed,
    )

    task_columns = select_tox21_task_columns(train_tox_df)
    _log(
        f"Tox21 sizes: train={len(train_tox_df)} val={len(val_tox_df)} test={len(test_tox_df)}"
    )
    _log(
        f"hERG sizes:  train={len(train_herg_df)} val={len(val_herg_df)} test={len(test_herg_df)}"
    )
    _log(f"Tox21 tasks ({len(task_columns)}): {task_columns}")

    _log("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_name,
        trust_remote_code=bool(ckpt_defaults["trust_remote_code"]),
    )

    train_tox_labels = _build_tox21_labels(train_tox_df, task_columns)
    val_tox_labels = _build_tox21_labels(val_tox_df, task_columns)
    test_tox_labels = _build_tox21_labels(test_tox_df, task_columns)

    train_herg_labels = _build_binary_labels(train_herg_df, label_col="label")
    val_herg_labels = _build_binary_labels(val_herg_df, label_col="label")
    test_herg_labels = _build_binary_labels(test_herg_df, label_col="label")

    train_tox_ds = Tox21TokenizedDataset(
        smiles_list=train_tox_df["smiles"].astype(str).tolist(),
        labels=train_tox_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_tox_ds = Tox21TokenizedDataset(
        smiles_list=val_tox_df["smiles"].astype(str).tolist(),
        labels=val_tox_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_tox_ds = Tox21TokenizedDataset(
        smiles_list=test_tox_df["smiles"].astype(str).tolist(),
        labels=test_tox_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_herg_ds = BinaryTokenizedDataset(
        smiles_list=train_herg_df["smiles"].astype(str).tolist(),
        labels=train_herg_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_herg_ds = BinaryTokenizedDataset(
        smiles_list=val_herg_df["smiles"].astype(str).tolist(),
        labels=val_herg_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_herg_ds = BinaryTokenizedDataset(
        smiles_list=test_herg_df["smiles"].astype(str).tolist(),
        labels=test_herg_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    phase1_batch = int(phase1_cfg.get("batch_size", 32))
    phase2_batch_tox = int(phase2_cfg.get("batch_size_tox21", 32))
    phase2_batch_herg = int(phase2_cfg.get("batch_size_herg", 32))

    tox21_train_sampler = None
    if bool(phase1_cfg.get("use_weighted_sampler", False)):
        tox_proxy = _build_tox21_proxy_binary_labels(train_tox_labels)
        tox21_train_sampler = create_balanced_sampler(labels=tox_proxy.tolist(), replacement=True)

    herg_train_sampler = None
    if bool(phase2_cfg.get("use_weighted_sampler_herg", True)):
        herg_train_sampler = create_balanced_sampler(labels=train_herg_labels.astype(int).tolist(), replacement=True)

    tox21_train_loader_phase1 = DataLoader(
        train_tox_ds,
        batch_size=phase1_batch,
        shuffle=(tox21_train_sampler is None),
        sampler=tox21_train_sampler,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    tox21_val_loader = DataLoader(
        val_tox_ds,
        batch_size=max(phase1_batch, phase2_batch_tox),
        shuffle=False,
        num_workers=0,
    )
    tox21_test_loader = DataLoader(
        test_tox_ds,
        batch_size=max(phase1_batch, phase2_batch_tox),
        shuffle=False,
        num_workers=0,
    )

    tox21_train_loader_phase2 = DataLoader(
        train_tox_ds,
        batch_size=phase2_batch_tox,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    herg_train_loader_phase2 = DataLoader(
        train_herg_ds,
        batch_size=phase2_batch_herg,
        shuffle=(herg_train_sampler is None),
        sampler=herg_train_sampler,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    herg_val_loader = DataLoader(
        val_herg_ds,
        batch_size=max(phase2_batch_herg, phase2_batch_tox),
        shuffle=False,
        num_workers=0,
    )
    herg_test_loader = DataLoader(
        test_herg_ds,
        batch_size=max(phase2_batch_herg, phase2_batch_tox),
        shuffle=False,
        num_workers=0,
    )

    _log("\nBuilding dual-head model...")
    model = create_pretrained_dual_head_model(
        pretrained_model=checkpoint_name,
        num_tox21_tasks=int(model_cfg.get("num_tox21_tasks", len(task_columns))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        herg_hidden_dim=model_cfg.get("herg_hidden_dim"),
        use_herg_mlp=bool(model_cfg.get("use_herg_mlp", True)),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")

    tox21_pos_weight = None
    if bool(phase1_cfg.get("use_task_pos_weights", True)) or bool(phase2_cfg.get("use_task_pos_weights", True)):
        tox21_pos_weight = compute_multitask_pos_weights(train_tox_labels).to(device)

    herg_pos_weight = None
    if bool(phase2_cfg.get("herg_use_pos_weight", True)):
        herg_pos_weight = _compute_binary_pos_weight(train_herg_labels)

    phase1_result = None
    if not args.skip_phase1:
        _log(
            "\n[Phase1] Pretraining on Tox21... "
            f"steps_per_epoch={len(tox21_train_loader_phase1)} batch_size={phase1_batch}"
        )
        phase1_result = _run_phase1_pretrain(
            model=model,
            train_loader=tox21_train_loader_phase1,
            val_loader=tox21_val_loader,
            task_names=task_columns,
            phase1_cfg=phase1_cfg,
            device=device,
            tox21_pos_weight=tox21_pos_weight,
        )
        _log(
            f"[Phase1] Done. best_score={phase1_result['best_score']:.4f}, "
            f"val_macro_auc={phase1_result['val_metrics']['macro_auc_roc']:.4f}"
        )

    if args.phase1_only:
        model_path = model_dir / "best_model.pt"
        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "task_names": task_columns,
            "pretrained_model": checkpoint_name,
            "max_length": int(max_length),
            "phase": "phase1_only",
        }
        torch.save(ckpt_payload, model_path)
        tokenizer.save_pretrained(str(model_dir / "tokenizer"))
        shutil.copy(config_path, model_dir / "config.yaml")

        summary = {
            "phase1": phase1_result,
            "phase2": None,
            "model_path": str(model_path),
            "tokenizer_dir": str(model_dir / "tokenizer"),
        }
        with open(model_dir / "pretrained_2head_phase1_summary.json", "w") as f:
            json.dump(_to_jsonable(summary), f, indent=2)

        _log("\nPhase-1 only run completed.")
        _log(f"- Model checkpoint: {model_path}")
        _log(f"- Summary: {model_dir / 'pretrained_2head_phase1_summary.json'}")
        return

    _log(
        "\n[Phase2] Joint fine-tuning on Tox21 + hERG... "
        f"tox_steps={len(tox21_train_loader_phase2)} herg_steps={len(herg_train_loader_phase2)} "
        f"steps_per_epoch={max(len(tox21_train_loader_phase2), len(herg_train_loader_phase2))}"
    )
    phase2_result = _run_phase2_joint_finetune(
        model=model,
        tox21_train_loader=tox21_train_loader_phase2,
        tox21_val_loader=tox21_val_loader,
        herg_train_loader=herg_train_loader_phase2,
        herg_val_loader=herg_val_loader,
        task_names=task_columns,
        phase2_cfg=phase2_cfg,
        device=device,
        tox21_pos_weight=tox21_pos_weight,
        herg_pos_weight_value=herg_pos_weight,
        tox21_train_size=len(train_tox_ds),
        herg_train_size=len(train_herg_ds),
    )

    _log(
        f"\n[Phase2] Done. effective_beta_herg={phase2_result['beta_herg_effective']:.4f}, "
        f"best_selection_score={phase2_result['best_score']:.4f}"
    )

    _log("\nCollecting validation predictions for threshold calibration...")
    tox21_val_preds = _evaluate_tox21(
        model=model,
        data_loader=tox21_val_loader,
        task_names=task_columns,
        device=device,
        criterion=None,
        thresholds=None,
        return_predictions=True,
    )
    herg_train_preds = _evaluate_herg(
        model=model,
        data_loader=herg_train_loader_phase2,
        device=device,
        criterion=None,
        threshold=0.5,
        return_predictions=True,
    )
    herg_val_preds = _evaluate_herg(
        model=model,
        data_loader=herg_val_loader,
        device=device,
        criterion=None,
        threshold=0.5,
        return_predictions=True,
    )

    tox21_threshold_map, tox21_threshold_info = _calibrate_tox21_task_thresholds(
        val_preds=tox21_val_preds,
        task_names=task_columns,
        calibration_cfg=tox21_thr_cfg,
    )
    herg_threshold_info = _calibrate_herg_threshold(
        train_preds=herg_train_preds,
        val_preds=herg_val_preds,
        calibration_cfg=herg_thr_cfg,
    )

    herg_threshold = float(herg_threshold_info.get("threshold", 0.5))

    _log("\nEvaluating final model on validation + test with calibrated thresholds...")
    tox21_val_metrics = _evaluate_tox21(
        model=model,
        data_loader=tox21_val_loader,
        task_names=task_columns,
        device=device,
        criterion=None,
        thresholds=tox21_threshold_map,
        return_predictions=False,
    )
    tox21_test_metrics = _evaluate_tox21(
        model=model,
        data_loader=tox21_test_loader,
        task_names=task_columns,
        device=device,
        criterion=None,
        thresholds=tox21_threshold_map,
        return_predictions=False,
    )

    herg_val_metrics = _evaluate_herg(
        model=model,
        data_loader=herg_val_loader,
        device=device,
        criterion=None,
        threshold=herg_threshold,
        return_predictions=False,
    )
    herg_test_metrics = _evaluate_herg(
        model=model,
        data_loader=herg_test_loader,
        device=device,
        criterion=None,
        threshold=herg_threshold,
        return_predictions=False,
    )

    model_path = model_dir / "best_model.pt"
    ckpt_payload = {
        "model_state_dict": model.state_dict(),
        "task_names": task_columns,
        "pretrained_model": checkpoint_name,
        "max_length": int(max_length),
        "tox21_thresholds": tox21_threshold_map,
        "herg_threshold": herg_threshold,
        "model_config": {
            "dropout": float(model_cfg.get("dropout", 0.1)),
            "use_herg_mlp": bool(model_cfg.get("use_herg_mlp", True)),
            "herg_hidden_dim": model_cfg.get("herg_hidden_dim"),
            "num_tox21_tasks": int(model_cfg.get("num_tox21_tasks", len(task_columns))),
        },
    }
    torch.save(ckpt_payload, model_path)
    tokenizer.save_pretrained(str(model_dir / "tokenizer"))
    shutil.copy(config_path, model_dir / "config.yaml")

    with open(model_dir / "tox21_task_thresholds.json", "w") as f:
        json.dump(_to_jsonable(tox21_threshold_info), f, indent=2)
    with open(model_dir / "herg_threshold.json", "w") as f:
        json.dump(_to_jsonable(herg_threshold_info), f, indent=2)

    summary = {
        "phase1": phase1_result,
        "phase2": phase2_result,
        "validation": {
            "tox21": tox21_val_metrics,
            "herg": herg_val_metrics,
        },
        "test": {
            "tox21": tox21_test_metrics,
            "herg": herg_test_metrics,
        },
        "tox21_threshold_calibration": tox21_threshold_info,
        "herg_threshold_calibration": herg_threshold_info,
        "paths": {
            "model_path": str(model_path),
            "tokenizer_dir": str(model_dir / "tokenizer"),
            "tox21_thresholds": str(model_dir / "tox21_task_thresholds.json"),
            "herg_threshold": str(model_dir / "herg_threshold.json"),
            "config_snapshot": str(model_dir / "config.yaml"),
        },
    }

    summary_path = model_dir / "pretrained_2head_herg_tox21_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(_to_jsonable(summary), f, indent=2)

    _log("\nTraining completed successfully.")
    _log(f"- Model checkpoint: {model_path}")
    _log(f"- Metrics summary: {summary_path}")
    _log(f"- Tox21 thresholds: {model_dir / 'tox21_task_thresholds.json'}")
    _log(f"- hERG threshold: {model_dir / 'herg_threshold.json'}")


if __name__ == "__main__":
    main()
