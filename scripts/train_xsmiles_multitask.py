#!/usr/bin/env python3
"""
Two-phase multi-task training for XSmiles (SMILESGraphHybridPredictor).

Phase 1:
- Train shared backbone + Tox21 multi-task head on Tox21 only.

Phase 2:
- Joint fine-tune with weighted losses from:
  - Tox21 multi-task head
  - ClinTox binary clinical head

Default behavior keeps backward compatibility with existing clinical inference:
- model.forward(...) returns clinical logits.
- model.forward_tox21(...) returns per-task Tox21 logits.
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.clinical_head import (
    calibrate_threshold_youden,
    calibrate_threshold_youden_cv,
    compute_binary_metrics,
)
from backend.data import get_task_names, load_clintox, load_tox21
from backend.graph_data import get_feature_dims, smiles_list_to_pyg_dataset
from backend.graph_models_hybrid import create_hybrid_model
from backend.graph_train import (
    FocalLoss,
    MaskedBCEWithLogitsLoss,
    MaskedFocalLoss,
    WeightedBCELoss,
    compute_multitask_pos_weights,
    evaluate_model,
    evaluate_multitask_model,
    train_multitask_model,
)
from backend.smiles_tokenizer import create_tokenizer_from_smiles
from backend.utils import ensure_dir, save_metrics, set_seed
from backend.workspace_mode import assert_clintox_enabled, assert_tox21_enabled


class HybridDataset:
    """Attach SMILES tokenizer outputs to each PyG graph sample."""

    def __init__(self, pyg_dataset, smiles_list, tokenizer):
        self.pyg_dataset = pyg_dataset
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        data = self.pyg_dataset[idx]
        smiles = self.smiles_list[idx]
        token_ids, attention_mask = self.tokenizer.encode(smiles)
        data.smiles_token_ids = torch.tensor(token_ids, dtype=torch.long)
        data.smiles_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return data


def collate_fn_hybrid(batch):
    """PyG collate with extra SMILES token tensors."""
    batch_data = Batch.from_data_list(batch)

    if hasattr(batch[0], "smiles_token_ids"):
        batch_data.smiles_token_ids = torch.stack([item.smiles_token_ids for item in batch])
        batch_data.smiles_attention_masks = torch.stack([item.smiles_attention_mask for item in batch])

    return batch_data


class HybridClinicalWrapper(nn.Module):
    """Wrapper for evaluate_model/train loops expecting model(batch)->clinical logits."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(
            batch,
            smiles_token_ids=batch.smiles_token_ids if hasattr(batch, "smiles_token_ids") else None,
            smiles_attention_mask=batch.smiles_attention_masks if hasattr(batch, "smiles_attention_masks") else None,
        )


class HybridTox21Wrapper(nn.Module):
    """Wrapper for evaluate_multitask_model/train loops using model.forward_tox21."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.forward_tox21(
            batch,
            smiles_token_ids=batch.smiles_token_ids if hasattr(batch, "smiles_token_ids") else None,
            smiles_attention_mask=batch.smiles_attention_masks if hasattr(batch, "smiles_attention_masks") else None,
        )


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def select_tox21_task_columns(df: pd.DataFrame) -> List[str]:
    expected_tasks = get_task_names("tox21")
    task_columns = [task for task in expected_tasks if task in df.columns]
    if task_columns:
        return task_columns

    fallback = [c for c in df.columns if c != "smiles"]
    if not fallback:
        raise ValueError("No Tox21 task columns found")
    return fallback


def _build_tox21_labels(df: pd.DataFrame, task_columns: List[str]) -> np.ndarray:
    return df[task_columns].astype(np.float32).values


def _extract_tox21_targets(batch: Batch, num_tasks: int) -> torch.Tensor:
    targets = batch.y.float()
    batch_size = int(batch.num_graphs)

    if targets.dim() == 1:
        if targets.numel() != batch_size * num_tasks:
            raise ValueError(
                f"Unexpected tox21 target size {targets.numel()} for batch={batch_size}, tasks={num_tasks}"
            )
        return targets.view(batch_size, num_tasks)

    if targets.dim() == 2:
        if targets.shape == (batch_size, num_tasks):
            return targets
        if targets.numel() == batch_size * num_tasks:
            return targets.view(batch_size, num_tasks)

    raise ValueError(f"Unsupported tox21 target shape: {tuple(targets.shape)}")


def _extract_clintox_targets(batch: Batch) -> torch.Tensor:
    targets = batch.y.float()

    if targets.dim() == 2 and targets.size(1) == 1:
        targets = targets[:, 0]
    elif targets.dim() > 1:
        targets = targets.reshape(-1)

    return targets


def _resolve_beta(
    beta_mode: str,
    beta_clintox: float,
    tox21_train_size: int,
    clintox_train_size: int,
) -> float:
    mode = str(beta_mode).strip().lower()
    if mode == "auto_ratio":
        denom = max(1, int(clintox_train_size))
        return float(max(1.0, tox21_train_size / denom))
    return float(beta_clintox)


def _build_phase2_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    backbone_lr = float(cfg["backbone_learning_rate"])
    head_lr = float(cfg["head_learning_rate"])
    clintox_head_lr_multiplier = float(cfg.get("clintox_head_lr_multiplier", 1.5))
    weight_decay = float(cfg["weight_decay"])

    backbone_params = []
    tox21_head_params = []
    clintox_head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("predictor."):
            clintox_head_params.append(param)
        elif name.startswith("tox21_head."):
            tox21_head_params.append(param)
        else:
            backbone_params.append(param)

    if not backbone_params:
        raise RuntimeError("No backbone parameters found for phase-2 optimizer")
    if not tox21_head_params:
        raise RuntimeError("No tox21_head parameters found for phase-2 optimizer")
    if not clintox_head_params:
        raise RuntimeError("No predictor parameters found for phase-2 optimizer")

    clintox_head_lr = head_lr * clintox_head_lr_multiplier
    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": tox21_head_params, "lr": head_lr},
            {"params": clintox_head_params, "lr": clintox_head_lr},
        ],
        weight_decay=weight_decay,
    )


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


def _create_clintox_criterion(phase_cfg: dict, pos_weight_value: Optional[float]) -> nn.Module:
    loss_type = str(phase_cfg.get("clintox_loss_type", "focal")).strip().lower()

    if loss_type == "focal":
        return FocalLoss(
            alpha=float(phase_cfg.get("clintox_focal_alpha", 0.25)),
            gamma=float(phase_cfg.get("clintox_focal_gamma", 2.0)),
            reduction="mean",
        )

    if loss_type == "weighted_bce":
        return WeightedBCELoss(pos_weight=pos_weight_value, reduction="mean")

    return nn.BCEWithLogitsLoss()


def _next_batch(it, loader):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)
    return batch, it


def _to_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def _safe_metric(metrics: Dict[str, float], key: str, default: float = 0.0) -> float:
    return _to_float(metrics.get(key), default=default)


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


def _evaluate_heads(
    model: nn.Module,
    tox21_loader: DataLoader,
    clintox_loader: DataLoader,
    task_names: List[str],
    device: str,
) -> Tuple[Dict, Dict]:
    tox_wrapper = HybridTox21Wrapper(model)
    clinical_wrapper = HybridClinicalWrapper(model)

    tox_metrics = evaluate_multitask_model(
        model=tox_wrapper,
        data_loader=tox21_loader,
        task_names=task_names,
        device=device,
        return_predictions=False,
    )
    clintox_metrics = evaluate_model(
        model=clinical_wrapper,
        data_loader=clintox_loader,
        device=device,
        return_predictions=False,
    )
    return tox_metrics, clintox_metrics


def _joint_selection_metric(
    selection_metric: str,
    tox_metrics: Dict,
    clintox_metrics: Dict,
) -> float:
    mode = str(selection_metric).strip().lower()
    tox_auc = _safe_metric(tox_metrics, "macro_auc_roc")
    clintox_auc = _safe_metric(clintox_metrics, "auc_roc")

    if mode == "tox21_macro_auc":
        return tox_auc
    if mode == "clintox_auc":
        return clintox_auc
    return 0.5 * tox_auc + 0.5 * clintox_auc


def _run_phase1_pretrain(
    model: nn.Module,
    tox21_train_loader: DataLoader,
    tox21_val_loader: DataLoader,
    task_names: List[str],
    phase1_cfg: dict,
    device: str,
    tox21_pos_weight: Optional[torch.Tensor],
) -> Dict:
    wrapper = HybridTox21Wrapper(model)

    history = train_multitask_model(
        model=wrapper,
        train_loader=tox21_train_loader,
        val_loader=tox21_val_loader,
        task_names=task_names,
        num_epochs=int(phase1_cfg.get("num_epochs", 80)),
        learning_rate=float(phase1_cfg.get("learning_rate", 8e-4)),
        weight_decay=float(phase1_cfg.get("weight_decay", 1e-4)),
        device=device,
        loss_type=str(phase1_cfg.get("loss_type", "focal")),
        focal_alpha=float(phase1_cfg.get("focal_alpha", 0.25)),
        focal_gamma=float(phase1_cfg.get("focal_gamma", 2.0)),
        pos_weight=tox21_pos_weight,
        early_stopping_patience=int(phase1_cfg.get("early_stopping_patience", 20)),
        early_stopping_metric=str(phase1_cfg.get("early_stopping_metric", "macro_auc_roc")),
        verbose=True,
    )

    val_metrics = evaluate_multitask_model(
        model=wrapper,
        data_loader=tox21_val_loader,
        task_names=task_names,
        device=device,
        return_predictions=False,
    )

    return {
        "history": history,
        "val_metrics": val_metrics,
    }


def _run_phase2_joint_finetune(
    model: nn.Module,
    tox21_train_loader: DataLoader,
    tox21_val_loader: DataLoader,
    clintox_train_loader: DataLoader,
    clintox_val_loader: DataLoader,
    task_names: List[str],
    phase2_cfg: dict,
    device: str,
    tox21_pos_weight: Optional[torch.Tensor],
    clintox_pos_weight_value: Optional[float],
    tox21_train_size: int,
    clintox_train_size: int,
) -> Dict:
    optimizer = _build_phase2_optimizer(model, phase2_cfg)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    tox21_criterion = _create_tox21_criterion(phase2_cfg, tox21_pos_weight)
    clintox_criterion = _create_clintox_criterion(phase2_cfg, clintox_pos_weight_value)

    alpha_tox21 = float(phase2_cfg.get("alpha_tox21", 1.0))
    beta_clintox = _resolve_beta(
        beta_mode=str(phase2_cfg.get("beta_mode", "auto_ratio")),
        beta_clintox=float(phase2_cfg.get("beta_clintox", 5.0)),
        tox21_train_size=tox21_train_size,
        clintox_train_size=clintox_train_size,
    )

    grad_clip = float(phase2_cfg.get("grad_clip_norm", 1.0))
    num_epochs = int(phase2_cfg.get("num_epochs", 80))
    patience_limit = int(phase2_cfg.get("early_stopping_patience", 20))
    selection_metric = str(phase2_cfg.get("selection_metric", "joint_auc"))

    history = {
        "train_joint_loss": [],
        "train_tox21_loss": [],
        "train_clintox_loss": [],
        "val_joint_score": [],
        "val_tox21_macro_auc": [],
        "val_clintox_auc": [],
        "val_clintox_f1": [],
    }

    best_state = None
    best_score = float("-inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        tox21_iter = iter(tox21_train_loader)
        clintox_iter = iter(clintox_train_loader)
        steps = max(len(tox21_train_loader), len(clintox_train_loader))

        train_joint_losses = []
        train_tox21_losses = []
        train_clintox_losses = []

        for _ in range(steps):
            tox21_batch, tox21_iter = _next_batch(tox21_iter, tox21_train_loader)
            clintox_batch, clintox_iter = _next_batch(clintox_iter, clintox_train_loader)

            tox21_batch = tox21_batch.to(device)
            clintox_batch = clintox_batch.to(device)

            optimizer.zero_grad()

            tox21_logits = model.forward_tox21(
                tox21_batch,
                smiles_token_ids=tox21_batch.smiles_token_ids if hasattr(tox21_batch, "smiles_token_ids") else None,
                smiles_attention_mask=tox21_batch.smiles_attention_masks if hasattr(tox21_batch, "smiles_attention_masks") else None,
            )
            tox21_targets = _extract_tox21_targets(tox21_batch, num_tasks=len(task_names)).to(device)
            tox21_loss = tox21_criterion(tox21_logits, tox21_targets)

            clintox_logits = model(
                clintox_batch,
                smiles_token_ids=clintox_batch.smiles_token_ids if hasattr(clintox_batch, "smiles_token_ids") else None,
                smiles_attention_mask=clintox_batch.smiles_attention_masks if hasattr(clintox_batch, "smiles_attention_masks") else None,
            ).squeeze(-1)
            clintox_targets = _extract_clintox_targets(clintox_batch).to(device)
            clintox_loss = clintox_criterion(clintox_logits, clintox_targets)

            if torch.isnan(tox21_loss) or torch.isinf(tox21_loss):
                optimizer.zero_grad()
                continue
            if torch.isnan(clintox_loss) or torch.isinf(clintox_loss):
                optimizer.zero_grad()
                continue

            joint_loss = alpha_tox21 * tox21_loss + beta_clintox * clintox_loss
            if torch.isnan(joint_loss) or torch.isinf(joint_loss):
                optimizer.zero_grad()
                continue

            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_joint_losses.append(float(joint_loss.detach().cpu().item()))
            train_tox21_losses.append(float(tox21_loss.detach().cpu().item()))
            train_clintox_losses.append(float(clintox_loss.detach().cpu().item()))

        mean_joint_loss = float(np.mean(train_joint_losses)) if train_joint_losses else float("nan")
        mean_tox21_loss = float(np.mean(train_tox21_losses)) if train_tox21_losses else float("nan")
        mean_clintox_loss = float(np.mean(train_clintox_losses)) if train_clintox_losses else float("nan")

        history["train_joint_loss"].append(mean_joint_loss)
        history["train_tox21_loss"].append(mean_tox21_loss)
        history["train_clintox_loss"].append(mean_clintox_loss)

        val_tox21_metrics, val_clintox_metrics = _evaluate_heads(
            model=model,
            tox21_loader=tox21_val_loader,
            clintox_loader=clintox_val_loader,
            task_names=task_names,
            device=device,
        )

        score = _joint_selection_metric(
            selection_metric=selection_metric,
            tox_metrics=val_tox21_metrics,
            clintox_metrics=val_clintox_metrics,
        )

        if not np.isfinite(score):
            score = float("-inf")

        history["val_joint_score"].append(float(score))
        history["val_tox21_macro_auc"].append(_safe_metric(val_tox21_metrics, "macro_auc_roc"))
        history["val_clintox_auc"].append(_safe_metric(val_clintox_metrics, "auc_roc"))
        history["val_clintox_f1"].append(_safe_metric(val_clintox_metrics, "f1"))

        scheduler.step(float(score) if np.isfinite(score) else 0.0)

        improved = score > best_score
        if improved:
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[Phase2][Epoch {epoch + 1:03d}] "
            f"train_joint={mean_joint_loss:.4f} "
            f"train_tox21={mean_tox21_loss:.4f} "
            f"train_clintox={mean_clintox_loss:.4f} "
            f"val_tox21_auc={_safe_metric(val_tox21_metrics, 'macro_auc_roc'):.4f} "
            f"val_clintox_auc={_safe_metric(val_clintox_metrics, 'auc_roc'):.4f} "
            f"val_clintox_f1={_safe_metric(val_clintox_metrics, 'f1'):.4f} "
            f"joint_score={score:.4f}"
        )

        if patience_counter >= patience_limit:
            print(f"[Phase2] Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_score": float(best_score),
        "beta_clintox_effective": float(beta_clintox),
    }


def _collect_clintox_predictions(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    clinical_wrapper = HybridClinicalWrapper(model)
    return evaluate_model(
        model=clinical_wrapper,
        data_loader=loader,
        device=device,
        return_predictions=True,
    )


def _collect_tox21_predictions(
    model: nn.Module,
    loader: DataLoader,
    task_names: List[str],
    device: str,
) -> Dict:
    tox21_wrapper = HybridTox21Wrapper(model)
    return evaluate_multitask_model(
        model=tox21_wrapper,
        data_loader=loader,
        task_names=task_names,
        device=device,
        return_predictions=True,
    )


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


def _calibrate_clintox_threshold(
    train_preds: Dict,
    val_preds: Dict,
    calibration_cfg: dict,
) -> Dict:
    enabled = bool(calibration_cfg.get("enabled", True))
    if not enabled:
        default_threshold = float(calibration_cfg.get("default_threshold", 0.35))
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
    default_threshold = float(calibration_cfg.get("default_threshold", 0.35))

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


def _evaluate_clintox_with_threshold(preds: Dict, threshold: float) -> Dict:
    y_true = np.asarray(preds.get("labels", []), dtype=np.float32)
    y_prob = np.asarray(preds.get("predictions", []), dtype=np.float32)
    return compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=float(threshold))


def main() -> None:
    assert_clintox_enabled("scripts/train_xsmiles_multitask.py")
    assert_tox21_enabled("scripts/train_xsmiles_multitask.py")

    parser = argparse.ArgumentParser(description="Train XSmiles with dual-head multi-task setup")
    parser.add_argument(
        "--config",
        type=str,
        default="config/xsmiles_multitask_config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Tox21 pretraining and run only phase-2 joint fine-tuning",
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run only phase-1 pretraining and stop",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override",
    )
    args = parser.parse_args()

    if args.skip_phase1 and args.phase1_only:
        raise ValueError("--skip-phase1 and --phase1-only cannot be used together")

    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    model_cfg = config["model"]
    training_cfg = config["training"]
    phase1_cfg = training_cfg.get("phase1", {})
    phase2_cfg = training_cfg.get("phase2", {})
    calibration_cfg = config.get("threshold_calibration", {})
    tox21_calibration_cfg = config.get("tox21_threshold_calibration", {})
    data_cfg = config["data"]
    output_cfg = config.get("output", {})

    seed = int(data_cfg.get("seed", 42))
    set_seed(seed)

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU")
        device = "cpu"

    model_dir_rel = args.output_dir or output_cfg.get("model_dir", "models/smilesgnn_multitask_model")
    model_dir = project_root / model_dir_rel
    ensure_dir(str(model_dir))

    print("=" * 88)
    print("XSmiles Multi-task Training (Phase1: Tox21, Phase2: Joint Tox21+ClinTox)")
    print("=" * 88)
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Output: {model_dir}")

    print("\nLoading datasets...")
    train_clin_df, val_clin_df, test_clin_df = load_clintox(
        cache_dir=str(project_root / data_cfg["cache_dir"]),
        split_type=str(data_cfg.get("split_type", "scaffold")),
        seed=seed,
    )
    train_tox_df, val_tox_df, test_tox_df = load_tox21(
        cache_dir=str(project_root / data_cfg["cache_dir"]),
        split_type=str(data_cfg.get("split_type", "scaffold")),
        seed=seed,
    )

    task_columns = select_tox21_task_columns(train_tox_df)
    print(
        f"ClinTox sizes: train={len(train_clin_df)} val={len(val_clin_df)} test={len(test_clin_df)}"
    )
    print(
        f"Tox21 sizes:   train={len(train_tox_df)} val={len(val_tox_df)} test={len(test_tox_df)}"
    )
    print(f"Tox21 tasks ({len(task_columns)}): {task_columns}")

    print("\nBuilding tokenizer from combined train split..." )
    tokenizer_smiles = pd.concat(
        [train_clin_df["smiles"].astype(str), train_tox_df["smiles"].astype(str)],
        axis=0,
    ).tolist()
    tokenizer = create_tokenizer_from_smiles(
        smiles_list=tokenizer_smiles,
        vocab_size=int(model_cfg.get("smiles_vocab_size", 100)),
        max_length=int(model_cfg.get("smiles_max_length", 128)),
        min_freq=1,
    )
    print(f"Tokenizer vocab size: {len(tokenizer.token_to_id)}")

    print("\nConverting splits to PyG datasets...")
    train_tox_labels = _build_tox21_labels(train_tox_df, task_columns)
    val_tox_labels = _build_tox21_labels(val_tox_df, task_columns)
    test_tox_labels = _build_tox21_labels(test_tox_df, task_columns)

    train_tox_pyg = smiles_list_to_pyg_dataset(
        train_tox_df["smiles"].tolist(),
        labels=train_tox_labels.tolist(),
    )
    val_tox_pyg = smiles_list_to_pyg_dataset(
        val_tox_df["smiles"].tolist(),
        labels=val_tox_labels.tolist(),
    )
    test_tox_pyg = smiles_list_to_pyg_dataset(
        test_tox_df["smiles"].tolist(),
        labels=test_tox_labels.tolist(),
    )

    train_clin_pyg = smiles_list_to_pyg_dataset(
        train_clin_df["smiles"].tolist(),
        labels=train_clin_df["CT_TOX"].astype(float).tolist(),
    )
    val_clin_pyg = smiles_list_to_pyg_dataset(
        val_clin_df["smiles"].tolist(),
        labels=val_clin_df["CT_TOX"].astype(float).tolist(),
    )
    test_clin_pyg = smiles_list_to_pyg_dataset(
        test_clin_df["smiles"].tolist(),
        labels=test_clin_df["CT_TOX"].astype(float).tolist(),
    )

    print(
        "Graph counts - "
        f"tox21(train/val/test)={len(train_tox_pyg)}/{len(val_tox_pyg)}/{len(test_tox_pyg)}, "
        f"clintox(train/val/test)={len(train_clin_pyg)}/{len(val_clin_pyg)}/{len(test_clin_pyg)}"
    )

    train_tox_ds = HybridDataset(train_tox_pyg, train_tox_df["smiles"].astype(str).tolist(), tokenizer)
    val_tox_ds = HybridDataset(val_tox_pyg, val_tox_df["smiles"].astype(str).tolist(), tokenizer)
    test_tox_ds = HybridDataset(test_tox_pyg, test_tox_df["smiles"].astype(str).tolist(), tokenizer)

    train_clin_ds = HybridDataset(train_clin_pyg, train_clin_df["smiles"].astype(str).tolist(), tokenizer)
    val_clin_ds = HybridDataset(val_clin_pyg, val_clin_df["smiles"].astype(str).tolist(), tokenizer)
    test_clin_ds = HybridDataset(test_clin_pyg, test_clin_df["smiles"].astype(str).tolist(), tokenizer)

    phase1_batch_size = int(phase1_cfg.get("batch_size", 64))
    phase2_batch_tox = int(phase2_cfg.get("batch_size_tox21", 64))
    phase2_batch_clin = int(phase2_cfg.get("batch_size_clintox", 32))

    tox21_train_loader_phase1 = DataLoader(
        train_tox_ds,
        batch_size=phase1_batch_size,
        shuffle=True,
        collate_fn=collate_fn_hybrid,
        num_workers=0,
    )
    tox21_val_loader = DataLoader(
        val_tox_ds,
        batch_size=phase1_batch_size,
        shuffle=False,
        collate_fn=collate_fn_hybrid,
        num_workers=0,
    )
    tox21_test_loader = DataLoader(
        test_tox_ds,
        batch_size=phase1_batch_size,
        shuffle=False,
        collate_fn=collate_fn_hybrid,
        num_workers=0,
    )

    tox21_train_loader_phase2 = DataLoader(
        train_tox_ds,
        batch_size=phase2_batch_tox,
        shuffle=True,
        collate_fn=collate_fn_hybrid,
        num_workers=0,
    )
    clintox_train_loader_phase2 = DataLoader(
        train_clin_ds,
        batch_size=phase2_batch_clin,
        shuffle=True,
        collate_fn=collate_fn_hybrid,
        num_workers=0,
    )

    clintox_val_loader = DataLoader(
        val_clin_ds,
        batch_size=phase2_batch_clin,
        shuffle=False,
        collate_fn=collate_fn_hybrid,
        num_workers=0,
    )
    clintox_test_loader = DataLoader(
        test_clin_ds,
        batch_size=phase2_batch_clin,
        shuffle=False,
        collate_fn=collate_fn_hybrid,
        num_workers=0,
    )
    clintox_train_loader_eval = DataLoader(
        train_clin_ds,
        batch_size=phase2_batch_clin,
        shuffle=False,
        collate_fn=collate_fn_hybrid,
        num_workers=0,
    )

    num_node_features, num_edge_features = get_feature_dims()
    model = create_hybrid_model(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_graph_layers=int(model_cfg["num_graph_layers"]),
        graph_model=str(model_cfg.get("graph_model", "gatv2")),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        use_residual=bool(model_cfg.get("use_residual", True)),
        use_jk=bool(model_cfg.get("use_jk", True)),
        jk_mode=str(model_cfg.get("jk_mode", "cat")),
        graph_pooling=str(model_cfg.get("graph_pooling", "meanmax")),
        smiles_vocab_size=len(tokenizer.token_to_id),
        smiles_d_model=int(model_cfg.get("smiles_d_model", 96)),
        smiles_num_layers=int(model_cfg.get("smiles_num_layers", 2)),
        fusion_method=str(model_cfg.get("fusion_method", "attention")),
        output_dim=int(model_cfg.get("clinical_output_dim", 1)),
        tox21_output_dim=int(model_cfg.get("tox21_output_dim", len(task_columns))),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {num_params:,}")

    tox21_pos_weight = None
    if bool(phase1_cfg.get("use_task_pos_weights", True)):
        tox21_pos_weight = compute_multitask_pos_weights(train_tox_labels)

    clintox_pos_weight_value = None
    if bool(phase2_cfg.get("clintox_use_pos_weight", True)):
        y = train_clin_df["CT_TOX"].astype(int).values
        num_pos = int(np.sum(y == 1))
        num_neg = int(np.sum(y == 0))
        if num_pos > 0:
            clintox_pos_weight_value = float(num_neg / num_pos)

    summary: Dict[str, object] = {
        "config_path": str(config_path),
        "device": device,
        "seed": seed,
        "tox21_tasks": task_columns,
    }

    if not args.skip_phase1 and bool(phase1_cfg.get("enabled", True)):
        print("\n" + "=" * 88)
        print("Phase 1 - Tox21 pretraining")
        print("=" * 88)
        phase1_result = _run_phase1_pretrain(
            model=model,
            tox21_train_loader=tox21_train_loader_phase1,
            tox21_val_loader=tox21_val_loader,
            task_names=task_columns,
            phase1_cfg=phase1_cfg,
            device=device,
            tox21_pos_weight=tox21_pos_weight,
        )
        summary["phase1"] = {
            "val_metrics": phase1_result["val_metrics"],
            "history": phase1_result["history"],
        }
    else:
        print("\nSkipping phase-1 pretraining")

    if args.phase1_only:
        print("\nPhase-1 only mode: skipping phase-2 and evaluation outputs")
        return

    print("\n" + "=" * 88)
    print("Phase 2 - Joint fine-tuning (Tox21 + ClinTox)")
    print("=" * 88)
    phase2_result = _run_phase2_joint_finetune(
        model=model,
        tox21_train_loader=tox21_train_loader_phase2,
        tox21_val_loader=tox21_val_loader,
        clintox_train_loader=clintox_train_loader_phase2,
        clintox_val_loader=clintox_val_loader,
        task_names=task_columns,
        phase2_cfg=phase2_cfg,
        device=device,
        tox21_pos_weight=tox21_pos_weight,
        clintox_pos_weight_value=clintox_pos_weight_value,
        tox21_train_size=len(train_tox_df),
        clintox_train_size=len(train_clin_df),
    )

    print(
        f"\nPhase-2 done. effective_beta_clintox={phase2_result['beta_clintox_effective']:.4f}, "
        f"best_selection_score={phase2_result['best_score']:.4f}"
    )

    print("\nRunning final evaluation...")
    tox21_wrapper = HybridTox21Wrapper(model)
    clinical_wrapper = HybridClinicalWrapper(model)

    tox21_val_metrics = evaluate_multitask_model(
        model=tox21_wrapper,
        data_loader=tox21_val_loader,
        task_names=task_columns,
        device=device,
        thresholds=0.5,
        return_predictions=False,
    )
    tox21_test_metrics = evaluate_multitask_model(
        model=tox21_wrapper,
        data_loader=tox21_test_loader,
        task_names=task_columns,
        device=device,
        thresholds=0.5,
        return_predictions=False,
    )

    tox21_val_preds = _collect_tox21_predictions(
        model=model,
        loader=tox21_val_loader,
        task_names=task_columns,
        device=device,
    )
    tox21_threshold_map, tox21_threshold_info = _calibrate_tox21_task_thresholds(
        val_preds=tox21_val_preds,
        task_names=task_columns,
        calibration_cfg=tox21_calibration_cfg,
    )

    tox21_val_metrics_calibrated = evaluate_multitask_model(
        model=tox21_wrapper,
        data_loader=tox21_val_loader,
        task_names=task_columns,
        device=device,
        thresholds=tox21_threshold_map,
        return_predictions=False,
    )
    tox21_test_metrics_calibrated = evaluate_multitask_model(
        model=tox21_wrapper,
        data_loader=tox21_test_loader,
        task_names=task_columns,
        device=device,
        thresholds=tox21_threshold_map,
        return_predictions=False,
    )

    clin_train_preds = _collect_clintox_predictions(model, clintox_train_loader_eval, device)
    clin_val_preds = _collect_clintox_predictions(model, clintox_val_loader, device)
    clin_test_preds = _collect_clintox_predictions(model, clintox_test_loader, device)

    threshold_info = _calibrate_clintox_threshold(
        train_preds=clin_train_preds,
        val_preds=clin_val_preds,
        calibration_cfg=calibration_cfg,
    )
    calibrated_threshold = float(threshold_info.get("threshold", 0.35))

    clin_train_metrics = _evaluate_clintox_with_threshold(clin_train_preds, calibrated_threshold)
    clin_val_metrics = _evaluate_clintox_with_threshold(clin_val_preds, calibrated_threshold)
    clin_test_metrics = _evaluate_clintox_with_threshold(clin_test_preds, calibrated_threshold)

    print("\nFinal Metrics")
    print("-" * 88)
    print(
        f"ClinTox test: AUC={clin_test_metrics['auc_roc']:.4f} "
        f"PR-AUC={clin_test_metrics['pr_auc']:.4f} "
        f"F1={clin_test_metrics['f1']:.4f} "
        f"Threshold={calibrated_threshold:.4f}"
    )
    print(
        f"Tox21 test: macro AUC={_safe_metric(tox21_test_metrics, 'macro_auc_roc'):.4f} "
        f"macro PR-AUC={_safe_metric(tox21_test_metrics, 'macro_pr_auc'):.4f} "
        f"macro F1={_safe_metric(tox21_test_metrics, 'macro_f1'):.4f}"
    )
    print(
        f"Tox21 test (calibrated): macro AUC={_safe_metric(tox21_test_metrics_calibrated, 'macro_auc_roc'):.4f} "
        f"macro PR-AUC={_safe_metric(tox21_test_metrics_calibrated, 'macro_pr_auc'):.4f} "
        f"macro F1={_safe_metric(tox21_test_metrics_calibrated, 'macro_f1'):.4f}"
    )

    model_path = model_dir / "best_model.pt"
    torch.save(model.state_dict(), model_path)

    tokenizer_path = model_dir / "tokenizer.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    clinical_metrics_for_server = {
        "test_auc_roc": float(clin_test_metrics["auc_roc"]),
        "test_accuracy": float(clin_test_metrics["accuracy"]),
        "test_f1": float(clin_test_metrics["f1"]),
        "test_pr_auc": float(clin_test_metrics["pr_auc"]),
        "test_loss": float(_safe_metric(clin_test_preds, "loss")),
    }
    save_metrics(clinical_metrics_for_server, str(model_dir / "smilesgnn_model_metrics.txt"))

    threshold_metrics_path = model_dir / "clinical_threshold_metrics.json"
    with open(threshold_metrics_path, "w") as f:
        json.dump(threshold_info, f, indent=2)

    tox21_thresholds_path = model_dir / "tox21_task_thresholds.json"
    with open(tox21_thresholds_path, "w") as f:
        json.dump(_to_jsonable(tox21_threshold_info), f, indent=2)

    training_summary = {
        **summary,
        "phase2": {
            "best_score": phase2_result["best_score"],
            "beta_clintox_effective": phase2_result["beta_clintox_effective"],
            "history": phase2_result["history"],
        },
        "threshold": threshold_info,
        "metrics": {
            "tox21": {
                "threshold_default_0p5": {
                    "val": tox21_val_metrics,
                    "test": tox21_test_metrics,
                },
                "threshold_calibrated": {
                    "val": tox21_val_metrics_calibrated,
                    "test": tox21_test_metrics_calibrated,
                },
                "threshold_calibration": tox21_threshold_info,
            },
            "clintox": {
                "train": clin_train_metrics,
                "val": clin_val_metrics,
                "test": clin_test_metrics,
            },
        },
        "model_config": {
            "num_node_features": int(num_node_features),
            "num_edge_features": int(num_edge_features),
            "tox21_output_dim": int(model_cfg.get("tox21_output_dim", len(task_columns))),
            "clinical_output_dim": int(model_cfg.get("clinical_output_dim", 1)),
        },
    }

    summary_path = model_dir / "xsmiles_multitask_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(_to_jsonable(training_summary), f, indent=2)

    print("\nArtifacts saved:")
    print(f"- Model: {model_path}")
    print(f"- Tokenizer: {tokenizer_path}")
    print(f"- Clinical metrics: {model_dir / 'smilesgnn_model_metrics.txt'}")
    print(f"- Threshold info: {threshold_metrics_path}")
    print(f"- Tox21 task thresholds: {tox21_thresholds_path}")
    print(f"- Full summary: {summary_path}")


if __name__ == "__main__":
    main()
