"""Clinical head utilities for Tox21-to-clinical transfer experiments."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


class ClinicalHead(nn.Module):
    """Small binary classification head on top of Tox21 task features."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


def create_clinical_head(input_dim: int = 12, hidden_dim: int = 32, dropout: float = 0.1) -> ClinicalHead:
    return ClinicalHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)


def scores_dict_to_feature_vector(task_scores: Dict[str, float], task_names: List[str]) -> np.ndarray:
    """Convert task score dict to a feature vector aligned with task_names."""
    vec = []
    for task in task_names:
        raw_val = task_scores.get(task, np.nan)
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            val = np.nan

        if not np.isfinite(val):
            val = 0.0
        vec.append(float(np.clip(val, 0.0, 1.0)))

    return np.asarray(vec, dtype=np.float32)


def calibrate_threshold_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
    default_threshold: float = 0.35,
) -> Dict[str, float]:
    """Calibrate binary threshold by maximizing Youden's J statistic."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)

    if y_true.size == 0 or y_prob.size == 0 or y_true.size != y_prob.size:
        return {
            "threshold": float(default_threshold),
            "youden_j": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "reason": "invalid_input",
        }

    if len(np.unique(y_true)) < 2:
        return {
            "threshold": float(default_threshold),
            "youden_j": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "reason": "insufficient_label_variation",
        }

    grid = np.arange(
        float(threshold_min),
        float(threshold_max) + 0.5 * float(threshold_step),
        float(threshold_step),
        dtype=np.float32,
    )

    best_threshold = float(default_threshold)
    best_j = -2.0
    best_sens = 0.0
    best_spec = 0.0

    for threshold in grid:
        y_pred = (y_prob >= threshold).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j_stat = float(sensitivity + specificity - 1.0)

        # Tie-break toward thresholds closer to 0.5 for stability.
        if (j_stat > best_j) or (
            j_stat == best_j and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_j = j_stat
            best_sens = float(sensitivity)
            best_spec = float(specificity)

    return {
        "threshold": float(best_threshold),
        "youden_j": float(best_j),
        "sensitivity": float(best_sens),
        "specificity": float(best_spec),
        "reason": "optimized_youden_j",
    }


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute binary classification metrics at a fixed threshold."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)

    if y_true.size == 0 or y_prob.size == 0 or y_true.size != y_prob.size:
        return {
            "n_samples": 0,
            "positive_rate": float("nan"),
            "auc_roc": float("nan"),
            "pr_auc": float("nan"),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "threshold": float(threshold),
        }

    y_pred = (y_prob >= float(threshold)).astype(int)

    if len(np.unique(y_true)) > 1:
        auc_roc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        auc_roc = float("nan")
        pr_auc = float("nan")

    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "n_samples": int(y_true.size),
        "positive_rate": float(np.mean(y_true == 1)),
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": float(threshold),
    }
