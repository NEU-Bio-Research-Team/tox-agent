#!/usr/bin/env python3
"""
Train Hu et al. pretrained GIN for Tox21 multi-task prediction.

Usage:
  /home/minhquang/miniconda3/envs/drug-tox-env/bin/python scripts/train_pretrained_gin_tox21.py \
      --config config/tox21_pretrained_gin_config.yaml --device cuda
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.data import get_task_names, load_tox21
from backend.pretrained_gnn import create_pretrained_gin_model, smiles_list_to_hu_dataset
from backend.graph_train import evaluate_multitask_model
from backend.utils import ensure_dir, save_metrics, set_seed


class MaskedFocalLoss(nn.Module):
    """Masked multi-task focal loss that ignores NaN labels per task."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=logits.device)
        task_count = 0

        for task_idx in range(logits.shape[1]):
            valid = ~torch.isnan(targets[:, task_idx])
            if valid.sum() < 1:
                continue

            y_true = targets[valid, task_idx]
            y_logit = logits[valid, task_idx]

            bce = nn.functional.binary_cross_entropy_with_logits(y_logit, y_true, reduction="none")
            prob = torch.sigmoid(y_logit)
            p_t = prob * y_true + (1 - prob) * (1 - y_true)
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            focal = alpha_t * (1 - p_t).pow(self.gamma) * bce

            total = total + focal.mean()
            task_count += 1

        if task_count == 0:
            return logits.sum() * 0.0
        return total / task_count


class GraphModelWrapper(nn.Module):
    """Wrap pretrained-gin predictor to the batch-only call style."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


def create_multitask_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create teacher-parity multi-task sampler: balance compounds by any-positive label."""
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {labels.shape}")

    # Treat a compound as positive if it is active in any task.
    any_positive = (np.nanmax(labels, axis=1) > 0).astype(int)
    class_counts = np.bincount(any_positive, minlength=2).astype(np.float32)

    sample_weights = np.ones_like(any_positive, dtype=np.float32)
    for class_idx in range(2):
        count = class_counts[class_idx]
        if count > 0:
            sample_weights[any_positive == class_idx] = 1.0 / (count * 2.0)

    sample_weights = sample_weights / max(float(sample_weights.sum()), 1e-8) * len(sample_weights)
    return WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)


def save_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(history["val_auc"], label="val_auc")
    axes[1].plot(history["val_pr_auc"], label="val_pr_auc", linestyle="--")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/tox21_pretrained_gin_config.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    tc = cfg["training"]
    dc = cfg["data"]
    oc = cfg["output"]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"

    set_seed(int(dc.get("seed", 42)))
    task_names = get_task_names("tox21")

    print("=" * 70)
    print("Pretrained GIN (Hu et al.) - Tox21")
    print("=" * 70)
    print(f"Device: {device}")

    train_df, val_df, test_df = load_tox21(
        cache_dir=str(project_root / dc["cache_dir"]),
        split_type=str(dc.get("split_type", "scaffold")),
        seed=int(dc.get("seed", 42)),
        enforce_workspace_mode=False,
    )

    print(f"Split sizes -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_labels = train_df[task_names].values.astype(np.float32)
    val_labels = val_df[task_names].values.astype(np.float32)
    test_labels = test_df[task_names].values.astype(np.float32)

    print("Converting SMILES to Hu et al. graph features...")
    train_ds = smiles_list_to_hu_dataset(train_df["smiles"].tolist(), train_labels)
    val_ds = smiles_list_to_hu_dataset(val_df["smiles"].tolist(), val_labels)
    test_ds = smiles_list_to_hu_dataset(test_df["smiles"].tolist(), test_labels)
    print(f"Graphs -> train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    batch_size = int(tc.get("batch_size", 64))
    if bool(tc.get("use_weighted_sampler", True)):
        train_valid_labels = np.stack([d.y.numpy().squeeze(0) for d in train_ds])
        train_sampler = create_multitask_sampler(train_valid_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    predictor = create_pretrained_gin_model(
        num_tasks=len(task_names),
        strategy=str(mc.get("strategy", "masking")),
        cache_dir=str(project_root / dc.get("pretrained_cache", "data/pretrained_gnns")),
        emb_dim=int(mc.get("emb_dim", 300)),
        num_layers=int(mc.get("num_layers", 5)),
        drop_ratio=float(mc.get("drop_ratio", 0.5)),
        jk=str(mc.get("jk", "last")),
        head_dropout=float(mc.get("head_dropout", 0.1)),
    )
    model = GraphModelWrapper(predictor).to(device)

    n_backbone = sum(p.numel() for p in predictor.backbone.parameters())
    n_head = sum(p.numel() for p in predictor.head.parameters())
    print(f"Backbone params: {n_backbone:,}")
    print(f"Head params    : {n_head:,}")

    criterion = MaskedFocalLoss(
        alpha=float(tc.get("focal_alpha", 0.25)),
        gamma=float(tc.get("focal_gamma", 2.0)),
    )

    backbone_params = list(predictor.backbone.parameters())
    head_params = list(predictor.head.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": float(tc.get("lr_backbone", 1e-4)), "weight_decay": float(tc.get("weight_decay", 1e-5))},
            {"params": head_params, "lr": float(tc.get("lr_head", 1e-3)), "weight_decay": float(tc.get("weight_decay", 1e-5))},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    num_epochs = int(tc.get("num_epochs", 100))
    patience = int(tc.get("early_stopping_patience", 20))
    grad_clip = float(tc.get("grad_clip", 1.0))

    history = {"train_loss": [], "val_auc": [], "val_pr_auc": []}
    best_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.y.squeeze(1) if batch.y.dim() == 3 else batch.y

            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_metrics = evaluate_multitask_model(
            model=model,
            data_loader=val_loader,
            task_names=task_names,
            device=device,
            return_predictions=False,
        )

        val_auc = float(val_metrics.get("macro_auc_roc", 0.0))
        val_pr_auc = float(val_metrics.get("macro_pr_auc", 0.0))

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr_auc)

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d}/{num_epochs} "
                f"train_loss={train_loss:.4f} val_auc={val_auc:.4f} val_pr_auc={val_pr_auc:.4f} "
                f"best_auc={best_auc:.4f} patience={no_improve}/{patience}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate_multitask_model(
        model=model,
        data_loader=val_loader,
        task_names=task_names,
        device=device,
        return_predictions=False,
    )
    test_metrics = evaluate_multitask_model(
        model=model,
        data_loader=test_loader,
        task_names=task_names,
        device=device,
        return_predictions=False,
    )

    out_dir = project_root / oc["model_dir"]
    ensure_dir(str(out_dir))

    torch.save(predictor.state_dict(), out_dir / "best_model.pt")
    shutil.copy(config_path, out_dir / "config.yaml")

    metrics = {
        "val_mean_auc_roc": float(val_metrics.get("macro_auc_roc", 0.0)),
        "val_mean_pr_auc": float(val_metrics.get("macro_pr_auc", 0.0)),
        "test_mean_auc_roc": float(test_metrics.get("macro_auc_roc", 0.0)),
        "test_mean_pr_auc": float(test_metrics.get("macro_pr_auc", 0.0)),
    }
    per_task = test_metrics.get("task_metrics", {}) or {}
    for task_name, task_metric in per_task.items():
        metrics[f"test_auc_{task_name}"] = float(task_metric.get("auc_roc", np.nan))

    save_metrics(metrics, str(out_dir / "tox21_pretrained_gin_metrics.txt"))
    save_training_curves(history, out_dir / "training_curves.png")

    print("\nValidation summary:")
    print(f"  mean_auc_roc={metrics['val_mean_auc_roc']:.4f} mean_pr_auc={metrics['val_mean_pr_auc']:.4f}")
    print("Test summary:")
    print(f"  mean_auc_roc={metrics['test_mean_auc_roc']:.4f} mean_pr_auc={metrics['test_mean_pr_auc']:.4f}")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
