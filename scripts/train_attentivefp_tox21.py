#!/usr/bin/env python3
"""
Train AttentiveFP on the Tox21 multi-task toxicity benchmark.

Usage:
    conda activate drug-tox-env
    python scripts/train_attentivefp_tox21.py \
        --config config/tox21_attentivefp_config.yaml --device cuda
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
from backend.graph_data import smiles_to_pyg_data, get_feature_dims
from backend.attentivefp_model import create_attentivefp_model
from backend.graph_train import evaluate_multitask_model, MaskedFocalLoss
from backend.utils import ensure_dir, save_metrics, set_seed


class GraphModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


def create_multitask_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    labels = np.asarray(labels, dtype=np.float32)
    any_positive = (np.nanmax(labels, axis=1) > 0).astype(int)
    class_counts = np.bincount(any_positive, minlength=2).astype(np.float32)
    sample_weights = np.ones_like(any_positive, dtype=np.float32)
    for c in range(2):
        if class_counts[c] > 0:
            sample_weights[any_positive == c] = 1.0 / (class_counts[c] * 2.0)
    sample_weights = sample_weights / max(float(sample_weights.sum()), 1e-8) * len(sample_weights)
    return WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)


def prepare_dataset(df, task_names):
    labels_array = df[task_names].values
    dataset = []
    for i, smi in enumerate(df["smiles"]):
        data = smiles_to_pyg_data(smi, label=labels_array[i])
        if data is not None:
            dataset.append(data)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/tox21_attentivefp_config.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config_path = project_root / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    dc = config["data"]
    oc = config.get("output", {})

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    set_seed(dc.get("seed", 42))
    task_names = get_task_names("tox21")

    print("=" * 70)
    print("AttentiveFP — Tox21 Multi-Task Toxicity Prediction")
    print("=" * 70)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_tox21(
        cache_dir=str(project_root / dc["cache_dir"]),
        split_type=dc["split_type"],
        seed=dc.get("seed", 42),
        enforce_workspace_mode=False,
    )
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_pyg = prepare_dataset(train_df, task_names)
    val_pyg = prepare_dataset(val_df, task_names)
    test_pyg = prepare_dataset(test_df, task_names)
    print(f"Graphs: train={len(train_pyg)}, val={len(val_pyg)}, test={len(test_pyg)}")

    batch_size = int(tc["batch_size"])
    train_sampler = None
    if tc.get("use_weighted_sampler", False):
        labels_arr = np.array([d.y.numpy().squeeze(0) for d in train_pyg])
        train_sampler = create_multitask_sampler(labels_arr)

    train_loader = DataLoader(train_pyg, batch_size=batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None))
    val_loader = DataLoader(val_pyg, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_pyg, batch_size=batch_size * 2, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    node_feat_dim, edge_feat_dim = get_feature_dims()
    backbone = create_attentivefp_model(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_channels=int(mc["hidden_channels"]),
        num_layers=int(mc["num_layers"]),
        num_timesteps=int(mc["num_timesteps"]),
        dropout=float(mc["dropout"]),
        num_tasks=len(task_names),
    )
    model = GraphModelWrapper(backbone).to(device)
    print(f"Parameters: {sum(p.numel() for p in backbone.parameters()):,}")

    # ── Training ──────────────────────────────────────────────────────────
    criterion = MaskedFocalLoss(
        alpha=float(tc.get("focal_alpha", 0.25)),
        gamma=float(tc.get("focal_gamma", 2.0)),
    )
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(tc["learning_rate"]),
                                 weight_decay=float(tc["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    num_epochs = int(tc["num_epochs"])
    patience = int(tc["early_stopping_patience"])
    history: Dict[str, List[float]] = {"train_loss": [], "val_auc": [], "val_pr_auc": []}
    best_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.y.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)

            optimizer.zero_grad()
            logits = model(batch)
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else float("nan")

        val_metrics = evaluate_multitask_model(
            model=model, data_loader=val_loader,
            task_names=task_names, device=device,
        )
        val_auc = float(val_metrics.get("macro_auc_roc", 0.0))
        val_pr = float(val_metrics.get("macro_pr_auc", 0.0))

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d}/{num_epochs} loss={train_loss:.4f} "
                  f"val_auc={val_auc:.4f} pr={val_pr:.4f} best={best_auc:.4f} "
                  f"p={no_improve}/{patience}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # ── Evaluate & Save ───────────────────────────────────────────────────
    test_metrics = evaluate_multitask_model(
        model=model, data_loader=test_loader,
        task_names=task_names, device=device,
    )

    out_dir = project_root / oc.get("model_dir", "models/tox21_attentivefp_model")
    ensure_dir(str(out_dir))
    torch.save(backbone.state_dict(), out_dir / "best_model.pt")
    shutil.copy(config_path, out_dir / "config.yaml")

    flat = {
        "test_mean_auc_roc": float(test_metrics.get("macro_auc_roc", 0.0)),
        "test_mean_pr_auc": float(test_metrics.get("macro_pr_auc", 0.0)),
    }
    for tn, tm in (test_metrics.get("task_metrics", {}) or {}).items():
        flat[f"test_auc_{tn}"] = float(tm.get("auc_roc", 0.0))
    save_metrics(flat, str(out_dir / "metrics.txt"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"]); ax[0].set_title("Loss")
    ax[1].plot(history["val_auc"], label="AUC"); ax[1].plot(history["val_pr_auc"], label="PR", ls="--")
    ax[1].set_title("Val Metrics"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out_dir / "training_curves.png", dpi=120); plt.close()

    print(f"\nTest AUC-ROC: {flat['test_mean_auc_roc']:.4f}  PR-AUC: {flat['test_mean_pr_auc']:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
