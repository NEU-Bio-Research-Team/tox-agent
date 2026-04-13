#!/usr/bin/env python3
"""
Fine-tune MoLFormer-XL (ibm/MoLFormer-XL-both-10pct) on Tox21.

Same architecture as ChemBERTa script but uses MoLFormer-XL checkpoint
with trust_remote_code=True and pooler_output.

Usage:
    conda activate drug-tox-env
    python scripts/train_molformer_tox21.py \
        --config config/tox21_molformer_config.yaml --device cuda
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.data import get_task_names, load_tox21
from backend.pretrained_mol_model import create_pretrained_mol_model
from backend.graph_train import MaskedFocalLoss
from backend.utils import ensure_dir, save_metrics, set_seed


class Tox21SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=202):
        self.encodings = tokenizer(
            smiles_list, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def create_multitask_sampler(labels):
    any_positive = (np.nanmax(labels, axis=1) > 0).astype(int)
    cc = np.bincount(any_positive, minlength=2).astype(np.float32)
    sw = np.ones_like(any_positive, dtype=np.float32)
    for c in range(2):
        if cc[c] > 0:
            sw[any_positive == c] = 1.0 / (cc[c] * 2.0)
    sw = sw / max(float(sw.sum()), 1e-8) * len(sw)
    return WeightedRandomSampler(sw.tolist(), num_samples=len(sw), replacement=True)


def evaluate_smiles_model(model, loader, device, criterion, task_names):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"])

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))

    per_task_auc = {}
    for t, name in enumerate(task_names):
        valid = ~np.isnan(all_labels[:, t])
        if valid.sum() < 2 or len(np.unique(all_labels[valid, t])) < 2:
            continue
        per_task_auc[name] = roc_auc_score(all_labels[valid, t], all_probs[valid, t])

    return {
        "mean_auc_roc": float(np.mean(list(per_task_auc.values()))) if per_task_auc else 0.0,
        "per_task_auc_roc": per_task_auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/tox21_molformer_config.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config_path = project_root / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc, tc, dc = config["model"], config["training"], config["data"]
    oc = config.get("output", {})

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    set_seed(dc.get("seed", 42))
    task_names = get_task_names("tox21")

    print("=" * 70)
    print(f"MoLFormer-XL — Tox21 ({mc['pretrained_model']})")
    print("=" * 70)

    train_df, val_df, test_df = load_tox21(
        cache_dir=str(project_root / dc["cache_dir"]),
        split_type=dc["split_type"], seed=dc.get("seed", 42),
        enforce_workspace_mode=False,
    )

    train_labels = train_df[task_names].values.astype(np.float32)
    val_labels = val_df[task_names].values.astype(np.float32)
    test_labels = test_df[task_names].values.astype(np.float32)

    print(f"Loading tokenizer: {mc['pretrained_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(mc["pretrained_model"], trust_remote_code=True)
    max_length = int(tc.get("max_length", 202))

    train_ds = Tox21SMILESDataset(train_df["smiles"].tolist(), train_labels, tokenizer, max_length)
    val_ds = Tox21SMILESDataset(val_df["smiles"].tolist(), val_labels, tokenizer, max_length)
    test_ds = Tox21SMILESDataset(test_df["smiles"].tolist(), test_labels, tokenizer, max_length)

    batch_size = int(tc["batch_size"])
    train_sampler = create_multitask_sampler(train_labels) if tc.get("use_weighted_sampler") else None
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Loading model: {mc['pretrained_model']}...")
    model = create_pretrained_mol_model(
        pretrained_model=mc["pretrained_model"],
        num_tasks=len(task_names),
        dropout=float(mc["dropout"]),
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": float(tc["weight_decay"])},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=float(tc["learning_rate"]))

    num_epochs = int(tc["num_epochs"])
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * float(tc.get("warmup_ratio", 0.1)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = MaskedFocalLoss(alpha=float(tc.get("focal_alpha", 0.25)), gamma=float(tc.get("focal_gamma", 2.0)))
    grad_clip = float(tc.get("grad_clip", 1.0))
    patience = int(tc["early_stopping_patience"])

    history: Dict[str, List[float]] = {"train_loss": [], "val_auc": []}
    best_auc, best_state, no_improve = 0.0, None, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_metrics = evaluate_smiles_model(model, val_loader, device, criterion, task_names)
        val_auc = val_metrics["mean_auc_roc"]

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d}/{num_epochs} loss={train_loss:.4f} "
                  f"val_auc={val_auc:.4f} best={best_auc:.4f} p={no_improve}/{patience}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_smiles_model(model, test_loader, device, criterion, task_names)

    out_dir = project_root / oc.get("model_dir", "models/tox21_molformer_model")
    ensure_dir(str(out_dir))
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    shutil.copy(config_path, out_dir / "config.yaml")

    flat = {"test_mean_auc_roc": test_metrics["mean_auc_roc"]}
    for tn, auc in test_metrics["per_task_auc_roc"].items():
        flat[f"test_auc_{tn}"] = auc
    save_metrics(flat, str(out_dir / "metrics.txt"))

    print(f"\nTest AUC-ROC: {test_metrics['mean_auc_roc']:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
