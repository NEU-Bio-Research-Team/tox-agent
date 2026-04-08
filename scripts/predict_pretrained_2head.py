#!/usr/bin/env python3
"""Inference utility for pretrained dual-head model (hERG + Tox21)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.data import get_task_names
from backend.pretrained_mol_model import create_pretrained_dual_head_model, get_checkpoint_defaults


def _load_task_thresholds(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None

    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("task_thresholds"), dict):
        threshold_map = payload.get("task_thresholds", {})
    elif isinstance(payload, dict):
        threshold_map = payload
    else:
        return None

    out: Dict[str, float] = {}
    for key, value in threshold_map.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    return out or None


def _load_model_artifacts(model_dir: Path, device: str):
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint format invalid. Expected dict with model_state_dict.")

    pretrained_model = ckpt.get("pretrained_model")
    if not pretrained_model:
        raise ValueError("Checkpoint missing pretrained_model metadata")

    model_cfg = ckpt.get("model_config", {}) if isinstance(ckpt.get("model_config"), dict) else {}

    model = create_pretrained_dual_head_model(
        pretrained_model=str(pretrained_model),
        num_tox21_tasks=int(model_cfg.get("num_tox21_tasks", len(ckpt.get("task_names", []) or get_task_names("tox21")))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        herg_hidden_dim=model_cfg.get("herg_hidden_dim"),
        use_herg_mlp=bool(model_cfg.get("use_herg_mlp", True)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer_dir = model_dir / "tokenizer"
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

    defaults = get_checkpoint_defaults(str(pretrained_model))
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        trust_remote_code=bool(defaults["trust_remote_code"]),
    )

    task_names = ckpt.get("task_names") or get_task_names("tox21")
    max_length = int(ckpt.get("max_length", defaults["max_length"]))
    herg_threshold = float(ckpt.get("herg_threshold", 0.5))
    tox21_thresholds = ckpt.get("tox21_thresholds") if isinstance(ckpt.get("tox21_thresholds"), dict) else None

    return model, tokenizer, list(task_names), max_length, herg_threshold, tox21_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with pretrained dual-head model")
    parser.add_argument("--smiles", type=str, required=True, help="Input SMILES")
    parser.add_argument("--name", type=str, default="Mol-000", help="Optional molecule name")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/pretrained_2head_herg_chemberta_model",
        help="Directory containing best_model.pt and tokenizer/",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--herg-threshold", type=float, default=None)
    parser.add_argument("--task-threshold", type=float, default=0.5)
    parser.add_argument(
        "--task-thresholds-path",
        type=str,
        default=None,
        help="Optional JSON with per-task thresholds",
    )
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU")
        device = "cpu"

    model_dir = project_root / args.model_dir
    model, tokenizer, task_names, max_length, default_herg_threshold, ckpt_task_thresholds = _load_model_artifacts(
        model_dir=model_dir,
        device=device,
    )

    task_thresholds = ckpt_task_thresholds
    if args.task_thresholds_path:
        file_thresholds = _load_task_thresholds(project_root / args.task_thresholds_path)
        if file_thresholds is not None:
            task_thresholds = file_thresholds

    herg_threshold = float(args.herg_threshold) if args.herg_threshold is not None else float(default_herg_threshold)

    enc = tokenizer(
        [str(args.smiles)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        heads = model.forward_heads(input_ids=input_ids, attention_mask=attention_mask)

    herg_logit = float(heads["herg_logits"].reshape(-1)[0].cpu().item())
    herg_prob = float(1.0 / (1.0 + np.exp(-herg_logit)))
    herg_is_toxic = bool(herg_prob >= herg_threshold)

    tox_logits = heads["tox21_logits"].reshape(-1).cpu().numpy().astype(np.float32)
    tox_probs = (1.0 / (1.0 + np.exp(-tox_logits))).astype(np.float32)

    task_scores: Dict[str, float] = {}
    toxic_flags: Dict[str, bool] = {}
    per_task_thresholds: Dict[str, float] = {}

    for idx, task_name in enumerate(task_names):
        score = float(tox_probs[idx])
        if task_thresholds and task_name in task_thresholds:
            thr = float(task_thresholds[task_name])
        else:
            thr = float(args.task_threshold)
        task_scores[task_name] = score
        toxic_flags[task_name] = bool(score >= thr)
        per_task_thresholds[task_name] = thr

    result = {
        "name": str(args.name),
        "smiles": str(args.smiles),
        "source": "pretrained_dual_head",
        "backbone": str(model_dir),
        "herg": {
            "label": "TOXIC" if herg_is_toxic else "NON_TOXIC",
            "is_toxic": herg_is_toxic,
            "p_toxic": herg_prob,
            "threshold_used": herg_threshold,
            "logit": herg_logit,
        },
        "tox21": {
            "task_scores": task_scores,
            "toxic_flags": toxic_flags,
            "task_thresholds": per_task_thresholds,
            "num_flagged_tasks": int(sum(1 for v in toxic_flags.values() if v)),
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
