#!/usr/bin/env python3
"""Run inference with multi-task XSmiles heads.

Examples:
    python scripts/predict_xsmiles_multitask.py --smiles "CCO" --mode both
    python scripts/predict_xsmiles_multitask.py --smiles "CCO" --mode profile
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.inference import (
    load_model,
    predict_clinical_toxicity,
    predict_xsmiles_toxicity_profile,
)
from backend.workspace_mode import assert_clintox_enabled


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


def main() -> None:
    assert_clintox_enabled("scripts/predict_xsmiles_multitask.py")

    parser = argparse.ArgumentParser(description="Predict with multi-task XSmiles")
    parser.add_argument("--smiles", type=str, required=True, help="Input SMILES")
    parser.add_argument("--name", type=str, default="Mol-000", help="Optional molecule name")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/smilesgnn_multitask_model",
        help="Model directory containing best_model.pt + tokenizer.pkl",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/xsmiles_multitask_config.yaml",
        help="Model config path",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--clinical-threshold", type=float, default=0.35)
    parser.add_argument("--task-threshold", type=float, default=0.5)
    parser.add_argument(
        "--task-thresholds-path",
        type=str,
        default=None,
        help="Optional JSON path for per-task thresholds (defaults to model_dir/tox21_task_thresholds.json)",
    )
    parser.add_argument("--mode", choices=["clinical", "profile", "both"], default="both")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU")
        device = "cpu"

    model_dir = project_root / args.model_dir
    config_path = project_root / args.config
    threshold_file = (
        project_root / args.task_thresholds_path
        if args.task_thresholds_path
        else model_dir / "tox21_task_thresholds.json"
    )
    task_thresholds = _load_task_thresholds(threshold_file)

    model, tokenizer, wrapped_model = load_model(
        model_dir=model_dir,
        config_path=config_path,
        device=device,
    )

    result = {}

    if args.mode in {"clinical", "both"}:
        result["clinical"] = predict_clinical_toxicity(
            smiles=str(args.smiles),
            tokenizer=tokenizer,
            wrapped_model=wrapped_model,
            device=device,
            threshold=float(args.clinical_threshold),
            name=str(args.name),
            enforce_workspace_mode=False,
        )

    if args.mode in {"profile", "both"}:
        result["toxicity_profile"] = predict_xsmiles_toxicity_profile(
            smiles=str(args.smiles),
            tokenizer=tokenizer,
            model=model,
            device=device,
            threshold=float(args.task_threshold),
            task_thresholds=task_thresholds,
        )
        result["toxicity_profile"]["task_thresholds_source"] = (
            str(threshold_file) if task_thresholds is not None else "default_scalar_threshold"
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
