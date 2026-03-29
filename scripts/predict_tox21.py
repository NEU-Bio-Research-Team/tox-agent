#!/usr/bin/env python3
"""Run Tox21 multi-task prediction for one or many SMILES strings.

Examples:
    python scripts/predict_tox21.py --smiles "CCO" --device cuda
    python scripts/predict_tox21.py --input-file test_data/smiles_only.csv --device cuda
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.inference import load_tox21_gatv2_model, predict_tox21_batch
from src.workspace_mode import assert_tox21_enabled


def _parse_txt_input(path: Path) -> Tuple[List[str], List[str]]:
    smiles = []
    names = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            smiles.append(parts[0].strip())
            names.append(parts[1].strip() if len(parts) > 1 else f"Mol-{i:03d}")
    return smiles, names


def _parse_tabular_input(path: Path) -> Tuple[List[str], List[str]]:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    smiles_col = next((c for c in df.columns if c.lower() == "smiles"), None)
    if smiles_col is None:
        raise ValueError("Input file must contain a SMILES column named 'smiles'.")

    name_col = next(
        (c for c in df.columns if c.lower() in {"name", "compound", "id", "compound_id"}),
        None,
    )

    smiles = df[smiles_col].astype(str).tolist()
    if name_col is None:
        names = [f"Mol-{i:03d}" for i in range(len(smiles))]
    else:
        names = df[name_col].astype(str).tolist()

    return smiles, names


def _load_inputs(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    if args.smiles:
        return [args.smiles], [args.name or "Mol-000"]

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".txt":
        return _parse_txt_input(input_path)
    if suffix in {".csv", ".xlsx"}:
        return _parse_tabular_input(input_path)

    raise ValueError("Unsupported input format. Use .txt, .csv, or .xlsx")


def main() -> None:
    assert_tox21_enabled("scripts/predict_tox21.py")

    parser = argparse.ArgumentParser(
        description="Predict Tox21 assay probabilities from SMILES"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", type=str, help="Single SMILES string")
    source.add_argument("--input-file", type=str, help="Path to .txt/.csv/.xlsx file")

    parser.add_argument("--name", type=str, default=None, help="Optional name for --smiles mode")
    parser.add_argument("--model-dir", type=str, default="models/tox21_gatv2_model")
    parser.add_argument("--config", type=str, default="config/tox21_gatv2_config.yaml")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--threshold", type=float, default=0.5, help="Default assay threshold")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20, help="Rows to print in console")
    parser.add_argument(
        "--output",
        type=str,
        default="results/tox21_predictions.csv",
        help="CSV path for predictions",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    smiles, names = _load_inputs(args)
    print(f"Loaded {len(smiles)} molecule(s)")

    model_dir = project_root / args.model_dir
    config_path = project_root / args.config
    model, task_names = load_tox21_gatv2_model(model_dir=model_dir, config_path=config_path, device=device)

    predictions = predict_tox21_batch(
        smiles_list=smiles,
        model=model,
        task_names=task_names,
        device=device,
        names=names,
        threshold=float(args.threshold),
        batch_size=int(args.batch_size),
    )

    parse_errors = int((predictions["AssayHits"] == -1).sum())
    alert_hits = int((predictions["MechanisticAlert"] == True).sum())

    print("\nSummary")
    print("=" * 72)
    print(f"Total molecules: {len(predictions)}")
    print(f"Mechanistic alerts: {alert_hits}")
    print(f"Parse errors: {parse_errors}")

    show_cols = [
        "Name",
        "SMILES",
        "AssayHits",
        "HitTasks",
        "MaxAssay",
        "MaxAssayProb",
    ]
    top_k = max(1, int(args.top_k))
    print("\nTop predictions")
    print("=" * 72)
    print(predictions[show_cols].head(top_k).to_string(index=False))

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")


if __name__ == "__main__":
    main()
