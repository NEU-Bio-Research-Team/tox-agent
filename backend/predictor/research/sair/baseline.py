"""Reproducible descriptor baseline for the SAIR Phase-1 pair table.

This is a B0 scientific benchmark, not a production model.  It deliberately
uses inexpensive 2D ligand descriptors and protein composition so that data,
split and metric plumbing can be exercised before investing in a learned
ligand--protein encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
FEATURE_NAMES = (
    "mol_wt",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "fraction_csp3",
    "heavy_atom_count",
    "formal_charge",
    "protein_length_log1p",
    *tuple(f"aa_fraction_{aa}" for aa in AMINO_ACIDS),
)


class BaselineInputError(ValueError):
    pass


def featurize_pair(smiles: str, protein_sequence: str) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise BaselineInputError(f"invalid SMILES: {smiles!r}")
    sequence = protein_sequence.strip().upper()
    if not sequence:
        raise BaselineInputError("protein sequence is required")
    counts = {aa: sequence.count(aa) for aa in AMINO_ACIDS}
    known = sum(counts.values())
    if known == 0:
        raise BaselineInputError("protein sequence has no recognised amino acids")
    composition = [counts[aa] / known for aa in AMINO_ACIDS]
    return np.asarray(
        [
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.RingCount(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule),
            molecule.GetNumHeavyAtoms(),
            sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
            math.log1p(len(sequence)),
            *composition,
        ],
        dtype=np.float32,
    )


def featurize_many(smiles: Iterable[str], sequences: Iterable[str]) -> np.ndarray:
    rows = [featurize_pair(smi, seq) for smi, seq in zip(smiles, sequences)]
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack(rows)


def deterministic_sample(frame, *, limit: int, salt: str):
    """Take the lowest stable hashes instead of a row-order-dependent head."""
    if limit <= 0 or len(frame) <= limit:
        return frame.copy()
    keys = frame["entry_id"].map(
        lambda value: int.from_bytes(
            hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).digest()[:8],
            "big",
        )
    )
    return frame.loc[keys.nsmallest(limit).index].sort_values("entry_id").copy()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def train_baseline(
    *,
    phase1_path: Path,
    phase1_manifest_path: Path,
    artifact_path: Path,
    report_path: Path,
    max_train: int,
    max_validation: int,
    max_test: int,
    sample_salt: str,
) -> dict[str, Any]:
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor

    manifest = json.loads(phase1_manifest_path.read_text())
    observed_sha = _file_sha256(phase1_path)
    expected_sha = manifest["output"]["sha256"]
    if observed_sha != expected_sha:
        raise BaselineInputError(
            f"Phase-1 checksum mismatch: expected {expected_sha}, got {observed_sha}"
        )
    columns = [
        "entry_id",
        "source_smiles",
        "protein_sequence",
        "pic50_median",
        "training_eligible",
        "split",
    ]
    frame = pd.read_parquet(phase1_path, columns=columns)
    frame = frame[frame["training_eligible"]]
    limits = {
        "train": max_train,
        "validation": max_validation,
        "test": max_test,
    }
    sampled = {
        split: deterministic_sample(
            frame[frame["split"] == split],
            limit=limit,
            salt=f"{sample_salt}:{split}",
        )
        for split, limit in limits.items()
    }
    if any(part.empty for part in sampled.values()):
        empty = sorted(name for name, part in sampled.items() if part.empty)
        raise BaselineInputError(f"empty benchmark partitions: {empty}")

    features = {
        name: featurize_many(part["source_smiles"], part["protein_sequence"])
        for name, part in sampled.items()
    }
    labels = {
        name: part["pic50_median"].to_numpy(dtype=np.float64)
        for name, part in sampled.items()
    }
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=17,
    )
    model.fit(features["train"], labels["train"])
    predictions = {name: model.predict(matrix) for name, matrix in features.items()}
    calibration_residuals = np.abs(
        labels["validation"] - predictions["validation"]
    )
    interval_half_width = float(
        np.quantile(calibration_residuals, 0.9, method="higher")
    )
    metrics = {
        name: regression_metrics(labels[name], predictions[name])
        for name in sampled
    }
    metrics["test"]["interval_90_coverage"] = float(
        np.mean(np.abs(labels["test"] - predictions["test"]) <= interval_half_width)
    )
    lower = np.quantile(features["train"], 0.005, axis=0).astype(float)
    upper = np.quantile(features["train"], 0.995, axis=0).astype(float)
    model_id = "sair-descriptor-hgb-b0"
    artifact = {
        "artifact_version": "toxagent-activity-baseline-v1",
        "model_id": model_id,
        "admitted": False,
        "admission_reason": "B0 descriptor benchmark; not a production activity model",
        "model": model,
        "feature_names": FEATURE_NAMES,
        "feature_lower_005": lower,
        "feature_upper_995": upper,
        "interval_90_half_width": interval_half_width,
        "dataset": {
            "phase1_sha256": observed_sha,
            "source_sha256": manifest["source"]["sha256"],
            "split_policy": manifest["split_policy"],
        },
        "sample": {
            "salt": sample_salt,
            "counts": {name: len(part) for name, part in sampled.items()},
        },
        "metrics": metrics,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)
    artifact_sha = _file_sha256(artifact_path)
    report = {
        key: value for key, value in artifact.items() if key != "model"
    }
    report["feature_lower_005"] = lower.tolist()
    report["feature_upper_995"] = upper.tolist()
    report["artifact"] = {
        "path": str(artifact_path),
        "sha256": artifact_sha,
        "bytes": artifact_path.stat().st_size,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def predict_with_baseline(
    artifact: dict[str, Any], *, smiles: str, protein_sequence: str
) -> dict[str, Any]:
    vector = featurize_pair(smiles, protein_sequence)
    prediction = float(artifact["model"].predict(vector.reshape(1, -1))[0])
    lower = np.asarray(artifact["feature_lower_005"])
    upper = np.asarray(artifact["feature_upper_995"])
    outside = (vector < lower) | (vector > upper)
    outside_fraction = float(np.mean(outside))
    if outside_fraction == 0:
        applicability = "in_domain"
    elif outside_fraction <= 0.2:
        applicability = "limited"
    else:
        applicability = "out_of_domain"
    half_width = float(artifact["interval_90_half_width"])
    return {
        "predicted_pic50": prediction,
        "uncertainty": half_width,
        "prediction_interval": [prediction - half_width, prediction + half_width],
        "applicability": applicability,
        "applicability_reasons": [
            f"{int(outside.sum())}/{len(vector)} descriptor values outside training 0.5-99.5% bounds"
        ],
        "model_id": artifact["model_id"],
        "admitted": bool(artifact["admitted"]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--phase1-manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=100_000)
    parser.add_argument("--max-validation", type=int, default=25_000)
    parser.add_argument("--max-test", type=int, default=25_000)
    parser.add_argument("--sample-salt", default="toxagent-sair-b0-v1")
    args = parser.parse_args(argv)
    report = train_baseline(
        phase1_path=args.phase1,
        phase1_manifest_path=args.phase1_manifest,
        artifact_path=args.artifact,
        report_path=args.report,
        max_train=args.max_train,
        max_validation=args.max_validation,
        max_test=args.max_test,
        sample_salt=args.sample_salt,
    )
    print(args.artifact)
    print(args.report)
    print(json.dumps(report["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
