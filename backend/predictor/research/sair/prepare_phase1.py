"""Build a pair-level SAIR Phase-1 table without using the 3D archives.

The source parquet repeats activity measurements across generated structures.
This ETL first removes exact measurement duplicates, then aggregates each
protein--ligand pair.  Label disagreement is retained as data quality metadata;
high-disagreement pairs are excluded from training, never silently averaged in.

The initial split is an exact-sequence cold-target split.  It prevents the same
protein sequence from appearing in multiple partitions, but does not claim to
be a protein-family or sequence-cluster holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PHASE1_COLUMNS = (
    "entry_id",
    "protein",
    "sequence",
    "SMILES",
    "source",
    "description",
    "potency",
    "assay",
    "pIC50",
    "family",
)

IDENTITY_COLUMNS = ("protein", "sequence", "SMILES", "family")
MEASUREMENT_COLUMNS = (
    "entry_id",
    "source",
    "description",
    "potency",
    "assay",
    "pIC50",
)

REQUIRED_NON_NULL_COLUMNS = tuple(
    name for name in PHASE1_COLUMNS if name != "description"
)


class Phase1DataError(ValueError):
    pass


def _require_dependencies():
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "SAIR Phase-1 ETL requires the predictor 'sair' optional dependencies"
        ) from exc
    return pd, pq


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cold_target_partition(
    sequence: str,
    *,
    salt: str,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> str:
    if not sequence:
        raise Phase1DataError("protein sequence is required for cold-target splitting")
    if not salt.strip():
        raise ValueError("split salt is required")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie in (0, 1)")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test partition")
    digest = hashlib.sha256(f"{salt}\0{sequence}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + validation_fraction:
        return "validation"
    return "test"


def aggregate_pairs(
    frame,
    *,
    max_label_span: float,
    split_salt: str,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
):
    """Collapse structure repetitions while preserving label disagreement."""
    pd, _ = _require_dependencies()
    if not math.isfinite(max_label_span) or max_label_span < 0:
        raise ValueError("max_label_span must be a finite non-negative number")
    missing = sorted(set(PHASE1_COLUMNS) - set(frame.columns))
    if missing:
        raise Phase1DataError(f"SAIR input is missing Phase-1 columns: {missing}")
    if frame.empty:
        raise Phase1DataError("SAIR input contains no rows")

    required = frame[list(PHASE1_COLUMNS)]
    nulls = {
        name: int(required[name].isna().sum()) for name in REQUIRED_NON_NULL_COLUMNS
    }
    nonzero_nulls = {name: count for name, count in nulls.items() if count}
    if nonzero_nulls:
        raise Phase1DataError(f"SAIR Phase-1 columns contain nulls: {nonzero_nulls}")
    if not required["pIC50"].map(math.isfinite).all():
        raise Phase1DataError("SAIR pIC50 contains non-finite values")

    identities = required[["entry_id", *IDENTITY_COLUMNS]].drop_duplicates()
    conflicting_ids = identities.loc[
        identities["entry_id"].duplicated(keep=False), "entry_id"
    ].unique()
    if len(conflicting_ids):
        preview = sorted(map(str, conflicting_ids[:10]))
        raise Phase1DataError(
            f"entry_id maps to conflicting protein/ligand identity: {preview}"
        )

    measurements = required[list(MEASUREMENT_COLUMNS)].drop_duplicates()
    grouped = measurements.groupby("entry_id", sort=False, observed=True)
    labels = grouped["pIC50"].agg(
        measurement_count="count",
        pic50_median="median",
        pic50_mean="mean",
        pic50_std="std",
        pic50_min="min",
        pic50_max="max",
    )
    context = grouped.agg(
        source_primary=("source", "first"),
        source_count=("source", "nunique"),
        assay_primary=("assay", "first"),
        assay_count=("assay", "nunique"),
    )
    labels["pic50_std"] = labels["pic50_std"].fillna(0.0)
    labels["pic50_span"] = labels["pic50_max"] - labels["pic50_min"]

    pairs = identities.set_index("entry_id").join(labels).join(context).reset_index()
    pairs = pairs.rename(
        columns={
            "protein": "target_id",
            "sequence": "protein_sequence",
            "SMILES": "source_smiles",
            "family": "protein_family",
        }
    )
    pairs["label_quality"] = pd.Series(
        ["accepted"] * len(pairs), index=pairs.index, dtype="string"
    )
    high_conflict = pairs["pic50_span"] > max_label_span
    pairs.loc[high_conflict, "label_quality"] = "high_measurement_disagreement"
    pairs["training_eligible"] = ~high_conflict

    sequence_to_split = {
        sequence: cold_target_partition(
            sequence,
            salt=split_salt,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        for sequence in pairs["protein_sequence"].unique()
    }
    pairs["split"] = pairs["protein_sequence"].map(sequence_to_split)
    return pairs.sort_values("entry_id").reset_index(drop=True), measurements


def validate_cold_target_split(pairs) -> dict[str, Any]:
    sequence_splits = pairs.groupby("protein_sequence", observed=True)["split"].nunique()
    crossing = sequence_splits[sequence_splits > 1]
    if len(crossing):
        raise Phase1DataError(
            f"cold-target leakage: {len(crossing)} protein sequences cross partitions"
        )
    eligible = pairs[pairs["training_eligible"]]
    return {
        "rows": int(len(pairs)),
        "training_eligible_rows": int(len(eligible)),
        "excluded_rows": int(len(pairs) - len(eligible)),
        "unique_targets": int(pairs["target_id"].nunique()),
        "unique_sequences": int(pairs["protein_sequence"].nunique()),
        "unique_ligands": int(pairs["source_smiles"].nunique()),
        "split_rows": {
            str(name): int(count)
            for name, count in eligible["split"].value_counts().sort_index().items()
        },
        "split_sequences": {
            str(name): int(count)
            for name, count in (
                eligible[["protein_sequence", "split"]]
                .drop_duplicates()["split"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "sequence_crossings": 0,
    }


def prepare_phase1(
    *,
    source: Path,
    output: Path,
    manifest_path: Path,
    dataset_revision: str,
    dataset_sha256: str,
    max_label_span: float,
    split_salt: str,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, Any]:
    _, pq = _require_dependencies()
    observed_sha256 = file_sha256(source)
    if observed_sha256 != dataset_sha256.lower():
        raise Phase1DataError(
            f"dataset checksum mismatch: expected {dataset_sha256}, got {observed_sha256}"
        )

    frame = pq.read_table(source, columns=list(PHASE1_COLUMNS)).to_pandas()
    description_nulls = int(frame["description"].isna().sum())
    pairs, measurements = aggregate_pairs(
        frame,
        max_label_span=max_label_span,
        split_salt=split_salt,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    audit = validate_cold_target_split(pairs)

    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output, index=False, compression="zstd")
    output_sha256 = file_sha256(output)
    manifest = {
        "manifest_version": "toxagent-sair-phase1-v1",
        "source": {
            "path": str(source),
            "revision": dataset_revision,
            "sha256": observed_sha256,
            "rows": int(len(frame)),
            "unique_measurements": int(len(measurements)),
            "optional_description_null_rows": description_nulls,
        },
        "output": {
            "path": str(output),
            "sha256": output_sha256,
            "bytes": output.stat().st_size,
        },
        "label_policy": {
            "value": "median_pic50_over_exactly_deduplicated_measurements",
            "max_label_span": max_label_span,
            "excluded_status": "high_measurement_disagreement",
        },
        "split_policy": {
            "strategy": "exact_sequence_cold_target_hash_v1",
            "salt": split_salt,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": round(1 - train_fraction - validation_fraction, 12),
            "limitation": "not a sequence-cluster or protein-family holdout",
        },
        "audit": audit,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--max-label-span", type=float, required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    manifest = prepare_phase1(
        source=args.source,
        output=args.output,
        manifest_path=args.manifest,
        dataset_revision=args.dataset_revision,
        dataset_sha256=args.dataset_sha256,
        max_label_span=args.max_label_span,
        split_salt=args.split_salt,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    print(args.output)
    print(args.manifest)
    print(json.dumps(manifest["audit"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
