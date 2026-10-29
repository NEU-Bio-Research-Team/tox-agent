"""Inspect a gated SAIR parquet file without loading its 3D structure corpus.

The repository does not bundle SAIR.  This command is the first safe operation
after a user has independently accepted the dataset terms and downloaded
``sair.parquet``.  It records physical schema and file identity, maps only a
small set of known aliases to logical fields, and fails closed when the minimum
Phase-1 contract cannot be identified.

Usage::

    PYTHONPATH=research python -m sair.inspect_schema \
        --dataset /path/to/sair.parquet --out /path/to/sair-schema-report.json

``pyarrow`` is intentionally imported only by the CLI so the normal predictor
service does not acquire a research-only parquet dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_REPORT_VERSION = "toxagent-sair-schema-report-v1"

LOGICAL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "entry_id": ("entry_id", "entryid"),
    "smiles": ("SMILES", "smiles", "canonical_smiles", "ligand_smiles"),
    "protein_id": ("protein", "protein_id", "target_id", "uniprot_id"),
    "protein_sequence": ("sequence", "protein_sequence", "target_sequence"),
    "pic50": ("pIC50", "pic50", "p_ic50"),
    "assay": ("assay", "assay_type"),
    "protein_family": ("family", "protein_family"),
}

REQUIRED_PHASE1_FIELDS = frozenset(
    {"entry_id", "smiles", "protein_id", "protein_sequence", "pic50"}
)

KNOWN_STRUCTURE_FIELDS = frozenset(
    {
        "aevplig_score",
        "onionnet_score",
        "vina_score_min",
        "iptm",
        "interaction_ptm",
        "complex_iplddt",
        "all_passed",
        "internal_energy",
    }
)


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_logical_columns(columns: Iterable[str]) -> dict[str, Any]:
    """Map physical names without silently choosing between ambiguous aliases."""
    physical = list(columns)
    by_folded: dict[str, list[str]] = {}
    for name in physical:
        by_folded.setdefault(name.casefold(), []).append(name)

    resolved: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for logical, aliases in LOGICAL_COLUMN_ALIASES.items():
        candidates: list[str] = []
        for alias in aliases:
            candidates.extend(by_folded.get(alias.casefold(), []))
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            resolved[logical] = candidates[0]
        elif len(candidates) > 1:
            ambiguous[logical] = candidates

    missing = sorted(REQUIRED_PHASE1_FIELDS - set(resolved) - set(ambiguous))
    return {
        "resolved": resolved,
        "ambiguous": ambiguous,
        "missing_required": missing,
        "phase1_compatible": not missing and not ambiguous,
    }


def build_schema_report(
    *,
    dataset_path: Path,
    columns: Mapping[str, str],
    row_count: int,
    row_groups: int,
    parquet_created_by: str | None,
    sha256: str | None,
    source_uri: str,
    dataset_version: str,
    license_id: str,
) -> dict[str, Any]:
    mapping = resolve_logical_columns(columns)
    folded = {name.casefold(): name for name in columns}
    structure_fields = sorted(
        folded[name] for name in KNOWN_STRUCTURE_FIELDS if name in folded
    )
    return {
        "schema_version": SCHEMA_REPORT_VERSION,
        "dataset": {
            "path": str(dataset_path),
            "bytes": dataset_path.stat().st_size,
            "sha256": sha256,
            "source_uri": source_uri,
            "dataset_version": dataset_version,
            "license": license_id,
        },
        "parquet": {
            "rows": int(row_count),
            "row_groups": int(row_groups),
            "created_by": parquet_created_by,
            "columns": [
                {"name": name, "physical_type": physical_type}
                for name, physical_type in columns.items()
            ],
        },
        "logical_mapping": mapping,
        "known_structure_fields_present": structure_fields,
        "notes": [
            "schema inspection does not validate label semantics or licensing",
            "3D structure archives are not required for the Phase-1 contract",
            "a compatible schema is necessary but not sufficient for model training",
        ],
    }


def inspect_parquet(
    path: Path,
    *,
    compute_sha256: bool,
    source_uri: str,
    dataset_version: str,
    license_id: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"SAIR parquet not found: {path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise RuntimeError(
            "SAIR schema inspection requires pyarrow; install it in the research "
            "environment, not the production predictor image"
        ) from exc

    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    columns = {
        field.name: str(field.type)
        for field in parquet.schema_arrow
    }
    return build_schema_report(
        dataset_path=path,
        columns=columns,
        row_count=metadata.num_rows,
        row_groups=metadata.num_row_groups,
        parquet_created_by=metadata.created_by,
        sha256=file_sha256(path) if compute_sha256 else None,
        source_uri=source_uri,
        dataset_version=dataset_version,
        license_id=license_id,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-uri", default="https://huggingface.co/datasets/SandboxAQ/SAIR")
    parser.add_argument("--dataset-version", default="unverified")
    parser.add_argument("--license", dest="license_id", default="CC-BY-4.0")
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="skip the potentially expensive file hash during exploratory inspection",
    )
    args = parser.parse_args(argv)

    report = inspect_parquet(
        args.dataset,
        compute_sha256=not args.skip_sha256,
        source_uri=args.source_uri,
        dataset_version=args.dataset_version,
        license_id=args.license_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.out)
    return 0 if report["logical_mapping"]["phase1_compatible"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
