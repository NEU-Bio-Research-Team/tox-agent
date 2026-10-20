from __future__ import annotations

import sys
from pathlib import Path


PREDICTOR = Path(__file__).resolve().parents[2]
if str(PREDICTOR / "research") not in sys.path:
    sys.path.insert(0, str(PREDICTOR / "research"))

from sair.inspect_schema import build_schema_report, resolve_logical_columns  # noqa: E402


def test_officially_described_phase1_columns_resolve_without_3d_fields():
    result = resolve_logical_columns(
        ["entry_id", "SMILES", "protein", "sequence", "pIC50", "assay", "family"]
    )
    assert result["phase1_compatible"] is True
    assert result["resolved"]["smiles"] == "SMILES"
    assert result["resolved"]["pic50"] == "pIC50"
    assert result["missing_required"] == []


def test_missing_protein_sequence_fails_the_phase1_contract():
    result = resolve_logical_columns(
        ["entry_id", "SMILES", "protein", "pIC50"]
    )
    assert result["phase1_compatible"] is False
    assert result["missing_required"] == ["protein_sequence"]


def test_alias_collision_is_reported_instead_of_silently_chosen():
    result = resolve_logical_columns(
        ["entry_id", "SMILES", "canonical_smiles", "protein", "sequence", "pIC50"]
    )
    assert result["phase1_compatible"] is False
    assert result["ambiguous"]["smiles"] == ["SMILES", "canonical_smiles"]


def test_schema_report_records_identity_and_keeps_3d_optional(tmp_path):
    dataset = tmp_path / "sair.parquet"
    dataset.write_bytes(b"fixture bytes, not a parquet")
    report = build_schema_report(
        dataset_path=dataset,
        columns={
            "entry_id": "int64",
            "SMILES": "string",
            "protein": "string",
            "sequence": "string",
            "pIC50": "double",
            "iptm": "double",
        },
        row_count=10,
        row_groups=1,
        parquet_created_by="synthetic-test",
        sha256="a" * 64,
        source_uri="https://huggingface.co/datasets/SandboxAQ/SAIR",
        dataset_version="synthetic-test-only",
        license_id="CC-BY-4.0",
    )
    assert report["logical_mapping"]["phase1_compatible"] is True
    assert report["known_structure_fields_present"] == ["iptm"]
    assert report["dataset"]["sha256"] == "a" * 64
    assert report["parquet"]["rows"] == 10
