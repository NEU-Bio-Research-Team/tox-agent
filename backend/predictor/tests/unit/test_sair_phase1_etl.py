from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PREDICTOR = Path(__file__).resolve().parents[2]
if str(PREDICTOR / "research") not in sys.path:
    sys.path.insert(0, str(PREDICTOR / "research"))

from sair.prepare_phase1 import (  # noqa: E402
    Phase1DataError,
    aggregate_pairs,
    cold_target_partition,
    validate_cold_target_split,
)


def _row(entry_id: int, **changes):
    row = {
        "entry_id": entry_id,
        "protein": f"P{entry_id}",
        "sequence": f"SEQUENCE{entry_id}",
        "SMILES": "CCO",
        "source": "ChEMBL",
        "description": "synthetic assay",
        "potency": 7.0,
        "assay": "biochem",
        "pIC50": 7.0,
        "family": "enzyme",
    }
    row.update(changes)
    return row


def test_exact_structure_repetitions_do_not_inflate_measurement_count():
    frame = pd.DataFrame([_row(1), _row(1), _row(1)])
    pairs, measurements = aggregate_pairs(
        frame, max_label_span=1.0, split_salt="test"
    )
    assert len(measurements) == 1
    assert pairs.loc[0, "measurement_count"] == 1
    assert pairs.loc[0, "pic50_median"] == 7.0
    assert bool(pairs.loc[0, "training_eligible"]) is True


def test_optional_description_may_be_missing():
    frame = pd.DataFrame([_row(1, description=None)])
    pairs, measurements = aggregate_pairs(
        frame, max_label_span=1.0, split_salt="test"
    )
    assert len(pairs) == 1
    assert len(measurements) == 1


def test_measurement_disagreement_is_retained_and_excluded_by_policy():
    frame = pd.DataFrame(
        [
            _row(1, description="assay-a", pIC50=5.0, potency=5.0),
            _row(1, description="assay-b", pIC50=8.0, potency=8.0),
        ]
    )
    pairs, _ = aggregate_pairs(frame, max_label_span=1.0, split_salt="test")
    assert pairs.loc[0, "pic50_median"] == 6.5
    assert pairs.loc[0, "pic50_span"] == 3.0
    assert pairs.loc[0, "label_quality"] == "high_measurement_disagreement"
    assert bool(pairs.loc[0, "training_eligible"]) is False


def test_entry_identity_conflict_fails_closed():
    frame = pd.DataFrame([_row(1), _row(1, protein="DIFFERENT")])
    with pytest.raises(Phase1DataError, match="conflicting protein/ligand identity"):
        aggregate_pairs(frame, max_label_span=1.0, split_salt="test")


def test_exact_sequence_is_never_split_across_partitions():
    frame = pd.DataFrame(
        [
            _row(1, sequence="SHARED", protein="P1"),
            _row(2, sequence="SHARED", protein="P1", SMILES="CCC"),
            _row(3, sequence="OTHER", protein="P2", SMILES="CCN"),
        ]
    )
    pairs, _ = aggregate_pairs(frame, max_label_span=1.0, split_salt="test")
    assert pairs.loc[pairs.protein_sequence == "SHARED", "split"].nunique() == 1
    assert validate_cold_target_split(pairs)["sequence_crossings"] == 0


def test_partition_is_deterministic_and_validates_ratios():
    assert cold_target_partition("MPEPTIDE", salt="v1") == cold_target_partition(
        "MPEPTIDE", salt="v1"
    )
    with pytest.raises(ValueError, match="leave a test partition"):
        cold_target_partition(
            "MPEPTIDE",
            salt="v1",
            train_fraction=0.9,
            validation_fraction=0.1,
        )
