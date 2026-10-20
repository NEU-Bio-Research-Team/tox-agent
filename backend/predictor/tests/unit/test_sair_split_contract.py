from __future__ import annotations

import sys
from pathlib import Path

import pytest


PREDICTOR = Path(__file__).resolve().parents[2]
if str(PREDICTOR / "research") not in sys.path:
    sys.path.insert(0, str(PREDICTOR / "research"))

from sair.contracts import (  # noqa: E402
    CanonicalActivityRow,
    SplitAssignment,
    SplitName,
    SplitStrategy,
    UnsafeSplitError,
    audit_split_assignments,
    build_split_manifest,
)


def _assignment(
    entry_id: str,
    split: SplitName,
    ligand: str,
    target: str,
    replicate: str | None = None,
) -> SplitAssignment:
    return SplitAssignment(entry_id, split, ligand, target, replicate)


def test_canonical_activity_row_keeps_pic50_as_finite_regression_value():
    row = CanonicalActivityRow("e1", "CCO", "EGFR", "MPEPTIDE", 7.25, "IC50")
    assert row.pic50 == 7.25
    with pytest.raises(ValueError, match="pic50 must be finite"):
        CanonicalActivityRow("e2", "CCO", "EGFR", "MPEPTIDE", float("nan"), "IC50")


def test_ligand_holdout_detects_a_ligand_group_crossing_partitions():
    audit = audit_split_assignments(
        [
            _assignment("e1", SplitName.TRAIN, "scaffold-a", "target-a"),
            _assignment("e2", SplitName.TEST, "scaffold-a", "target-b"),
        ],
        strategy=SplitStrategy.LIGAND_GROUP_HOLDOUT,
    )
    assert audit["safe"] is False
    assert audit["violations"]["holdout_group_crossings"] == [
        {"group": "scaffold-a", "splits": ["test", "train"]}
    ]


def test_target_holdout_allows_shared_ligand_but_not_shared_target_group():
    safe = audit_split_assignments(
        [
            _assignment("e1", SplitName.TRAIN, "scaffold-a", "cluster-a"),
            _assignment("e2", SplitName.TEST, "scaffold-a", "cluster-b"),
        ],
        strategy=SplitStrategy.TARGET_GROUP_HOLDOUT,
    )
    assert safe["safe"] is True

    unsafe = audit_split_assignments(
        [
            _assignment("e1", SplitName.TRAIN, "scaffold-a", "cluster-a"),
            _assignment("e2", SplitName.TEST, "scaffold-b", "cluster-a"),
        ],
        strategy=SplitStrategy.TARGET_GROUP_HOLDOUT,
    )
    assert unsafe["safe"] is False


def test_replicates_never_cross_partitions_regardless_of_holdout_strategy():
    audit = audit_split_assignments(
        [
            _assignment("e1", SplitName.TRAIN, "l1", "t1", "replicate-a"),
            _assignment("e2", SplitName.VALIDATION, "l2", "t2", "replicate-a"),
        ],
        strategy=SplitStrategy.PAIR_GROUP_HOLDOUT,
    )
    assert audit["safe"] is False
    assert audit["violations"]["replicate_group_crossings"]


def test_split_manifest_is_deterministic_and_refuses_unsafe_assignments():
    rows = [
        _assignment("e2", SplitName.TEST, "l2", "t2"),
        _assignment("e1", SplitName.TRAIN, "l1", "t1"),
    ]
    first = build_split_manifest(
        rows,
        strategy=SplitStrategy.PAIR_GROUP_HOLDOUT,
        dataset_sha256="a" * 64,
        grouping_method="synthetic-test-groups",
        seed=17,
    )
    second = build_split_manifest(
        reversed(rows),
        strategy=SplitStrategy.PAIR_GROUP_HOLDOUT,
        dataset_sha256="a" * 64,
        grouping_method="synthetic-test-groups",
        seed=17,
    )
    assert first == second
    assert [row["entry_id"] for row in first["assignments"]] == ["e1", "e2"]

    with pytest.raises(UnsafeSplitError, match="failed leakage audit"):
        build_split_manifest(
            [
                _assignment("e1", SplitName.TRAIN, "l1", "t1"),
                _assignment("e1", SplitName.TEST, "l2", "t2"),
            ],
            strategy=SplitStrategy.PAIR_GROUP_HOLDOUT,
            dataset_sha256="a" * 64,
            grouping_method="synthetic-test-groups",
            seed=17,
        )
