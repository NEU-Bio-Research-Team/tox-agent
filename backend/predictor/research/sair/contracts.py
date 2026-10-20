"""Dataset-independent contracts for preparing a future SAIR benchmark.

These types do not encode assumptions about the physical SAIR parquet schema.
The schema inspector must first resolve source columns explicitly; only then
may an ETL step create canonical rows and split assignments like these.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SplitStrategy(str, Enum):
    LIGAND_GROUP_HOLDOUT = "ligand_group_holdout"
    TARGET_GROUP_HOLDOUT = "target_group_holdout"
    PAIR_GROUP_HOLDOUT = "pair_group_holdout"


@dataclass(frozen=True, slots=True)
class CanonicalActivityRow:
    """Minimum Phase-1 regression row after source-specific normalization."""

    entry_id: str
    canonical_smiles: str
    target_id: str
    protein_sequence: str
    pic50: float
    assay_context: str

    def __post_init__(self) -> None:
        required = {
            "entry_id": self.entry_id,
            "canonical_smiles": self.canonical_smiles,
            "target_id": self.target_id,
            "protein_sequence": self.protein_sequence,
            "assay_context": self.assay_context,
        }
        empty = sorted(name for name, value in required.items() if not value.strip())
        if empty:
            raise ValueError(f"canonical activity row has empty fields: {empty}")
        value = float(self.pic50)
        if not math.isfinite(value):
            raise ValueError("pic50 must be finite")
        object.__setattr__(self, "pic50", value)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    entry_id: str
    split: SplitName
    ligand_group: str
    target_group: str
    replicate_group: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SplitName(self.split))
        if not self.entry_id.strip():
            raise ValueError("entry_id is required")
        if not self.ligand_group.strip() or not self.target_group.strip():
            raise ValueError("ligand_group and target_group are required")
        if self.replicate_group is not None and not self.replicate_group.strip():
            raise ValueError("replicate_group must be non-empty when provided")


class UnsafeSplitError(ValueError):
    pass


def _cross_split_groups(
    assignments: list[SplitAssignment],
    *,
    key_name: str,
) -> list[dict[str, Any]]:
    seen: dict[Any, set[str]] = {}
    for assignment in assignments:
        if key_name == "pair_group":
            key: Any = (assignment.ligand_group, assignment.target_group)
        else:
            key = getattr(assignment, key_name)
        if key is not None:
            seen.setdefault(key, set()).add(assignment.split.value)
    return [
        {"group": key, "splits": sorted(splits)}
        for key, splits in sorted(seen.items(), key=lambda item: str(item[0]))
        if len(splits) > 1
    ]


def audit_split_assignments(
    assignments: Iterable[SplitAssignment],
    *,
    strategy: SplitStrategy,
) -> dict[str, Any]:
    """Return a deterministic leakage report; never silently repairs a split."""
    strategy = SplitStrategy(strategy)
    rows = list(assignments)
    entry_counts: dict[str, int] = {}
    split_counts = {name.value: 0 for name in SplitName}
    for assignment in rows:
        entry_counts[assignment.entry_id] = entry_counts.get(assignment.entry_id, 0) + 1
        split_counts[assignment.split.value] += 1

    duplicate_entries = sorted(
        entry_id for entry_id, count in entry_counts.items() if count > 1
    )
    grouping_key = {
        SplitStrategy.LIGAND_GROUP_HOLDOUT: "ligand_group",
        SplitStrategy.TARGET_GROUP_HOLDOUT: "target_group",
        SplitStrategy.PAIR_GROUP_HOLDOUT: "pair_group",
    }[strategy]
    holdout_leaks = _cross_split_groups(rows, key_name=grouping_key)
    replicate_leaks = _cross_split_groups(rows, key_name="replicate_group")
    violations = {
        "duplicate_entry_ids": duplicate_entries,
        "holdout_group_crossings": holdout_leaks,
        "replicate_group_crossings": replicate_leaks,
    }
    return {
        "strategy": strategy.value,
        "rows": len(rows),
        "split_counts": split_counts,
        "safe": bool(rows) and not any(violations.values()),
        "violations": violations,
    }


def build_split_manifest(
    assignments: Iterable[SplitAssignment],
    *,
    strategy: SplitStrategy,
    dataset_sha256: str,
    grouping_method: str,
    seed: int | None,
) -> dict[str, Any]:
    """Build an immutable manifest only after all declared leakage guards pass."""
    if len(dataset_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in dataset_sha256.lower()
    ):
        raise ValueError("dataset_sha256 must be a 64-character hexadecimal digest")
    if not grouping_method.strip():
        raise ValueError("grouping_method is required")

    rows = list(assignments)
    audit = audit_split_assignments(rows, strategy=strategy)
    if not audit["safe"]:
        raise UnsafeSplitError(f"split failed leakage audit: {audit['violations']}")

    canonical_assignments = [
        {
            "entry_id": row.entry_id,
            "split": row.split.value,
            "ligand_group": row.ligand_group,
            "target_group": row.target_group,
            "replicate_group": row.replicate_group,
        }
        for row in sorted(rows, key=lambda item: item.entry_id)
    ]
    assignment_bytes = json.dumps(
        canonical_assignments,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "manifest_version": "toxagent-sair-split-v1",
        "dataset_sha256": dataset_sha256.lower(),
        "strategy": strategy.value,
        "grouping_method": grouping_method,
        "seed": seed,
        "assignment_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        "split_counts": audit["split_counts"],
        "assignments": canonical_assignments,
    }
