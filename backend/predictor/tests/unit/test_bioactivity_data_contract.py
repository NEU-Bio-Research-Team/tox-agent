"""Invariants of the bioactivity data contract and split design.

These guard the failures that would make the benchmark look better than the
model is: leaked compounds, collapsed measurement contexts, corrupted parent
structures, and metrics that emit a number where they have no data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PREDICTOR = Path(__file__).resolve().parents[2]
for candidate in (PREDICTOR / "research", PREDICTOR / "evals"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

pytest.importorskip("rdkit", reason="bioactivity pipeline needs RDKit")

from bioactivity.ingest.build_dataset import assay_context_key  # noqa: E402
from bioactivity.ingest.split import (  # noqa: E402
    FRACTIONS,
    SPLIT_NAMES,
    cluster_split,
    group_rows,
    random_split,
    scaffold_split,
    temporal_split,
)
from bioactivity.ingest.standardize import standardize_smiles  # noqa: E402
from bioactivity.metrics import (  # noqa: E402
    binary_metrics,
    cliff_metrics,
    macro_summary,
    regression_metrics,
)


# --------------------------------------------------------------------------
# standardization
# --------------------------------------------------------------------------

def test_salt_is_stripped_to_its_organic_parent():
    """Sodium aspirinate must standardize to aspirin, not stay a mixture."""
    result = standardize_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
    assert result.status == "ok"
    assert result.standardized_smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert result.had_salt_counterion is True


def test_coordination_complex_keeps_its_metal_and_is_flagged():
    """A metal complex must never be 'repaired' into one of its ligands.

    Largest-fragment choice on cisplatin picks ammonia, which both loses the
    platinum and yields a structure the assay never measured.
    """
    result = standardize_smiles("N.N.Cl[Pt]Cl")
    assert result.status == "flagged"
    assert result.has_coordination_metal is True
    assert "Pt" in (result.standardized_smiles or "")
    assert result.standardized_smiles != "N"


def test_enantiomers_share_connectivity_key_but_differ_as_structures():
    """The split grouping unit must not separate stereoisomers."""
    left = standardize_smiles("C[C@H](N)C(=O)O")
    right = standardize_smiles("C[C@@H](N)C(=O)O")
    assert left.connectivity_key == right.connectivity_key
    assert left.standardized_smiles != right.standardized_smiles


def test_unparseable_smiles_is_rejected_with_a_reason():
    result = standardize_smiles("not_a_smiles")
    assert result.status == "rejected"
    assert result.reason == "unparseable"
    assert result.standardized_smiles is None


# --------------------------------------------------------------------------
# measurement context
# --------------------------------------------------------------------------

def test_assay_context_separates_binding_from_functional():
    """Two formats for one target must not share a context key."""
    binding = {"assay_type": "B", "bao_label": "single protein format"}
    functional = {"assay_type": "F", "bao_label": "cell-based format"}
    assert assay_context_key(binding) != assay_context_key(functional)


def test_assay_context_separates_variant_from_wild_type():
    wild = {"assay_type": "B", "bao_label": "single protein format"}
    mutant = dict(wild, assay_variant_mutation="T790M")
    assert assay_context_key(wild) != assay_context_key(mutant)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def _table(n_compounds: int = 240, n_tasks: int = 4) -> list[dict[str, object]]:
    """A synthetic table with the fields the splitters read."""
    rows = []
    for index in range(n_compounds):
        # Distinct but valid connectivity keys and scaffolds.
        key = f"KEY{index:012d}"
        for task in range(n_tasks):
            rows.append(
                {
                    "standardized_smiles": "C" * (1 + index % 12) + "O",
                    "connectivity_key": key,
                    "murcko_scaffold": f"scaffold{index % 20}",
                    "target_chembl_id": f"CHEMBL{task}",
                    "standard_type": "IC50",
                    "assay_context_key": "B|single protein format|WT",
                    "task_key": f"CHEMBL{task}|IC50|B|single protein format|WT",
                    "pactivity": 5.0 + (index % 40) / 10.0,
                    "document_year_min": 2000 + index % 24,
                    "high_disagreement": "0",
                }
            )
    return rows


@pytest.mark.parametrize("builder", [temporal_split, scaffold_split, cluster_split, random_split])
def test_no_compound_appears_in_two_splits(builder):
    """The leakage invariant, asserted structurally for every view."""
    table = _table()
    groups = group_rows(table)
    assignment, _ = builder(table, groups)

    seen: dict[str, str] = {}
    for split, keys in assignment.items():
        for key in keys:
            assert key not in seen, f"{key} in both {seen.get(key)} and {split}"
            seen[key] = split
    assert set(seen) == set(groups), "every compound must be assigned exactly once"


@pytest.mark.parametrize("builder", [temporal_split, scaffold_split, cluster_split, random_split])
def test_every_split_is_populated(builder):
    """A view with an empty test split silently scores nothing.

    This is the regression test for the greedy packer that let a few large
    scaffold blocks overshoot and starve the tail splits.
    """
    table = _table()
    groups = group_rows(table)
    assignment, meta = builder(table, groups)
    for name in SPLIT_NAMES:
        assert meta["counts"][name]["rows"] > 0, f"{name} split is empty"


def test_grouping_is_global_across_targets():
    """One compound's rows must land in one split for the whole panel."""
    table = _table()
    groups = group_rows(table)
    assignment, _ = temporal_split(table, groups)

    split_of = {key: split for split, keys in assignment.items() for key in keys}
    for row in table:
        # Every row of a compound resolves through the same key, so a compound
        # cannot train at one target and test at another.
        assert row["connectivity_key"] in split_of


def test_temporal_split_is_ordered_in_time():
    """Train must not contain chemistry newer than test."""
    table = _table()
    groups = group_rows(table)
    _, meta = temporal_split(table, groups)
    boundaries = meta["year_boundaries"]
    assert boundaries["train"]["year_max"] <= boundaries["test"]["year_min"]


def test_split_fractions_are_approximately_respected():
    table = _table()
    groups = group_rows(table)
    _, meta = temporal_split(table, groups)
    total = sum(meta["counts"][name]["rows"] for name in SPLIT_NAMES)
    for name in SPLIT_NAMES:
        actual = meta["counts"][name]["rows"] / total
        assert abs(actual - FRACTIONS[name]) < 0.08, f"{name}: {actual:.3f}"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_metrics_are_undefined_rather_than_invented():
    """A task with no label variance must not report an R2."""
    constant = np.full(50, 7.0)
    metrics = regression_metrics(constant, np.random.default_rng(0).normal(7, 1, 50))
    assert metrics["r2"].value is None
    assert metrics["r2"].reason == "constant_labels"
    # MAE is still well defined on constant labels.
    assert metrics["mae"].value is not None


def test_single_class_binary_view_reports_no_auroc():
    y_true = np.full(60, 5.0)  # nothing reaches the 7.0 threshold
    metrics = binary_metrics(y_true, np.linspace(0, 1, 60), threshold=7.0)
    assert metrics["pr_auc"].value is None
    assert "single_class" in metrics["pr_auc"].reason


def test_macro_summary_names_the_tasks_it_could_not_score():
    per_task = {
        "good": regression_metrics(np.linspace(4, 9, 40), np.linspace(4, 9, 40)),
        "flat": regression_metrics(np.full(40, 6.0), np.full(40, 6.0)),
    }
    summary = macro_summary(per_task, "r2")
    assert "flat" in summary["undefined_tasks"]
    assert summary["n_tasks"] == 1


def test_cliff_direction_accuracy_catches_inverted_pairs():
    """A model can have a fine RMSE and still order every cliff backwards."""
    y_true = np.array([5.0, 8.0, 6.0, 9.0])
    inverted = np.array([8.0, 5.0, 9.0, 6.0])
    metrics = cliff_metrics(y_true, inverted, [(0, 1), (2, 3)])
    assert metrics["direction_accuracy"].value == 0.0

    metrics_ok = cliff_metrics(y_true, y_true, [(0, 1), (2, 3)])
    assert metrics_ok["direction_accuracy"].value == 1.0


def test_cliff_metrics_report_absence_of_pairs():
    metrics = cliff_metrics(np.array([5.0, 6.0]), np.array([5.0, 6.0]), [])
    assert metrics["delta_mae"].value is None
    assert metrics["delta_mae"].reason == "no_cliff_pairs"
