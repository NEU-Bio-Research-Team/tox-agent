"""Freeze the four split views and prove they do not leak.

Views (data contract, section 4.1):

  temporal  PRIMARY -- closest to prospective use; release decisions use this
  cluster   chemical-cluster OOD -- generalization to new chemistry
  scaffold  Bemis-Murcko -- comparability with the usual QSAR literature
  random    diagnostic upper bound only; never a release criterion

Two invariants hold across every view:

  * Grouping is GLOBAL. A compound's records go to one split across the whole
    panel, not per target. Otherwise a multi-task encoder trains on a structure
    at target A and is tested on it at target B, and every cold-chemistry number
    is optimistic.
  * Grouping is by stereo-insensitive connectivity key, so a compound's
    enantiomer cannot sit in test while the compound itself trains.

Each view yields train / validation / calibration / test. Calibration is held
separately from validation because conformal intervals fitted on data used for
early stopping are not valid.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from bioactivity.ingest.task_keys import task_unit_key  # noqa: E402

LOG = logging.getLogger("split")

SPLIT_NAMES = ("train", "validation", "calibration", "test")

#: Row-count fractions. Temporal uses these as time quantiles; the grouped
#: views use them as targets for greedy group packing.
#:
#: Calibration gets a full 10% rather than the 5% a conformal fit would strictly
#: need. The contract asks for >= `MIN_TASK_TEST_ROWS` calibration rows PER TASK,
#: and a 5% slice of the panel only clears that for tasks with ~3,000 rows --
#: which left two thirds of the panel unscorable. Widening calibration and test
#: costs training rows but is what makes per-task calibration measurable.
FRACTIONS = {"train": 0.60, "validation": 0.10, "calibration": 0.10, "test": 0.20}

SEED = 42

#: Per-task floors from the data contract, section 3.5. A task below these is
#: kept in the dataset (it still carries multi-task training signal) but is
#: excluded from per-task metric reporting, because a metric over 30 test rows
#: is noise presented as a measurement.
MIN_TASK_TEST_ROWS = 150
MIN_TASK_CALIBRATION_ROWS = 150
MIN_TASK_TRAIN_ROWS = 500


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_table(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def compound_year(rows: Iterable[dict[str, Any]]) -> int | None:
    """Earliest publication year over a compound's records.

    A compound that appears across many years belongs to its FIRST appearance:
    a later measurement of an already-known structure is not a new compound, and
    treating it as one lets the temporal test set contain old chemistry.
    """
    years = [
        int(r["document_year_min"])
        for r in rows
        if str(r.get("document_year_min") or "").strip().isdigit()
    ]
    return min(years) if years else None


def group_rows(table: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Row indices per connectivity key -- the global grouping unit."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(table):
        groups[row["connectivity_key"]].append(index)
    return dict(groups)


def _pack_sequential(
    ordered_groups: list[tuple[str, list[int]]], total_rows: int
) -> dict[str, list[str]]:
    """Assign whole groups to splits IN ORDER, for views where order is meaning.

    Used by the temporal view, where the group order is chronological and must
    be preserved: train has to be the earliest chemistry and test the latest.

    Quotas are compared against a running total using cumulative boundaries.
    Tracking fill per split instead lets one oversized group push the cursor
    forward by only a single split, which starves the tail splits -- that bug
    produced an empty scaffold test set.
    """
    assignment: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    order = list(SPLIT_NAMES)

    boundaries: list[float] = []
    cumulative = 0.0
    for name in order:
        cumulative += FRACTIONS[name] * total_rows
        boundaries.append(cumulative)

    current = 0
    running = 0.0
    for key, indices in ordered_groups:
        # The last split is the overflow bucket, so no group is ever dropped.
        while current < len(order) - 1 and running >= boundaries[current]:
            current += 1
        assignment[order[current]].append(key)
        running += len(indices)

    return assignment


def _pack_balanced(
    blocks: list[tuple[str, list[str], int]], total_rows: int
) -> dict[str, list[str]]:
    """Assign whole blocks to whichever split is furthest below its quota.

    Used by the scaffold, cluster and random views, where block order carries no
    meaning. Sequential packing there is fragile: a few very large scaffold or
    cluster blocks overshoot one split and leave later splits empty. Filling the
    largest remaining deficit keeps every split populated while still assigning
    each block wholly to one split.

    `blocks` is (block_id, member group keys, row count).
    """
    assignment: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    quotas = {name: FRACTIONS[name] * total_rows for name in SPLIT_NAMES}
    filled = {name: 0.0 for name in SPLIT_NAMES}

    # Largest blocks first: placing them while all splits still have room avoids
    # a single huge block having to land in an almost-full split.
    for _, keys, rows in sorted(blocks, key=lambda b: (-b[2], b[0])):
        # Deficit as a fraction of quota, so small splits are not starved by
        # absolute-size comparisons. Name breaks ties deterministically.
        target = min(
            SPLIT_NAMES,
            key=lambda name: (
                (filled[name] - quotas[name]) / max(quotas[name], 1.0),
                name,
            ),
        )
        assignment[target].extend(keys)
        filled[target] += rows

    return assignment


def _report(
    name: str,
    assignment: dict[str, list[str]],
    groups: dict[str, list[int]],
    table: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-split counts plus per-task test coverage, and leakage assertions."""
    seen: dict[str, str] = {}
    for split, keys in assignment.items():
        for key in keys:
            if key in seen:
                raise AssertionError(
                    f"{name}: compound {key} in both {seen[key]} and {split}"
                )
            seen[key] = split

    unassigned = set(groups) - set(seen)
    if unassigned:
        raise AssertionError(f"{name}: {len(unassigned)} compounds unassigned")

    counts = {}
    for split, keys in assignment.items():
        rows = sum(len(groups[k]) for k in keys)
        counts[split] = {"compounds": len(keys), "rows": rows}

    # Per-task test/calibration counts drive the panel floor check: a task with
    # 20 test rows cannot support a per-task metric, however good the aggregate.
    per_task: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split, keys in assignment.items():
        for key in keys:
            for index in groups[key]:
                # Same key the models and the eligibility gate use.
                task = task_unit_key(table[index])
                per_task[task][split] += 1

    return {
        "counts": counts,
        "per_task": {task: dict(splits) for task, splits in sorted(per_task.items())},
    }


# --------------------------------------------------------------------------
# the four views
# --------------------------------------------------------------------------

def temporal_split(
    table: list[dict[str, Any]], groups: dict[str, list[int]]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Chronological by each compound's earliest publication year.

    Year boundaries are chosen as row quantiles rather than fixed calendar years
    so that every split is populated; the resolved boundaries are recorded in
    the manifest.
    """
    years: dict[str, int] = {}
    undated: list[str] = []
    for key, indices in groups.items():
        year = compound_year(table[i] for i in indices)
        if year is None:
            undated.append(key)
        else:
            years[key] = year

    # Undated compounds cannot be placed in time. They go to train: putting them
    # in test would silently weaken the prospective claim.
    ordered = sorted(years.items(), key=lambda kv: (kv[1], kv[0]))
    ordered_groups = [(key, groups[key]) for key, _ in ordered]
    total_rows = sum(len(groups[k]) for k, _ in ordered)

    assignment = _pack_sequential(ordered_groups, total_rows)
    assignment["train"].extend(undated)

    boundaries: dict[str, dict[str, int]] = {}
    for split, keys in assignment.items():
        dated = [years[k] for k in keys if k in years]
        if dated:
            boundaries[split] = {"year_min": min(dated), "year_max": max(dated)}

    meta = _report("temporal", assignment, groups, table)
    meta["year_boundaries"] = boundaries
    meta["undated_compounds_to_train"] = len(undated)
    return assignment, meta


def _ecfp_matrix(smiles_list: list[str], n_bits: int = 2048) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    matrix = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for row, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fingerprint = generator.GetFingerprintAsNumPy(mol)
        matrix[row] = fingerprint
    return matrix


def cluster_split(
    table: list[dict[str, Any]], groups: dict[str, list[int]], seed: int = SEED
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Chemical-cluster OOD split on ECFP4.

    Butina clustering needs the full pairwise similarity matrix, which is
    quadratic and does not fit for a panel of this size. MiniBatchKMeans over
    folded ECFP4 scales linearly and still produces contiguous regions of
    chemical space, which is what the OOD view needs: whole neighbourhoods held
    out, not scattered molecules. The approximation is recorded in the manifest
    so the view is not mistaken for literature-standard Butina.
    """
    from sklearn.cluster import MiniBatchKMeans

    keys = sorted(groups)
    representative = {}
    for key in keys:
        representative[key] = table[groups[key][0]]["standardized_smiles"]

    LOG.info("computing ECFP4 for %d compounds", len(keys))
    matrix = _ecfp_matrix([representative[k] for k in keys])

    n_clusters = max(8, len(keys) // 50)
    LOG.info("MiniBatchKMeans into %d clusters", n_clusters)
    model = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, batch_size=1024, n_init=3
    )
    labels = model.fit_predict(matrix.astype(np.float32))

    by_cluster: dict[int, list[str]] = defaultdict(list)
    for key, label in zip(keys, labels):
        by_cluster[int(label)].append(key)

    # Pack at cluster granularity: a cluster must not straddle two splits, or
    # the OOD view degenerates into a random one.
    total_rows = sum(len(v) for v in groups.values())
    blocks = [
        (
            f"cluster-{cluster_id:05d}",
            sorted(by_cluster[cluster_id]),
            sum(len(groups[k]) for k in by_cluster[cluster_id]),
        )
        for cluster_id in sorted(by_cluster)
    ]
    assignment = _pack_balanced(blocks, total_rows)

    meta = _report("cluster", assignment, groups, table)
    meta["n_clusters"] = n_clusters
    meta["clustering"] = "MiniBatchKMeans on ECFP4 (radius 2, 2048 bits)"
    meta["clustering_is_approximation_of"] = "Butina (infeasible at this scale)"
    return assignment, meta


def scaffold_split(
    table: list[dict[str, Any]], groups: dict[str, list[int]]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Bemis-Murcko scaffold split, largest scaffold groups into train."""
    by_scaffold: dict[str, list[str]] = defaultdict(list)
    for key, indices in groups.items():
        scaffold = table[indices[0]]["murcko_scaffold"] or "__acyclic__"
        by_scaffold[scaffold].append(key)

    total_rows = sum(len(v) for v in groups.values())
    blocks = [
        (
            scaffold,
            sorted(by_scaffold[scaffold]),
            sum(len(groups[k]) for k in by_scaffold[scaffold]),
        )
        for scaffold in sorted(by_scaffold)
    ]
    assignment = _pack_balanced(blocks, total_rows)

    meta = _report("scaffold", assignment, groups, table)
    meta["n_scaffolds"] = len(by_scaffold)
    return assignment, meta


def random_split(
    table: list[dict[str, Any]], groups: dict[str, list[int]], seed: int = SEED
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Diagnostic only. Still grouped by compound, so it is not row-random."""
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    ordered_groups = [(key, groups[key]) for key in keys]
    total_rows = sum(len(v) for v in groups.values())
    assignment = _pack_sequential(ordered_groups, total_rows)
    meta = _report("random", assignment, groups, table)
    meta["role"] = "diagnostic upper bound; not a release criterion"
    return assignment, meta


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

BUILDERS = {
    "temporal": temporal_split,
    "cluster": cluster_split,
    "scaffold": scaffold_split,
    "random": random_split,
}


def task_eligibility(meta: dict[str, Any]) -> dict[str, Any]:
    """Which tasks have enough held-out data to carry a per-task metric.

    Applied to the PRIMARY (temporal) view, because that is the view release
    decisions use. The rule is a fixed sample floor -- it never consults a model
    score -- and both the eligible and ineligible lists are published, so the
    reported macro cannot quietly become an average over whichever tasks a model
    happened to do well on.
    """
    eligible: list[str] = []
    ineligible: dict[str, dict[str, Any]] = {}

    for task, counts in meta["per_task"].items():
        train = counts.get("train", 0)
        calibration = counts.get("calibration", 0)
        test = counts.get("test", 0)
        failures = []
        if train < MIN_TASK_TRAIN_ROWS:
            failures.append(f"train={train}<{MIN_TASK_TRAIN_ROWS}")
        if calibration < MIN_TASK_CALIBRATION_ROWS:
            failures.append(f"calibration={calibration}<{MIN_TASK_CALIBRATION_ROWS}")
        if test < MIN_TASK_TEST_ROWS:
            failures.append(f"test={test}<{MIN_TASK_TEST_ROWS}")

        if failures:
            ineligible[task] = {
                "reasons": failures,
                "counts": {"train": train, "calibration": calibration, "test": test},
            }
        else:
            eligible.append(task)

    return {
        "floors": {
            "min_train_rows": MIN_TASK_TRAIN_ROWS,
            "min_calibration_rows": MIN_TASK_CALIBRATION_ROWS,
            "min_test_rows": MIN_TASK_TEST_ROWS,
        },
        "basis_view": "temporal",
        "n_eligible": len(eligible),
        "n_ineligible": len(ineligible),
        "eligible_tasks": sorted(eligible),
        "ineligible_tasks": ineligible,
        "note": "ineligible tasks stay in the dataset for training signal but "
        "are excluded from per-task and macro metric reporting",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--include-high-disagreement",
        action="store_true",
        help="keep replicate-disagreement rows (excluded from HQ-Exact by default)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    table = load_table(args.dataset)
    LOG.info("loaded %d aggregated rows", len(table))
    if not args.include_high_disagreement:
        before = len(table)
        table = [r for r in table if r["high_disagreement"] != "1"]
        LOG.info("excluded %d high-disagreement rows", before - len(table))

    groups = group_rows(table)
    LOG.info("%d unique compounds (connectivity keys)", len(groups))

    manifest: dict[str, Any] = {
        "split_manifest_id": "toxact-chembl37-hq-v1-splits",
        "dataset": str(args.dataset.name),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "seed": args.seed,
        "fractions": FRACTIONS,
        "grouping": "global, by stereo-insensitive InChIKey connectivity block",
        "primary_view": "temporal",
        "n_rows": len(table),
        "n_compounds": len(groups),
        "views": {},
    }

    for name, builder in BUILDERS.items():
        LOG.info("building %s split", name)
        if name in ("cluster", "random"):
            assignment, meta = builder(table, groups, seed=args.seed)
        else:
            assignment, meta = builder(table, groups)
        manifest["views"][name] = meta

        payload = json.dumps(
            {"view": name, "assignment": {k: sorted(v) for k, v in assignment.items()}},
            sort_keys=True,
        )
        (args.out / f"split-{name}.json").write_text(payload + "\n")
        manifest["views"][name]["assignment_sha256"] = hashlib.sha256(
            payload.encode()
        ).hexdigest()

        rows = meta["counts"]
        LOG.info(
            "  %-9s train=%d val=%d cal=%d test=%d rows",
            name, rows["train"]["rows"], rows["validation"]["rows"],
            rows["calibration"]["rows"], rows["test"]["rows"],
        )

    # Eligibility is derived from the primary view and written into the same
    # manifest, so a benchmark run cannot pick a different task set than the
    # one the split was frozen with.
    eligibility = task_eligibility(manifest["views"]["temporal"])
    manifest["task_eligibility"] = eligibility
    LOG.info(
        "task eligibility: %d eligible, %d below floor",
        eligibility["n_eligible"], eligibility["n_ineligible"],
    )

    (args.out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOG.info("wrote split manifest to %s", args.out / "split_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
