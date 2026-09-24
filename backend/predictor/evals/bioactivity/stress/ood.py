"""Measure how out-of-distribution each split view actually is.

A view is only an OOD stress test if its test compounds are genuinely far from
training chemistry. That has to be measured, not assumed: a clustering that
produces many small clusters can hold out whole clusters and still leave every
test compound with a close neighbour in an adjacent cluster, at which point the
"OOD" view is no harder than a random one -- and a model looks like it
generalizes when it is interpolating.

The diagnostic is the distribution of each test compound's maximum ECFP4
Tanimoto similarity to any training compound. Lower is more OOD. Reporting it
per view makes the views comparable and exposes a clustering that is not
separating anything.

Usage:
    python -m bioactivity.stress.ood --dataset <hq_exact.csv.gz> \
        --splits <split dir> --out <results dir>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
PREDICTOR = HERE.parents[3]
for candidate in (PREDICTOR / "research", PREDICTOR / "evals"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

LOG = logging.getLogger("ood")

#: Chunk size for the similarity matmul, to bound peak memory.
CHUNK = 2000


def load_table(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def fingerprint_matrix(smiles_list: list[str]) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    matrix = np.zeros((len(smiles_list), 2048), dtype=np.float32)
    for row, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            matrix[row] = generator.GetFingerprintAsNumPy(mol)
    return matrix


def max_similarity_to_train(
    test_fp: np.ndarray, train_fp: np.ndarray
) -> np.ndarray:
    """Each test compound's best Tanimoto against the whole training set."""
    train_pop = train_fp.sum(axis=1)
    best = np.zeros(test_fp.shape[0], dtype=np.float32)
    for start in range(0, test_fp.shape[0], CHUNK):
        block = test_fp[start : start + CHUNK]
        intersection = block @ train_fp.T
        union = block.sum(axis=1)[:, None] + train_pop[None, :] - intersection
        similarity = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        )
        best[start : start + CHUNK] = similarity.max(axis=1)
    return best


def describe(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0}
    quantiles = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "n": int(values.size),
        "mean_max_tanimoto": round(float(values.mean()), 4),
        "median_max_tanimoto": round(float(quantiles[2]), 4),
        "p5": round(float(quantiles[0]), 4),
        "p25": round(float(quantiles[1]), 4),
        "p75": round(float(quantiles[3]), 4),
        "p95": round(float(quantiles[4]), 4),
        # A near-duplicate in train is the clearest sign a view is not OOD.
        "fraction_above_0.7": round(float((values > 0.7).mean()), 4),
        "fraction_above_0.9": round(float((values > 0.9).mean()), 4),
        "fraction_below_0.4": round(float((values < 0.4).mean()), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--views", default="temporal,cluster,scaffold,random"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    table = load_table(args.dataset)
    table = [r for r in table if r["high_disagreement"] != "1"]

    # One fingerprint per compound, not per row.
    by_compound: dict[str, str] = {}
    for row in table:
        by_compound.setdefault(row["connectivity_key"], row["standardized_smiles"])
    keys = sorted(by_compound)
    LOG.info("fingerprinting %d compounds", len(keys))
    matrix = fingerprint_matrix([by_compound[k] for k in keys])
    index_of = {key: i for i, key in enumerate(keys)}

    report: dict[str, Any] = {"dataset": args.dataset.name, "views": {}}
    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        path = args.splits / f"split-{view}.json"
        if not path.exists():
            LOG.warning("no split for view %s; skipping", view)
            continue
        assignment = json.loads(path.read_text())["assignment"]

        train_rows = [index_of[k] for k in assignment["train"] if k in index_of]
        test_rows = [index_of[k] for k in assignment["test"] if k in index_of]
        LOG.info(
            "view %-9s train=%d test=%d compounds", view, len(train_rows), len(test_rows)
        )
        best = max_similarity_to_train(matrix[test_rows], matrix[train_rows])
        report["views"][view] = describe(best)

    out_path = args.out / "ood_similarity.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"\n{'view':<10} {'median':>8} {'mean':>8} {'p95':>8} "
        f"{'>0.7':>7} {'>0.9':>7} {'<0.4':>7}"
    )
    print("-" * 62)
    for view, stats in report["views"].items():
        print(
            f"{view:<10} {stats['median_max_tanimoto']:>8.3f} "
            f"{stats['mean_max_tanimoto']:>8.3f} {stats['p95']:>8.3f} "
            f"{stats['fraction_above_0.7']:>7.3f} "
            f"{stats['fraction_above_0.9']:>7.3f} "
            f"{stats['fraction_below_0.4']:>7.3f}"
        )
    print(
        "\nLower similarity = more OOD. A view whose test compounds mostly have "
        "a >0.7 neighbour in train\nis measuring interpolation, not "
        "generalization, whatever it is named."
    )
    LOG.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
