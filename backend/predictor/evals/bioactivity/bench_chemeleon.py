"""Benchmark the CheMeleon-initialized Chemprop baseline (B3.5).

Run under the Python 3.11 env that has chemprop installed -- NOT the
`drug-tox-env` that runs `run_benchmark.py` for B0-B2b. See
`models/chemeleon_chemprop.py`'s module docstring for why the two live in
different interpreters and cannot share one process.

This script still reads the SAME frozen dataset and split manifest as
`run_benchmark.py`, and reuses its scoring functions directly (rdkit, scipy
and sklearn are all present in the chemprop env, so the import works) --
scoring logic is not duplicated, only the model-fitting half differs.

Usage:
    /path/to/py3.11/python bench_chemeleon.py \
        --dataset <hq_exact.csv.gz> --splits <manifests/splits> \
        --out <results dir> --views temporal --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PREDICTOR = HERE.parents[2]
for candidate in (PREDICTOR / "research", PREDICTOR / "evals"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_benchmark as rb  # noqa: E402  (reuse verify_manifest/score_view/etc.)
from bioactivity.models.chemeleon_chemprop import (  # noqa: E402
    ChemeleonMultiTaskBaseline,
)
from bioactivity.stress.activity_cliffs import build_cliff_set  # noqa: E402

LOG = logging.getLogger("bench_chemeleon")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", default="temporal")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = rb.verify_manifest(args.splits, args.dataset)
    rows = rb.load_dataset(args.dataset)
    LOG.info("dataset: %d rows, split manifest verified", len(rows))

    eligibility = manifest.get("task_eligibility")
    eligible_tasks = set(eligibility["eligible_tasks"]) if eligibility else None

    report = {
        "dataset": args.dataset.name,
        "model": "b3.5-chemeleon-chemprop",
        "seed": args.seed,
        "epochs": args.epochs,
        "views": {},
    }

    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        assignment = rb.load_split(args.splits, view)
        indices = rb.partition(rows, assignment)
        train_idx, test_idx = indices["train"], indices["test"]
        LOG.info("view %s: train=%d test=%d", view, len(train_idx), len(test_idx))

        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]

        LOG.info("building held-out cliff set for %s", view)
        cliff_pairs, _cliff_summary = build_cliff_set(rows, restrict_to=set(test_idx))

        workdir = args.out / f"chemeleon_workdir_{view}"
        model = ChemeleonMultiTaskBaseline(
            seed=args.seed, epochs=args.epochs, workdir=workdir,
        )

        started = time.time()
        model.fit(train_rows)
        fit_seconds = time.time() - started

        started = time.time()
        predictions = model.predict(test_rows)
        predict_seconds = time.time() - started

        if np.isnan(predictions).any():
            raise ValueError(
                f"chemeleon returned NaN for "
                f"{int(np.isnan(predictions).sum())} rows on view {view}"
            )

        scored = rb.score_view(
            rows, train_idx, test_idx, predictions, cliff_pairs, eligible_tasks
        )
        scored["model"] = model.describe()
        scored["timing_seconds"] = {
            "fit": round(fit_seconds, 1),
            "predict": round(predict_seconds, 1),
        }
        report["views"][view] = {
            "role": "PRIMARY" if view == rb.PRIMARY_VIEW else (
                "diagnostic" if view in rb.DIAGNOSTIC_VIEWS else "secondary"
            ),
            "models": {model.model_id: scored},
        }

        macro_mae = scored["macro"]["mae"]["macro"]
        LOG.info(
            "%s on %s: macro MAE=%s tasks=%d cliff dir-acc=%s (fit %.0fs, predict %.0fs)",
            model.model_id, view,
            "n/a" if macro_mae is None else f"{macro_mae:.4f}",
            scored["n_tasks_scored"],
            scored["cliffs"]["direction_accuracy"]["value"],
            fit_seconds, predict_seconds,
        )

        np.save(args.out / f"preds-{view}-{model.model_id}.npy", predictions)

    out_path = args.out / "benchmark_report_chemeleon.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    LOG.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
