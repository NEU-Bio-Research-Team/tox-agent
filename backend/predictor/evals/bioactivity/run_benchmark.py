"""Run the bioactivity benchmark against a frozen split manifest.

Contract with the rest of the repo:

  * The split is READ, never recomputed. If the manifest is missing or its
    dataset hash does not match, the run fails. It never re-splits, because a
    silently substituted split is the failure mode this benchmark exists to
    prevent.
  * The primary view is `temporal`. `random` is reported as a diagnostic and is
    explicitly marked as not a release criterion.
  * Tasks present in test but absent from train are reported as unscorable
    rather than scored against a global fallback -- a fixed-target model has no
    claim on an unseen task.
  * Every number is written with the label distribution it was computed over, so
    a strong MAE on a narrow task cannot be read as a strong model.

Usage:
    python run_benchmark.py --dataset <hq_exact.csv.gz> --splits <split dir> \
        --out <results dir> [--models b0,b1,b2b] [--views temporal,cluster]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
PREDICTOR = HERE.parents[2]
sys.path.insert(0, str(PREDICTOR / "research"))
sys.path.insert(0, str(PREDICTOR / "evals"))

from bioactivity.metrics import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    bedroc,
    binary_metrics,
    cliff_metrics,
    enrichment_factor,
    label_distribution,
    macro_summary,
    paired_bootstrap,
    regression_metrics,
)
from bioactivity.ingest.task_keys import task_unit_key  # noqa: E402
from bioactivity.stress.activity_cliffs import build_cliff_set  # noqa: E402

LOG = logging.getLogger("run_benchmark")

MODEL_ALIASES = {
    "b0": "b0-per-task-median",
    "b1": "b1-ecfp4-knn",
    "b2a": "b2a-ecfp4-rf",
    "b2b": "b2b-ecfp4-lightgbm",
}

PRIMARY_VIEW = "temporal"
DIAGNOSTIC_VIEWS = {"random"}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def load_split(splits_dir: Path, view: str) -> dict[str, set[str]]:
    path = splits_dir / f"split-{view}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no frozen split for view '{view}' at {path}. "
            "Run bioactivity.ingest.split first; this runner never splits."
        )
    payload = json.loads(path.read_text())
    return {name: set(keys) for name, keys in payload["assignment"].items()}


def verify_manifest(splits_dir: Path, dataset_path: Path) -> dict[str, Any]:
    """Fail unless the manifest was built from exactly this dataset file."""
    manifest_path = splits_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing split manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    actual = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    expected = manifest.get("dataset_sha256")
    if expected and expected != actual:
        raise ValueError(
            "dataset does not match the frozen split manifest.\n"
            f"  manifest: {expected}\n  dataset:  {actual}\n"
            "Rebuild the splits deliberately rather than scoring against a "
            "split that was frozen on different data."
        )
    return manifest


def partition(
    rows: list[dict[str, Any]], assignment: dict[str, set[str]]
) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {name: [] for name in assignment}
    for index, row in enumerate(rows):
        key = row["connectivity_key"]
        for name, keys in assignment.items():
            if key in keys:
                indices[name].append(index)
                break
    return indices


def build_model(name: str, seed: int):
    from bioactivity.models.ecfp_baselines import REGISTRY

    model_id = MODEL_ALIASES.get(name, name)
    if model_id not in REGISTRY:
        raise KeyError(
            f"unknown model '{name}'. Available: "
            f"{sorted(set(REGISTRY) | set(MODEL_ALIASES))}"
        )
    return REGISTRY[model_id](seed=seed)


def score_view(
    rows: list[dict[str, Any]],
    train_idx: list[int],
    test_idx: list[int],
    predictions: np.ndarray,
    cliff_pairs: list[Any],
    eligible_tasks: set[str] | None = None,
) -> dict[str, Any]:
    """All metrics for one (model, view), per task and aggregated.

    `eligible_tasks` comes from the frozen split manifest. Tasks outside it are
    still reported, with their counts, but are kept out of the macro so the
    headline number is not an average over 30-row tasks.
    """
    y_true = np.array([float(rows[i]["pactivity"]) for i in test_idx], dtype=float)

    # Grouping must use the MODELLING unit, which is what the split manifest's
    # eligibility gate is keyed on. Using the finer aggregation `task_key` here
    # made every eligibility lookup miss and left the macro empty.
    train_tasks = {task_unit_key(rows[i]) for i in train_idx}
    position = {row_index: slot for slot, row_index in enumerate(test_idx)}

    by_task: dict[str, list[int]] = defaultdict(list)
    for row_index in test_idx:
        by_task[task_unit_key(rows[row_index])].append(row_index)

    per_task_metrics: dict[str, dict[str, Any]] = {}
    per_task_raw: dict[str, dict[str, Any]] = {}
    below_floor: dict[str, dict[str, Any]] = {}
    unscorable: dict[str, str] = {}

    for task_key, row_indices in sorted(by_task.items()):
        if task_key not in train_tasks:
            # A fixed-target model was never given this task. Scoring it would
            # measure the fallback, not the model.
            unscorable[task_key] = "task_absent_from_train"
            continue

        slots = [position[i] for i in row_indices]
        task_true = y_true[slots]
        task_pred = predictions[slots]

        metrics = regression_metrics(task_true, task_pred)
        # Ineligible tasks are measured and published, but only eligible ones
        # feed the macro -- see the split manifest's task_eligibility block.
        if eligible_tasks is None or task_key in eligible_tasks:
            per_task_raw[task_key] = metrics
        else:
            below_floor[task_key] = {
                "n_test_rows": int(task_true.size),
                "mae": metrics["mae"].as_dict(),
            }

        entry: dict[str, Any] = {
            "regression": {k: v.as_dict() for k, v in metrics.items()},
            "labels": label_distribution(task_true),
        }
        for threshold in DEFAULT_THRESHOLDS:
            entry[f"binary_at_{threshold:g}"] = {
                k: v.as_dict()
                for k, v in binary_metrics(
                    task_true, task_pred, threshold=threshold
                ).items()
            }
            entry[f"ef1pct_at_{threshold:g}"] = enrichment_factor(
                task_true, task_pred, threshold=threshold, fraction=0.01
            ).as_dict()
            entry[f"ef5pct_at_{threshold:g}"] = enrichment_factor(
                task_true, task_pred, threshold=threshold, fraction=0.05
            ).as_dict()
            entry[f"bedroc_at_{threshold:g}"] = bedroc(
                task_true, task_pred, threshold=threshold
            ).as_dict()
        per_task_metrics[task_key] = entry

    # -- cliffs, on held-out pairs only ---------------------------------
    usable_pairs = [
        (position[p.index_a], position[p.index_b])
        for p in cliff_pairs
        if p.index_a in position and p.index_b in position
    ]
    cliffs = cliff_metrics(y_true, predictions, usable_pairs)

    summary = {
        "n_test_rows": len(test_idx),
        "n_test_rows_scored": sum(
            m["regression"]["mae"]["n"] for m in per_task_metrics.values()
        ),
        "n_tasks_scored": len(per_task_metrics),
        "n_tasks_in_macro": len(per_task_raw),
        "n_tasks_below_floor": len(below_floor),
        "tasks_below_floor": below_floor,
        "n_tasks_unscorable": len(unscorable),
        "unscorable_tasks": unscorable,
        "macro": {
            metric: macro_summary(per_task_raw, metric)
            for metric in ("mae", "rmse", "medae", "r2", "spearman")
        },
        "cliffs": {k: v.as_dict() for k, v in cliffs.items()},
        "per_task": per_task_metrics,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default="b0,b1,b2b")
    parser.add_argument("--views", default="temporal,cluster,scaffold,random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="paired bootstrap resamples (0 disables)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = verify_manifest(args.splits, args.dataset)
    rows = load_dataset(args.dataset)
    LOG.info("dataset: %d rows, split manifest verified", len(rows))

    eligibility = manifest.get("task_eligibility")
    eligible_tasks = set(eligibility["eligible_tasks"]) if eligibility else None
    if eligible_tasks is None:
        LOG.warning(
            "split manifest carries no task_eligibility block; scoring every "
            "task, including ones with too little held-out data to be meaningful"
        )
    else:
        LOG.info(
            "macro is computed over %d eligible tasks (%d below floor)",
            eligibility["n_eligible"], eligibility["n_ineligible"],
        )

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    view_names = [v.strip() for v in args.views.split(",") if v.strip()]

    report: dict[str, Any] = {
        "dataset": args.dataset.name,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "split_manifest_id": manifest.get("split_manifest_id"),
        "primary_view": PRIMARY_VIEW,
        "task_eligibility": eligibility,
        "seed": args.seed,
        "models": model_names,
        "views": {},
    }

    for view in view_names:
        assignment = load_split(args.splits, view)
        indices = partition(rows, assignment)
        train_idx, test_idx = indices["train"], indices["test"]
        LOG.info(
            "view %s: train=%d test=%d", view, len(train_idx), len(test_idx)
        )
        if not test_idx:
            LOG.warning("view %s has an empty test split; skipping", view)
            continue

        LOG.info("  building held-out cliff set for %s", view)
        cliff_pairs, cliff_summary = build_cliff_set(rows, restrict_to=set(test_idx))

        view_report: dict[str, Any] = {
            "role": "diagnostic only, not a release criterion"
            if view in DIAGNOSTIC_VIEWS
            else ("PRIMARY" if view == PRIMARY_VIEW else "secondary"),
            "counts": {name: len(idx) for name, idx in indices.items()},
            "cliff_set": {
                k: v for k, v in cliff_summary.items() if k != "per_task"
            },
            "models": {},
        }

        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        stored: dict[str, np.ndarray] = {}

        for name in model_names:
            model = build_model(name, args.seed)
            LOG.info("  fitting %s on %s", model.model_id, view)
            started = time.time()
            model.fit(train_rows)
            fit_seconds = time.time() - started

            started = time.time()
            predictions = model.predict(test_rows)
            predict_seconds = time.time() - started

            if np.isnan(predictions).any():
                raise ValueError(
                    f"{model.model_id} returned NaN for "
                    f"{int(np.isnan(predictions).sum())} rows"
                )

            scored = score_view(
                rows, train_idx, test_idx, predictions, cliff_pairs, eligible_tasks
            )
            scored["model"] = model.describe()
            scored["timing_seconds"] = {
                "fit": round(fit_seconds, 2),
                "predict": round(predict_seconds, 2),
                "predict_ms_per_row": round(
                    1000 * predict_seconds / max(len(test_rows), 1), 4
                ),
            }
            view_report["models"][model.model_id] = scored
            stored[model.model_id] = predictions

            macro_mae = scored["macro"]["mae"]["macro"]
            LOG.info(
                "  %-22s macro MAE=%s  tasks=%d  cliff dir-acc=%s",
                model.model_id,
                "n/a" if macro_mae is None else f"{macro_mae:.4f}",
                scored["n_tasks_scored"],
                scored["cliffs"]["direction_accuracy"]["value"],
            )

            # Per-row predictions enable paired error analysis later without
            # re-running the fit.
            np.save(
                args.out / f"preds-{view}-{model.model_id}.npy", predictions
            )

        # -- paired comparison against the strongest baseline -------------
        if args.bootstrap and len(stored) > 1:
            y_true = np.array(
                [float(rows[i]["pactivity"]) for i in test_idx], dtype=float
            )
            groups = [rows[i]["connectivity_key"] for i in test_idx]

            def macro_of(model_id: str) -> float:
                value = view_report["models"][model_id]["macro"]["mae"]["macro"]
                return float("inf") if value is None else value

            reference = min(stored, key=macro_of)
            comparisons = {}
            for model_id, predictions in stored.items():
                if model_id == reference:
                    continue
                comparisons[model_id] = paired_bootstrap(
                    y_true, predictions, stored[reference], groups,
                    n_resamples=args.bootstrap, seed=args.seed,
                )
            view_report["paired_vs_best"] = {
                "reference_model": reference,
                "interpretation": "negative difference means the model beats "
                "the reference on MAE",
                "comparisons": comparisons,
            }

        report["views"][view] = view_report

    out_path = args.out / "benchmark_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    LOG.info("wrote %s", out_path)

    # -- console summary -------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'view':<10} {'role':<12} {'model':<24} {'macro MAE':>10} {'tasks':>6}")
    print("=" * 78)
    for view, view_report in report["views"].items():
        for model_id, scored in view_report["models"].items():
            macro = scored["macro"]["mae"]["macro"]
            print(
                f"{view:<10} {view_report['role'][:11]:<12} {model_id:<24} "
                f"{('n/a' if macro is None else f'{macro:.4f}'):>10} "
                f"{scored['n_tasks_scored']:>6}"
            )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
