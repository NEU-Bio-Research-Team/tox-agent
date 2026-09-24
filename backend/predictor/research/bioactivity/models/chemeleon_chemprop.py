"""B3.5: Chemprop D-MPNN initialized from the CheMeleon foundation checkpoint.

CheMeleon is a ~10M-parameter D-MPNN pretrained on 1M PubChem molecules to
predict noise-free Mordred descriptors; the published headline result is a 97%
win rate against plain Chemprop/RF/fastprop on the MoleculeACE activity-cliff
suite (Burns, 2026 -- see plan footnote [^19]). It is officially merged into
Chemprop's CLI (`--from-foundation CHEMELEON`), so this module drives that CLI
rather than hand-rolling an equivalent forward pass: the CLI path is what the
authors tested, and reimplementing the loading logic risks a subtle mismatch
that silently degrades the checkpoint's pretrained weights.

Environment note -- read before running anything else in this file:
    Chemprop 2.x requires Python >=3.11 (`enum.StrEnum`). `drug-tox-env`
    (Python 3.10, home to the other baselines and `run_benchmark.py`) cannot
    import chemprop at all. This module runs under `comosa_phase1`, the only
    Python 3.11 env on this machine with rdkit+torch already installed.
    Consequently this baseline is invoked through
    `bench_chemeleon.py` (a small standalone driver), NOT through
    `run_benchmark.py`'s in-process model registry -- the two halves of the
    benchmark run under different Python interpreters and never import each
    other. This module still implements the SAME fit/predict/describe shape
    as `ecfp_baselines.BaseBioactivityBaseline` for consistency and so it can
    be folded back into the shared registry once/if the environment split is
    resolved.

Multi-task design, and what it trades away:
    Unlike the classical per-task baselines, one Chemprop encoder is trained
    JOINTLY across every (target, standard_type) task_unit_key, with a
    per-task output column and NaN-masked loss for rows missing that task --
    this is what lets the pretrained representation transfer across targets,
    which is the entire point of fine-tuning a foundation checkpoint rather
    than training 27 independent small networks from scratch.

    The cost: Chemprop's CLI has no built-in per-row categorical conditioning
    equivalent to the classical baselines' `ContextEncoder`. Records sharing a
    task_unit_key but differing in assay context are aggregated (median) into
    one training target per compound, the same way `build_dataset.py`
    aggregates within a task_key -- this loses assay-context resolution
    within a task_unit_key that the classical baselines keep. That trade-off
    is deliberate here and should be treated as a documented limitation of
    this specific baseline, not silently glossed over when comparing macro
    MAE against B0-B2b.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LOG = logging.getLogger(__name__)

MODEL_ID = "b3.5-chemeleon-chemprop"
FOUNDATION_CHECKPOINT = "CHEMELEON"

#: Column-name-safe encoding of a task_unit_key ("CHEMBL203|IC50" has a pipe,
#: which is legal in a CSV header but easy to mishandle elsewhere).
def _column_name(task_unit_key: str) -> str:
    return task_unit_key.replace("|", "__")


def _task_key_from_column(column: str) -> str:
    return column.replace("__", "|")


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    LOG.info("$ %s", " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Surface both streams: Lightning/Chemprop put the useful error on
        # stderr, but configuration echoes often land on stdout.
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
        )
    LOG.debug("stdout tail:\n%s", result.stdout[-2000:])


@dataclass
class ChemeleonMultiTaskBaseline:
    """Fine-tunes one CheMeleon-initialized D-MPNN across all training tasks.

    `fit`/`predict` intentionally mirror `BaseBioactivityBaseline`'s shape
    (same method names and row-dict contract) so this class could later join
    the shared model registry if the Python-version split is resolved; today
    it is driven by `bench_chemeleon.py` directly.
    """

    model_id: str = MODEL_ID
    seed: int = 42
    epochs: int = 30
    workdir: Path | None = None
    chemprop_bin: str | None = None

    def __post_init__(self) -> None:
        if self.workdir is None:
            raise ValueError("workdir is required (holds CSVs and checkpoints)")
        self.workdir = Path(self.workdir)
        if self.chemprop_bin is None:
            # Resolve the `chemprop` console script next to THIS interpreter,
            # not whatever `chemprop` happens to be first on PATH -- this
            # module only ever runs under the Python 3.11 env that has
            # chemprop installed, and there is no guarantee PATH agrees.
            self.chemprop_bin = str(Path(sys.executable).parent / "chemprop")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.task_columns: list[str] = []
        self.global_fallback: dict[str, float] = {}
        self.model_dir: Path | None = None

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        validation_rows: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        from bioactivity.ingest.task_keys import task_unit_key

        # One row per unique structure; median-aggregate a task's pActivity
        # across assay contexts (see module docstring for what this costs).
        by_smiles: dict[str, dict[str, list[float]]] = {}
        task_units: set[str] = set()
        for row in rows:
            smiles = row["standardized_smiles"]
            unit = task_unit_key(row)
            task_units.add(unit)
            by_smiles.setdefault(smiles, {}).setdefault(unit, []).append(
                float(row["pactivity"])
            )

        self.task_columns = sorted(_column_name(u) for u in task_units)
        LOG.info(
            "fitting %s: %d unique compounds, %d tasks",
            self.model_id, len(by_smiles), len(self.task_columns),
        )

        # Global per-task median, used as the fallback for a task the fitted
        # model never saw a column for (mirrors the classical baselines'
        # `global_fallback`, required because `run_benchmark.py` rejects any
        # NaN in a model's predictions).
        all_values: dict[str, list[float]] = {}
        for per_task in by_smiles.values():
            for unit, values in per_task.items():
                all_values.setdefault(unit, []).extend(values)
        self.global_fallback = {
            unit: float(np.median(values)) for unit, values in all_values.items()
        }

        train_csv = self.workdir / "train.csv"
        with open(train_csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["smiles", *self.task_columns])
            for smiles, per_task in by_smiles.items():
                row_out = [smiles]
                for column in self.task_columns:
                    unit = _task_key_from_column(column)
                    values = per_task.get(unit)
                    row_out.append(
                        "" if values is None else round(float(np.median(values)), 5)
                    )
                writer.writerow(row_out)

        self.model_dir = self.workdir / "model"
        cmd = [
            self.chemprop_bin, "train",
            "-i", str(train_csv),
            "--smiles-columns", "smiles",
            "--target-columns", *self.task_columns,
            "--from-foundation", FOUNDATION_CHECKPOINT,
            "--epochs", str(self.epochs),
            "--patience", "5",
            "--split", "random",
            "--split-sizes", "0.9", "0.1", "0.0",
            "--data-seed", str(self.seed),
            "--pytorch-seed", str(self.seed),
            "--num-workers", "4",
            "--save-dir", str(self.model_dir),
            "--accelerator", "gpu",
            "--devices", "1",
        ]
        _run(cmd)
        LOG.info("%s: fit complete, checkpoint at %s", self.model_id, self.model_dir)

    # -- prediction ---------------------------------------------------------

    def predict(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        from bioactivity.ingest.task_keys import task_unit_key

        if self.model_dir is None:
            raise RuntimeError("predict() called before fit()")

        unique_smiles = sorted({row["standardized_smiles"] for row in rows})
        query_csv = self.workdir / "query.csv"
        with open(query_csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["smiles"])
            for smiles in unique_smiles:
                writer.writerow([smiles])

        checkpoint = self.model_dir / "model_0" / "best.pt"
        preds_csv = self.workdir / "preds.csv"
        cmd = [
            self.chemprop_bin, "predict",
            "-i", str(query_csv),
            "--smiles-columns", "smiles",
            "--model-paths", str(checkpoint),
            "-o", str(preds_csv),
        ]
        _run(cmd)

        by_smiles: dict[str, dict[str, float]] = {}
        with open(preds_csv, newline="") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                smiles = record["smiles"]
                by_smiles[smiles] = {
                    column: float(record[column])
                    for column in self.task_columns
                    if record.get(column) not in (None, "")
                }

        trained_units = {_task_key_from_column(c) for c in self.task_columns}
        predictions = np.full(len(rows), np.nan, dtype=float)
        n_fallback = 0
        for index, row in enumerate(rows):
            unit = task_unit_key(row)
            column = _column_name(unit)
            per_smiles = by_smiles.get(row["standardized_smiles"], {})
            if unit in trained_units and column in per_smiles:
                predictions[index] = per_smiles[column]
            else:
                # Task the model never saw a training column for; classical
                # baselines fall back the same way for a genuinely unseen task.
                predictions[index] = self.global_fallback.get(
                    unit, float(np.median(list(self.global_fallback.values())))
                )
                n_fallback += 1
        if n_fallback:
            LOG.warning(
                "%d/%d predictions used the global fallback (task not in "
                "training columns)", n_fallback, len(rows),
            )
        return predictions

    # -- provenance -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "foundation_checkpoint": FOUNDATION_CHECKPOINT,
            "chemprop_version": _chemprop_version(),
            "task_unit": "(target_chembl_id, standard_type), multi-task shared "
            "encoder -- NOT per-(task, assay_context) like the classical "
            "baselines; see module docstring",
            "epochs": self.epochs,
            "seed": self.seed,
            "n_tasks": len(self.task_columns),
            "n_tasks_on_global_fallback": 0,  # filled by caller if it tracks it
        }


def _chemprop_version() -> str:
    try:
        import chemprop

        return str(chemprop.__version__)
    except Exception:
        return "unknown"
