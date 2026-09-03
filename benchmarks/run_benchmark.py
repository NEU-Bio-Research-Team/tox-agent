#!/usr/bin/env python3
"""Scientific benchmark (plan Phase 6).

Calls the application service directly — the same code path the API serves, so
a benchmark number and an API answer cannot disagree.

Rules this enforces:

* The split comes from a frozen manifest. Nothing re-splits at run time.
* Thresholds are only *applied*. Nothing is fitted here; fitting on the test
  set is how a calibration silently becomes a leak.
* Missing Tox21 labels are masked per task and the surviving count is reported,
  so a task evaluated on 300 molecules is not read as one evaluated on 783.
* hERG, Tox21 and ClinTox are scored separately. There is no pooled accuracy —
  averaging across chemically unrelated assays produces a number that means
  nothing.

Usage:
  python benchmarks/run_benchmark.py                 # full frozen split
  python benchmarks/run_benchmark.py --limit 100     # smoke size, for CI
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
SPLIT_MANIFEST = HERE / "manifests" / "eval-split-v1.json"
DEFAULT_OUT = HERE / "results"
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260903


# ---------------------------------------------------------------- metrics
def ece(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(y_true[mask].mean() - p[mask].mean())
    return float(total)


def point_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    both = len(np.unique(y_true)) == 2

    return {
        "n": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if y_true.size else None,
        "threshold": float(threshold),
        "auc_roc": float(roc_auc_score(y_true, p)) if both else None,
        "pr_auc": float(average_precision_score(y_true, p)) if both else None,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if both else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if both else None,
        "sensitivity": tp / (tp + fn) if (tp + fn) else None,
        "specificity": tn / (tn + fp) if (tn + fp) else None,
        "brier": float(brier_score_loss(y_true, p)),
        "ece_10bin": ece(y_true, p),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def bootstrap_ci(y_true: np.ndarray, p: np.ndarray, threshold: float,
                 keys=("auc_roc", "pr_auc", "f1", "mcc")) -> dict:
    """Percentile bootstrap 95% CI. Seeded, so a rerun reproduces it."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = y_true.size
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        yt, pp = y_true[idx], p[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = point_metrics(yt, pp, threshold)
        for k in keys:
            if m[k] is not None:
                samples[k].append(m[k])
    out = {}
    for k, vals in samples.items():
        if len(vals) < 100:
            out[k] = None
            continue
        out[k] = {
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)),
            "n_resamples": len(vals),
        }
    return out


# ---------------------------------------------------------------- runner
def score_endpoint(name, y_true, probs, threshold, per_sample=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    result = point_metrics(y_true, probs, threshold)
    result["bootstrap_95ci"] = bootstrap_ci(y_true, probs, threshold)
    if per_sample is not None:
        result["per_sample_file"] = per_sample
    return result


def predict_chunked(predictor, smiles: list[str], endpoints: list[str], chunk: int = 200):
    """Respect the service's batch cap instead of raising it.

    The 256-molecule limit is a deliberate guard on the API; a benchmark that
    lifted it would stop exercising the code path the service actually runs.
    """
    results, errors = [], []
    for start in range(0, len(smiles), chunk):
        part, part_errors = predictor.predict_batch(smiles[start : start + chunk], endpoints)
        results.extend(part)
        for e in part_errors:
            errors.append(e)
    return results, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap molecules per dataset")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not SPLIT_MANIFEST.exists():
        print(f"missing {SPLIT_MANIFEST}. Run benchmarks/build_split_manifest.py first.")
        return 1
    split = json.loads(SPLIT_MANIFEST.read_text())

    from toxpred.application.predictor import ToxicityPredictor
    from toxpred.domain.endpoints import TOX21_TASKS
    from toxpred.scientific.bootstrap import build_registry

    registry = build_registry()
    predictor = ToxicityPredictor(registry)
    served = set(registry.describe_capabilities())
    print(f"served endpoints: {sorted(served)}")

    provider = registry.for_capability("herg")
    herg_threshold = provider.artifact_herg_threshold
    tox21_thresholds = provider.artifact_tox21_thresholds

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provenance": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip() or None,
            "split_manifest": str(SPLIT_MANIFEST.relative_to(ROOT)),
            "split_sha256": split.get("content_sha256"),
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "limit": args.limit,
            "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
            "threshold_policy": "applied only; nothing fitted on the test split",
        },
        "endpoints": {},
    }

    # ---- hERG ------------------------------------------------------------
    if "herg" in served:
        rows = split["datasets"]["herg"]["test"][: args.limit]
        smiles = [r["canonical_smiles"] for r in rows]
        y = [r["label"] for r in rows]
        t0 = time.perf_counter()
        results, errors = predict_chunked(predictor, smiles, ["herg"])
        elapsed = time.perf_counter() - t0
        assert not errors, f"frozen split should be clean, got {len(errors)} errors"
        p = [r.herg.probability_blocker for r in results]

        per_sample = args.out / "herg_per_sample.json"
        per_sample.write_text(json.dumps(
            [{"canonical_smiles": s, "label": int(t), "probability_blocker": float(pi)}
             for s, t, pi in zip(smiles, y, p)], indent=2) + "\n")

        report["endpoints"]["herg"] = score_endpoint(
            "herg", y, p, herg_threshold, per_sample.name
        )
        report["endpoints"]["herg"]["threshold_source"] = "artifact"
        report["endpoints"]["herg"]["latency"] = {
            "total_s": round(elapsed, 2),
            "per_molecule_ms": round(elapsed / max(len(smiles), 1) * 1000, 2),
        }
        print(f"  hERG   n={len(smiles)} auc={report['endpoints']['herg']['auc_roc']:.4f}")

    # ---- Tox21 -----------------------------------------------------------
    if "tox21" in served:
        rows = split["datasets"]["tox21"]["test"][: args.limit]
        smiles = [r["canonical_smiles"] for r in rows]
        t0 = time.perf_counter()
        results, errors = predict_chunked(predictor, smiles, ["tox21"])
        elapsed = time.perf_counter() - t0
        assert not errors, f"frozen split should be clean, got {len(errors)} errors"

        by_task: dict[str, dict] = {}
        per_sample_rows = []
        prob_lookup = {
            task: [next(a for a in r.tox21.assays if a.task == task).probability_activity
                   for r in results]
            for task in TOX21_TASKS
        }
        for i, row in enumerate(rows):
            per_sample_rows.append({
                "canonical_smiles": row["canonical_smiles"],
                "labels": row["labels"],
                "probabilities": {t: float(prob_lookup[t][i]) for t in TOX21_TASKS},
            })
        per_sample = args.out / "tox21_per_sample.json"
        per_sample.write_text(json.dumps(per_sample_rows, indent=2) + "\n")

        for task in TOX21_TASKS:
            labels = [r["labels"][task] for r in rows]
            mask = [i for i, v in enumerate(labels) if v is not None]
            if len(mask) < 20:
                by_task[task] = {"n": len(mask), "skipped": "fewer than 20 labelled molecules"}
                continue
            y = np.array([labels[i] for i in mask], dtype=int)
            p = np.array([prob_lookup[task][i] for i in mask], dtype=float)
            by_task[task] = score_endpoint(task, y, p, tox21_thresholds[task])
            by_task[task]["threshold_source"] = "artifact"
            by_task[task]["n_masked_out"] = len(labels) - len(mask)

        scored = [m for m in by_task.values() if m.get("auc_roc") is not None]
        report["endpoints"]["tox21"] = {
            "per_task": by_task,
            "per_sample_file": per_sample.name,
            "macro": {
                "auc_roc": float(np.mean([m["auc_roc"] for m in scored])) if scored else None,
                "pr_auc": float(np.mean([m["pr_auc"] for m in scored])) if scored else None,
                "n_tasks_scored": len(scored),
                "note": "macro average over tasks; never pooled across molecules",
            },
            "latency": {
                "total_s": round(elapsed, 2),
                "per_molecule_ms": round(elapsed / max(len(smiles), 1) * 1000, 2),
            },
        }
        macro = report["endpoints"]["tox21"]["macro"]
        print(f"  Tox21  n={len(smiles)} macro_auc={macro['auc_roc']:.4f} "
              f"({macro['n_tasks_scored']}/12 tasks)")

    # ---- ClinTox ---------------------------------------------------------
    if "clintox" not in served:
        report["endpoints"]["clintox"] = {
            "skipped": True,
            "reason": registry.unavailable()
            .get("clintox-smilesgnn-v1", {})
            .get("reason", "not served by this build"),
        }
        print("  ClinTox skipped — not served")

    # ---- reproduction check ---------------------------------------------
    # The artifact ships the metrics its training run reported on the test
    # split. Recomputing them from the frozen split through the new code path
    # is the strongest available check that the split, the provider and the
    # scoring all still agree with the model that was shipped.
    recorded_path = (
        ROOT / "models" / "pretrained_2head_herg_chemberta_model"
        / "pretrained_2head_herg_tox21_metrics.json"
    )
    if recorded_path.exists() and args.limit is None:
        recorded = json.loads(recorded_path.read_text()).get("test", {})
        checks = []
        if "herg" in report["endpoints"] and "herg" in recorded:
            for key in ("auc_roc", "pr_auc"):
                checks.append({
                    "metric": f"herg.{key}",
                    "recorded": recorded["herg"][key],
                    "recomputed": report["endpoints"]["herg"][key],
                })
        if "tox21" in report["endpoints"] and "tox21" in recorded:
            checks.append({
                "metric": "tox21.macro_auc_roc",
                "recorded": recorded["tox21"]["macro_auc_roc"],
                "recomputed": report["endpoints"]["tox21"]["macro"]["auc_roc"],
            })
        tolerance = 5e-3
        for c in checks:
            c["abs_delta"] = abs(c["recorded"] - c["recomputed"])
            c["within_tolerance"] = c["abs_delta"] <= tolerance
        report["reproduction_check"] = {
            "source": str(recorded_path.relative_to(ROOT)),
            "tolerance": tolerance,
            "passed": all(c["within_tolerance"] for c in checks) if checks else None,
            "checks": checks,
        }
        status = "PASS" if report["reproduction_check"]["passed"] else "FAIL"
        print(f"\n  reproduction vs artifact-recorded test metrics: {status}")
        for c in checks:
            print(f"    {c['metric']:22s} recorded={c['recorded']:.4f} "
                  f"recomputed={c['recomputed']:.4f} delta={c['abs_delta']:.5f}")

    out = args.out / "benchmark_report.json"
    payload = json.dumps(report, indent=2, sort_keys=True)
    report["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
