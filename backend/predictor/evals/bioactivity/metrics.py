"""Metric suite for the bioactivity benchmark (protocol section 7).

Design rules carried over from the toxicity benchmark:

  * A metric that is undefined for a task returns None and says why. It never
    emits a placeholder number -- a 0.5 AUROC for a task with no actives looks
    like a measurement and is not one.
  * Macro scores are reported unweighted AND weighted, plus the worst decile,
    because a panel average hides the tasks the model cannot do.
  * Confidence intervals come from bootstrap resampling at COMPOUND level, not
    row level. Rows sharing a compound are not independent, so row bootstrap
    understates the interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats

#: Derived binary views from the data contract, section 2.1.
DEFAULT_THRESHOLDS = (6.0, 7.0)

MIN_SAMPLES_FOR_METRIC = 20
MIN_POSITIVES_FOR_RANKING = 5


@dataclass
class MetricResult:
    """A metric value, or an explicit reason it could not be computed."""

    value: float | None
    n: int
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "n": self.n}
        if self.reason:
            out["undefined_reason"] = self.reason
        return out


def _undefined(n: int, reason: str) -> MetricResult:
    return MetricResult(value=None, n=n, reason=reason)


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------

def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, MetricResult]:
    """MAE, RMSE, R2 and rank correlations for one task."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(y_true.size)

    if n == 0:
        return {k: _undefined(0, "no_samples") for k in
                ("mae", "rmse", "medae", "r2", "spearman", "pearson")}

    errors = y_pred - y_true
    out: dict[str, MetricResult] = {
        "mae": MetricResult(float(np.mean(np.abs(errors))), n),
        "rmse": MetricResult(float(np.sqrt(np.mean(errors ** 2))), n),
        "medae": MetricResult(float(np.median(np.abs(errors))), n),
    }

    # R2 and correlations need label variance. A task whose labels are almost
    # constant produces a meaningless (often hugely negative) R2.
    variance = float(np.var(y_true))
    if n < 3 or variance < 1e-12:
        reason = "constant_labels" if variance < 1e-12 else "too_few_samples"
        out["r2"] = _undefined(n, reason)
        out["spearman"] = _undefined(n, reason)
        out["pearson"] = _undefined(n, reason)
        return out

    out["r2"] = MetricResult(
        float(1.0 - np.sum(errors ** 2) / np.sum((y_true - y_true.mean()) ** 2)), n
    )
    # Constant predictions make correlation undefined rather than zero.
    if float(np.var(y_pred)) < 1e-12:
        out["spearman"] = _undefined(n, "constant_predictions")
        out["pearson"] = _undefined(n, "constant_predictions")
    else:
        out["spearman"] = MetricResult(
            float(stats.spearmanr(y_true, y_pred).statistic), n
        )
        out["pearson"] = MetricResult(
            float(stats.pearsonr(y_true, y_pred).statistic), n
        )
    return out


def label_distribution(y_true: np.ndarray) -> dict[str, float]:
    """Label spread, so a suspiciously good task can be recognized as narrow."""
    y_true = np.asarray(y_true, dtype=float)
    if y_true.size == 0:
        return {}
    q25, q75 = np.percentile(y_true, [25, 75])
    return {
        "n": int(y_true.size),
        "min": float(y_true.min()),
        "max": float(y_true.max()),
        "mean": float(y_true.mean()),
        "std": float(y_true.std()),
        "iqr": float(q75 - q25),
    }


# --------------------------------------------------------------------------
# ranking / virtual screening
# --------------------------------------------------------------------------

def enrichment_factor(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float, fraction: float
) -> MetricResult:
    """EF at a top fraction: hit rate in the top slice over the base rate."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = int(y_true.size)
    actives = (y_true >= threshold).astype(int)
    n_actives = int(actives.sum())

    if n < MIN_SAMPLES_FOR_METRIC:
        return _undefined(n, "too_few_samples")
    if n_actives < MIN_POSITIVES_FOR_RANKING:
        return _undefined(n, f"too_few_actives_at_{threshold}")
    if n_actives == n:
        return _undefined(n, f"all_actives_at_{threshold}")

    k = max(1, int(math.ceil(fraction * n)))
    top = np.argsort(-y_score)[:k]
    hit_rate = float(actives[top].sum()) / k
    base_rate = n_actives / n
    return MetricResult(float(hit_rate / base_rate), n)


def bedroc(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float, alpha: float = 20.0
) -> MetricResult:
    """BEDROC: early-recognition-weighted ranking metric (Truchon & Bayly)."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = int(y_true.size)
    actives = (y_true >= threshold).astype(int)
    n_actives = int(actives.sum())

    if n < MIN_SAMPLES_FOR_METRIC:
        return _undefined(n, "too_few_samples")
    if n_actives < MIN_POSITIVES_FOR_RANKING or n_actives == n:
        return _undefined(n, f"degenerate_actives_at_{threshold}")

    order = np.argsort(-y_score)
    ranks = np.where(actives[order] == 1)[0] + 1
    ratio = n_actives / n

    rie_sum = float(np.sum(np.exp(-alpha * ranks / n)))
    random_sum = ratio * (1 - math.exp(-alpha)) / (math.exp(alpha / n) - 1)
    rie = rie_sum / random_sum

    factor = ratio * math.sinh(alpha / 2) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ratio))
    offset = 1.0 / (1 - math.exp(alpha * (1 - ratio)))
    return MetricResult(float(rie * factor + offset), n)


def binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float
) -> dict[str, MetricResult]:
    """PR-AUC (primary when actives are rare) and AUROC for a derived view."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = int(y_true.size)
    actives = (y_true >= threshold).astype(int)
    n_actives = int(actives.sum())

    if n < MIN_SAMPLES_FOR_METRIC or n_actives == 0 or n_actives == n:
        reason = (
            "too_few_samples" if n < MIN_SAMPLES_FOR_METRIC
            else f"single_class_at_{threshold}"
        )
        return {
            "pr_auc": _undefined(n, reason),
            "auroc": _undefined(n, reason),
            "prevalence": MetricResult(n_actives / n if n else None, n),
        }

    return {
        "pr_auc": MetricResult(float(average_precision_score(actives, y_score)), n),
        "auroc": MetricResult(float(roc_auc_score(actives, y_score)), n),
        "prevalence": MetricResult(float(n_actives / n), n),
    }


# --------------------------------------------------------------------------
# activity cliffs
# --------------------------------------------------------------------------

def cliff_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pair_indices: Sequence[tuple[int, int]],
) -> dict[str, MetricResult]:
    """Error on activity-cliff pairs, and whether the direction is right.

    A model can have a fine global RMSE and still order every cliff pair
    backwards, which is the failure that matters in a medicinal-chemistry
    series -- so direction accuracy is reported separately from magnitude.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_pairs = len(pair_indices)

    if n_pairs == 0:
        return {
            "delta_mae": _undefined(0, "no_cliff_pairs"),
            "direction_accuracy": _undefined(0, "no_cliff_pairs"),
            "mae_cliff": _undefined(0, "no_cliff_pairs"),
        }

    true_delta = np.array([y_true[i] - y_true[j] for i, j in pair_indices])
    pred_delta = np.array([y_pred[i] - y_pred[j] for i, j in pair_indices])

    involved = sorted({i for pair in pair_indices for i in pair})
    mae_cliff = float(np.mean(np.abs(y_pred[involved] - y_true[involved])))

    # Ties in the prediction count as wrong: the model failed to separate them.
    correct = float(np.mean(np.sign(pred_delta) == np.sign(true_delta)))

    return {
        "delta_mae": MetricResult(float(np.mean(np.abs(pred_delta - true_delta))), n_pairs),
        "direction_accuracy": MetricResult(correct, n_pairs),
        "mae_cliff": MetricResult(mae_cliff, len(involved)),
    }


# --------------------------------------------------------------------------
# calibration / uncertainty
# --------------------------------------------------------------------------

def interval_metrics(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, nominal: float
) -> dict[str, MetricResult]:
    """Empirical coverage and sharpness for a predictive interval.

    Coverage without sharpness is not usefulness: an infinitely wide interval
    covers everything. Both are always reported together.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = int(y_true.size)
    if n == 0:
        return {
            "coverage": _undefined(0, "no_samples"),
            "mean_width": _undefined(0, "no_samples"),
            "coverage_error": _undefined(0, "no_samples"),
        }

    inside = (y_true >= lower) & (y_true <= upper)
    coverage = float(np.mean(inside))
    return {
        "coverage": MetricResult(coverage, n),
        "mean_width": MetricResult(float(np.mean(upper - lower)), n),
        "coverage_error": MetricResult(float(coverage - nominal), n),
    }


def risk_coverage_curve(
    y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray
) -> dict[str, Any]:
    """MAE as the most-uncertain predictions are abstained from.

    If error does not fall as coverage falls, the uncertainty estimate carries
    no usable signal and must not be presented as a confidence indicator.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    n = y_true.size
    if n < MIN_SAMPLES_FOR_METRIC:
        return {"points": [], "monotonic": None, "reason": "too_few_samples"}

    order = np.argsort(uncertainty)  # most certain first
    points = []
    for fraction in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3):
        k = max(1, int(fraction * n))
        keep = order[:k]
        points.append(
            {
                "coverage": round(k / n, 4),
                "mae": round(float(np.mean(np.abs(y_pred[keep] - y_true[keep]))), 4),
            }
        )
    maes = [p["mae"] for p in points]
    return {
        "points": points,
        # Error should fall (or hold) as we abstain more; tolerance absorbs noise.
        "monotonic": bool(all(a >= b - 1e-9 for a, b in zip(maes, maes[1:]))),
    }


# --------------------------------------------------------------------------
# aggregation across tasks
# --------------------------------------------------------------------------

def macro_summary(
    per_task: dict[str, dict[str, MetricResult]], metric: str
) -> dict[str, Any]:
    """Unweighted and weighted macro, plus worst decile (protocol 7.1)."""
    values, weights, defined_tasks, undefined = [], [], [], {}
    for task, metrics in sorted(per_task.items()):
        result = metrics.get(metric)
        if result is None or result.value is None:
            undefined[task] = (result.reason if result else "missing") or "missing"
            continue
        values.append(result.value)
        weights.append(result.n)
        defined_tasks.append(task)

    if not values:
        return {
            "macro": None,
            "weighted_macro": None,
            "n_tasks": 0,
            "undefined_tasks": undefined,
        }

    array = np.array(values, dtype=float)
    weight_array = np.array(weights, dtype=float)

    # "Worst decile" for error-like metrics means the largest values; for
    # score-like metrics the smallest. Error metrics are the ones we gate on.
    lower_is_better = metric in {"mae", "rmse", "medae", "delta_mae", "mae_cliff"}
    ordered = np.sort(array)[::-1] if lower_is_better else np.sort(array)
    k = max(1, int(math.ceil(0.1 * ordered.size)))

    return {
        "macro": float(array.mean()),
        "weighted_macro": float(np.average(array, weights=weight_array)),
        "worst_decile": float(ordered[:k].mean()),
        "n_tasks": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "undefined_tasks": undefined,
    }


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    metric: str = "mae",
) -> dict[str, Any]:
    """Paired bootstrap on the A-B metric difference, resampled by compound.

    Resampling compounds rather than rows respects the fact that a compound's
    measurements across targets are correlated. Returns the CI of (A - B); for
    an error metric a wholly negative interval means A is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    group_keys = list(by_group)

    def score(indices: np.ndarray, predictions: np.ndarray) -> float:
        errors = predictions[indices] - y_true[indices]
        if metric == "rmse":
            return float(np.sqrt(np.mean(errors ** 2)))
        return float(np.mean(np.abs(errors)))

    all_indices = np.arange(y_true.size)
    observed = score(all_indices, pred_a) - score(all_indices, pred_b)

    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    for step in range(n_resamples):
        sampled = rng.choice(len(group_keys), size=len(group_keys), replace=True)
        indices = np.fromiter(
            (i for s in sampled for i in by_group[group_keys[s]]), dtype=int
        )
        differences[step] = score(indices, pred_a) - score(indices, pred_b)

    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "metric": metric,
        "observed_difference": float(observed),
        "ci_95": [float(low), float(high)],
        "excludes_zero": bool(low > 0 or high < 0),
        "n_resamples": n_resamples,
        "resample_unit": "compound",
        "n_groups": len(group_keys),
    }
