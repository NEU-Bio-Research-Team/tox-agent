"""Baselines B0-B2: dummy, ECFP kNN, and ECFP tree ensembles.

These are not filler. On activity-cliff and low-data tasks, descriptor-based
models regularly match or beat graph neural networks (MoleculeACE benchmarked 24
methods and found exactly this), so a neural architecture that cannot beat
LightGBM-on-ECFP4 under a frozen split has not demonstrated anything.

All three are fitted per (target, standard_type) -- the modelling unit defined
in `ingest.task_keys`, not the finer aggregation unit. Splitting further by assay
context fragments the panel into mostly-tiny groups; instead the context is
handed to the estimator as one-hot features, so a binding and a cell-based row
are modelled together but remain distinguishable.

A fingerprint model has no mechanism for sharing strength across targets, so
per-target fitting is the honest classical formulation; pooling every target
into one regressor would understate the baseline.

A task with too few training rows gets the task-median predictor instead of a
failed fit, and that substitution is recorded per task rather than hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from bioactivity.ingest.task_keys import task_unit_key
from bioactivity.models.featurize import (
    FEATURE_SCHEMA_VERSION,
    ContextEncoder,
    Featurizer,
)

LOG = logging.getLogger(__name__)

#: Below this many training rows, a per-task supervised fit is not meaningful.
MIN_TRAIN_ROWS_PER_TASK = 30


@dataclass
class PerTaskModel:
    """A fitted per-task regressor plus the fallback it may have used."""

    task_key: str
    estimator: Any | None
    fallback_value: float
    n_train: int
    used_fallback: bool
    reason: str | None = None


class BaseBioactivityBaseline:
    """Common per-task fit/predict plumbing.

    Subclasses implement `_make_estimator`. `model_id` names the artifact in
    benchmark reports.
    """

    model_id = "base"
    requires_descriptors = False

    def __init__(self, *, seed: int = 42, **kwargs: Any) -> None:
        self.seed = seed
        self.params = kwargs
        self.featurizer = Featurizer(use_descriptors=self.requires_descriptors)
        self.context = ContextEncoder()
        self.models: dict[str, PerTaskModel] = {}
        self.global_fallback: float = 0.0

    # -- to implement -----------------------------------------------------

    def _make_estimator(self, n_train: int) -> Any:
        raise NotImplementedError

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        """Last-mile dtype adaptation; overridden where an estimator needs it."""
        return features

    def _design_matrix(
        self, rows: Sequence[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Structure features concatenated with assay-context features."""
        structure, valid = self.featurizer.featurize(
            [r["standardized_smiles"] for r in rows]
        )
        context = self.context.transform(rows)
        return np.hstack([structure, context]), valid

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        validation_rows: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Fit one estimator per task key.

        `validation_rows` is accepted for interface symmetry with the neural
        baselines; the classical models here do not early-stop on it.
        """
        del validation_rows

        labels = np.array([float(r["pactivity"]) for r in rows], dtype=float)
        self.global_fallback = float(np.median(labels)) if labels.size else 0.0

        # Context vocabulary comes from training rows only; test-time contexts
        # outside it encode as `unknown` rather than leaking into the fit.
        self.context.fit(rows)

        by_task: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            by_task.setdefault(task_unit_key(row), []).append(index)

        for task_key, indices in by_task.items():
            task_labels = labels[indices]
            median = float(np.median(task_labels))

            if len(indices) < MIN_TRAIN_ROWS_PER_TASK:
                self.models[task_key] = PerTaskModel(
                    task_key=task_key, estimator=None, fallback_value=median,
                    n_train=len(indices), used_fallback=True,
                    reason=f"only_{len(indices)}_train_rows",
                )
                continue

            features, valid = self._design_matrix([rows[i] for i in indices])
            if valid.sum() < MIN_TRAIN_ROWS_PER_TASK:
                self.models[task_key] = PerTaskModel(
                    task_key=task_key, estimator=None, fallback_value=median,
                    n_train=int(valid.sum()), used_fallback=True,
                    reason="too_few_featurizable_rows",
                )
                continue

            # Labels can be near-constant within a narrow assay context; a tree
            # ensemble on constant targets is a wasteful way to store a mean.
            if float(np.std(task_labels[valid])) < 1e-6:
                self.models[task_key] = PerTaskModel(
                    task_key=task_key, estimator=None, fallback_value=median,
                    n_train=int(valid.sum()), used_fallback=True,
                    reason="constant_labels",
                )
                continue

            estimator = self._make_estimator(int(valid.sum()))
            estimator.fit(self._prepare_features(features[valid]), task_labels[valid])
            self.models[task_key] = PerTaskModel(
                task_key=task_key, estimator=estimator, fallback_value=median,
                n_train=int(valid.sum()), used_fallback=False,
            )

        n_fallback = sum(1 for m in self.models.values() if m.used_fallback)
        LOG.info(
            "%s: fitted %d tasks (%d on median fallback)",
            self.model_id, len(self.models), n_fallback,
        )

    # -- prediction -------------------------------------------------------

    def predict(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        """Predict pActivity for each row, grouped by task for efficiency."""
        predictions = np.full(len(rows), np.nan, dtype=float)

        by_task: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            by_task.setdefault(task_unit_key(row), []).append(index)

        for task_key, indices in by_task.items():
            model = self.models.get(task_key)
            if model is None:
                # An unseen task cannot be predicted by a fixed-target model.
                # The benchmark runner reports these rather than scoring them.
                predictions[indices] = self.global_fallback
                continue
            if model.estimator is None:
                predictions[indices] = model.fallback_value
                continue

            features, valid = self._design_matrix([rows[i] for i in indices])
            index_array = np.asarray(indices)
            if valid.any():
                predictions[index_array[valid]] = model.estimator.predict(
                    self._prepare_features(features[valid])
                )
            if (~valid).any():
                predictions[index_array[~valid]] = model.fallback_value

        return predictions

    # -- provenance -------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "uses_descriptors": self.requires_descriptors,
            "n_context_features": self.context.n_features,
            "task_unit": "(target_chembl_id, standard_type)",
            "seed": self.seed,
            "params": self.params,
            "n_tasks": len(self.models),
            "n_tasks_on_fallback": sum(
                1 for m in self.models.values() if m.used_fallback
            ),
            "fallback_tasks": {
                key: model.reason
                for key, model in sorted(self.models.items())
                if model.used_fallback
            },
        }


# --------------------------------------------------------------------------
# B0: dummy
# --------------------------------------------------------------------------

class MedianBaseline(BaseBioactivityBaseline):
    """B0. Per-task median. Exists to prove the pipeline and metrics work.

    Any model that does not clearly beat this is broken, not merely weak.
    """

    model_id = "b0-per-task-median"

    def _make_estimator(self, n_train: int) -> Any:
        return None

    def fit(self, rows, *, validation_rows=None) -> None:  # type: ignore[override]
        del validation_rows
        labels = np.array([float(r["pactivity"]) for r in rows], dtype=float)
        self.global_fallback = float(np.median(labels)) if labels.size else 0.0

        by_task: dict[str, list[float]] = {}
        for row in rows:
            by_task.setdefault(task_unit_key(row), []).append(
                float(row["pactivity"])
            )
        for task_key, values in by_task.items():
            self.models[task_key] = PerTaskModel(
                task_key=task_key, estimator=None,
                fallback_value=float(np.median(values)),
                n_train=len(values), used_fallback=True, reason="dummy_by_design",
            )
        LOG.info("%s: %d task medians", self.model_id, len(self.models))

    def describe(self) -> dict[str, Any]:
        described = super().describe()
        # Every task is "fallback" by construction here, which would otherwise
        # read as a pipeline failure in the report.
        described["fallback_tasks"] = {}
        described["n_tasks_on_fallback"] = 0
        described["note"] = "dummy predictor; per-task median by design"
        return described


# --------------------------------------------------------------------------
# B1: similarity
# --------------------------------------------------------------------------

class _TanimotoKnn:
    """Similarity-weighted kNN with Tanimoto computed as one matrix product.

    sklearn's `metric="jaccard"` goes through a scipy per-pair callback, which
    at panel scale takes hours. For binary fingerprints Tanimoto has a closed
    form that reduces to a single BLAS call:

        |A & B| = A @ B.T
        T(A,B)  = |A & B| / (|A| + |B| - |A & B|)

    Same metric and the same selected neighbours, in seconds rather than hours
    (0.5s versus minutes-to-hours for a single 8.8k x 2.9k task).

    Neighbours are weighted by similarity directly, not by sklearn's
    1/distance. For a near-duplicate pair Tanimoto approaches 1, so 1/distance
    diverges and a single neighbour swamps the average; similarity weighting
    stays bounded. Predictions therefore track sklearn's closely (r ~ 0.996 on
    random fingerprints) without being identical to it.

    Only the fingerprint block participates in the similarity: chemical
    neighbourhood is a property of structure, and letting one-hot assay-context
    columns contribute would make two unrelated molecules in the same assay
    format look similar.
    """

    def __init__(self, k: int, n_structure_features: int) -> None:
        self.k = k
        self.n_structure = n_structure_features
        self._train: np.ndarray | None = None
        self._popcount: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_TanimotoKnn":
        block = (features[:, : self.n_structure] > 0).astype(np.float32)
        self._train = block
        self._popcount = block.sum(axis=1)
        self._y = np.asarray(labels, dtype=float)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._train is None or self._y is None or self._popcount is None:
            raise RuntimeError("_TanimotoKnn.predict before fit")

        query = (features[:, : self.n_structure] > 0).astype(np.float32)
        intersection = query @ self._train.T
        union = (
            query.sum(axis=1)[:, None] + self._popcount[None, :] - intersection
        )
        similarity = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        )

        k = int(min(self.k, similarity.shape[1]))
        # argpartition beats a full sort: only the top-k order matters.
        top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        rows = np.arange(similarity.shape[0])[:, None]
        weights = similarity[rows, top]
        neighbour_labels = self._y[top]

        total = weights.sum(axis=1)
        # A query with no overlap against any training molecule gets the plain
        # neighbour mean rather than a divide-by-zero.
        weighted = np.where(
            total > 0,
            (weights * neighbour_labels).sum(axis=1) / np.where(total > 0, total, 1.0),
            neighbour_labels.mean(axis=1),
        )
        return weighted


class KnnBaseline(BaseBioactivityBaseline):
    """B1. Tanimoto-weighted k-nearest-neighbour over ECFP4.

    This is the local-SAR baseline: when a test compound is close to training
    chemistry it is very strong, and the gap between it and a neural model on
    the cluster-OOD view is a direct read on how much genuine generalization the
    neural model adds.
    """

    model_id = "b1-ecfp4-knn"

    def __init__(self, *, k: int = 5, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(seed=seed, k=k, **kwargs)
        self.k = k

    def _make_estimator(self, n_train: int) -> Any:
        return _TanimotoKnn(
            k=min(self.k, max(1, n_train - 1)),
            n_structure_features=self.featurizer.n_features,
        )


# --------------------------------------------------------------------------
# B2: trees
# --------------------------------------------------------------------------

class RandomForestBaseline(BaseBioactivityBaseline):
    """B2a. Random forest on ECFP4 plus descriptors."""

    model_id = "b2a-ecfp4-rf"
    requires_descriptors = True

    def __init__(self, *, n_estimators: int = 300, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(seed=seed, n_estimators=n_estimators, **kwargs)
        self.n_estimators = n_estimators

    def _make_estimator(self, n_train: int) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=self.seed,
            n_jobs=-1,
        )


class LightGbmBaseline(BaseBioactivityBaseline):
    """B2b. LightGBM on ECFP4 plus descriptors -- usually the strongest B2."""

    model_id = "b2b-ecfp4-lightgbm"
    requires_descriptors = True

    def __init__(
        self,
        *,
        n_estimators: int = 600,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            seed=seed, n_estimators=n_estimators,
            learning_rate=learning_rate, num_leaves=num_leaves, **kwargs
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

    def _make_estimator(self, n_train: int) -> Any:
        import lightgbm as lgb

        # Leaf count has to fall with task size or the trees memorize; small
        # ChEMBL tasks are the common case, not the exception.
        num_leaves = min(self.num_leaves, max(4, n_train // 8))
        return lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=num_leaves,
            min_child_samples=max(5, n_train // 100),
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=1.0,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )


REGISTRY = {
    MedianBaseline.model_id: MedianBaseline,
    KnnBaseline.model_id: KnnBaseline,
    RandomForestBaseline.model_id: RandomForestBaseline,
    LightGbmBaseline.model_id: LightGbmBaseline,
}
