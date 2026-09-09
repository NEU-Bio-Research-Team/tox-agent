"""Frozen embedding-space applicability estimators."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EmbeddingDomainScore:
    knn_distance: float
    mahalanobis_distance: float
    density_proxy: float
    reference_sha256: str
    method: str = "chemberta_embedding_ad_v1"


class EmbeddingDomain:
    def __init__(self, reference: np.ndarray, *, k: int = 5, regularization: float = 1e-5) -> None:
        matrix = np.asarray(reference, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
            raise ValueError("reference must be a finite 2D matrix with at least two rows")
        self._reference = matrix
        self._k = max(1, min(int(k), matrix.shape[0]))
        self._mean = matrix.mean(axis=0)
        covariance = np.cov(matrix, rowvar=False)
        covariance = np.atleast_2d(covariance) + np.eye(matrix.shape[1]) * regularization
        self._precision = np.linalg.pinv(covariance)
        self.reference_sha256 = hashlib.sha256(matrix.tobytes(order="C")).hexdigest()

    def score(self, embedding: np.ndarray) -> EmbeddingDomainScore:
        vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if vector.shape[0] != self._reference.shape[1] or not np.isfinite(vector).all():
            raise ValueError("embedding dimension/value mismatch")
        distances = np.linalg.norm(self._reference - vector, axis=1)
        nearest = np.partition(distances, self._k - 1)[: self._k]
        delta = vector - self._mean
        mahalanobis = float(np.sqrt(max(0.0, delta @ self._precision @ delta)))
        knn = float(nearest.mean())
        return EmbeddingDomainScore(
            knn_distance=knn,
            mahalanobis_distance=mahalanobis,
            density_proxy=float(1.0 / (1.0 + knn)),
            reference_sha256=self.reference_sha256,
        )
