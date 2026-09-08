"""Split-conformal binary prediction sets with explicit target coverage."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ConformalArtifact:
    endpoint: str
    task: str | None
    target_coverage: float
    nonconformity_quantile: float
    calibration_split_sha256: str
    method: str = "split_conformal_binary_v1"

    def __post_init__(self) -> None:
        if not 0 < self.target_coverage < 1:
            raise ValueError("target coverage must lie in (0, 1)")
        if not 0 <= self.nonconformity_quantile <= 1:
            raise ValueError("nonconformity quantile must lie in [0, 1]")


def fit_binary_conformal(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    endpoint: str,
    task: str | None,
    target_coverage: float,
    calibration_split_sha256: str,
) -> ConformalArtifact:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("non-empty, aligned calibration probabilities and labels are required")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("binary conformal labels must be 0 or 1")
    scores = sorted(
        1 - (float(p) if y == 1 else 1 - float(p))
        for p, y in zip(probabilities, labels)
    )
    n = len(scores)
    rank = min(n, math.ceil((n + 1) * target_coverage))
    return ConformalArtifact(
        endpoint, task, target_coverage, scores[rank - 1], calibration_split_sha256
    )


def prediction_set(
    probability_positive: float, artifact: ConformalArtifact
) -> tuple[str, ...]:
    p = float(probability_positive)
    if not 0 <= p <= 1:
        raise ValueError("probability must lie in [0, 1]")
    threshold = artifact.nonconformity_quantile
    labels: list[str] = []
    if p <= threshold:
        labels.append("negative")
    if 1 - p <= threshold:
        labels.append("positive")
    return tuple(labels)
