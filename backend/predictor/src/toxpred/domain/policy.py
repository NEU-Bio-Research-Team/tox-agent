"""Threshold policy.

The repository this replaces resolved a clinical threshold from three parallel
sources — ``CLINICAL_THRESHOLD`` env var, ``config/workspace_mode.yaml`` and a
per-request default — and never consulted the value the model was actually
calibrated at. The dual-head ChemBERTa artifact ships a hERG threshold of
0.4133 (Youden-J, 3-fold CV) and twelve per-task Tox21 thresholds; the running
service applied 0.30 to all of them.

Here the artifact is the only default. An override is permitted but must be
carried through to the response as ``threshold_source="request_override"`` so a
label can never be read without knowing which operating point produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .endpoints import TOX21_TASKS


class ThresholdSource(str, Enum):
    ARTIFACT = "artifact"
    """Calibrated on a validation split and shipped inside the model release."""

    MANIFEST_DECLARED = "manifest_declared"
    """Chosen operationally and declared in the manifest — NOT calibrated.

    Distinct from ARTIFACT on purpose. The ChemBERTa release ships a hERG
    threshold fitted by Youden-J over 3-fold CV; the ClinTox checkpoint ships no
    threshold at all, so any value used with it is a policy choice. Collapsing
    the two under one label is how 0.30 came to look like a calibrated number.
    """

    REQUEST_OVERRIDE = "request_override"
    """Supplied by the caller for this request only."""


POLICY_VERSION = "tox-policy-v1"


@dataclass(frozen=True, slots=True)
class ResolvedThreshold:
    value: float
    source: ThresholdSource

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"threshold must lie in [0, 1], got {self.value}")


@dataclass(frozen=True)
class PredictionPolicySnapshot:
    """Immutable resolved policy for one request.

    Built once per request and passed down; no module below reads configuration
    or the environment again.
    """

    policy_version: str
    herg_threshold: ResolvedThreshold
    tox21_thresholds: Mapping[str, ResolvedThreshold]
    clintox_threshold: ResolvedThreshold | None = None
    _frozen_tasks: tuple[str, ...] = field(default=TOX21_TASKS, repr=False)

    def __post_init__(self) -> None:
        missing = set(self._frozen_tasks) - set(self.tox21_thresholds)
        if missing:
            raise ValueError(f"missing Tox21 thresholds for: {sorted(missing)}")
        extra = set(self.tox21_thresholds) - set(self._frozen_tasks)
        if extra:
            raise ValueError(f"unknown Tox21 tasks in thresholds: {sorted(extra)}")

    @classmethod
    def from_artifact(
        cls,
        *,
        herg_threshold: float,
        tox21_thresholds: Mapping[str, float],
        clintox_threshold: float | None = None,
        clintox_threshold_source: ThresholdSource = ThresholdSource.MANIFEST_DECLARED,
        herg_override: float | None = None,
        tox21_override: Mapping[str, float] | None = None,
        clintox_override: float | None = None,
    ) -> "PredictionPolicySnapshot":
        def resolve(
            artifact_value: float,
            override: float | None,
            default_source: ThresholdSource = ThresholdSource.ARTIFACT,
        ) -> ResolvedThreshold:
            if override is None:
                return ResolvedThreshold(float(artifact_value), default_source)
            return ResolvedThreshold(float(override), ThresholdSource.REQUEST_OVERRIDE)

        overrides = dict(tox21_override or {})
        unknown = set(overrides) - set(TOX21_TASKS)
        if unknown:
            raise ValueError(f"override for unknown Tox21 task(s): {sorted(unknown)}")

        absent = [t for t in TOX21_TASKS if t not in tox21_thresholds]
        if absent:
            raise ValueError(f"artifact is missing Tox21 thresholds for: {absent}")

        tox21 = {
            task: resolve(tox21_thresholds[task], overrides.get(task))
            for task in TOX21_TASKS
        }
        return cls(
            policy_version=POLICY_VERSION,
            herg_threshold=resolve(herg_threshold, herg_override),
            tox21_thresholds=MappingProxyType(tox21),
            clintox_threshold=(
                None if clintox_threshold is None
                else resolve(clintox_threshold, clintox_override, clintox_threshold_source)
            ),
        )


def apply_threshold(probability: float, threshold: ResolvedThreshold) -> bool:
    """Positive iff probability >= threshold.

    The boundary is inclusive and is asserted by a unit test, because the
    previous implementation used ``>=`` in one place and ``>`` in another.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must lie in [0, 1], got {probability}")
    return probability >= threshold.value
