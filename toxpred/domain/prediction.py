"""Typed prediction results.

The central invariant of this module: a hERG probability cannot be serialised
under a ``clinical`` key. In the code this replaces, ``backend/inference.py``
took ``torch.sigmoid(head_outputs["herg_logits"])`` and emitted it as
``clinical.p_toxic`` with a "clinical" threshold — an hERG cardiotoxicity
probability presented as clinical-trial toxicity. Each endpoint here is a
separate frozen type with its own field name, and ``PredictionResult`` builds
its payload from those types alone, so the substitution has nowhere to happen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .endpoints import TOX21_TASK_ORDER_VERSION, TOX21_TASKS, Endpoint
from .policy import ResolvedThreshold, apply_threshold


def _probability(value: float, field: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1], got {value}")
    return value


@dataclass(frozen=True, slots=True)
class ClinToxPrediction:
    """Clinical-trial toxicity. Only a ClinTox-trained model may produce this."""

    probability_clinical_toxicity: float
    threshold: ResolvedThreshold
    model_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probability_clinical_toxicity",
            _probability(self.probability_clinical_toxicity, "probability_clinical_toxicity"),
        )

    @property
    def label(self) -> str:
        return "positive" if apply_threshold(
            self.probability_clinical_toxicity, self.threshold) else "negative"

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability_clinical_toxicity": self.probability_clinical_toxicity,
            "label": self.label,
            "threshold": self.threshold.value,
            "threshold_source": self.threshold.source.value,
            "model_id": self.model_id,
        }


@dataclass(frozen=True, slots=True)
class HergPrediction:
    """hERG channel blockade liability. Never a clinical-toxicity statement."""

    probability_blocker: float
    threshold: ResolvedThreshold
    model_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probability_blocker",
            _probability(self.probability_blocker, "probability_blocker"),
        )

    @property
    def label(self) -> str:
        return "blocker" if apply_threshold(
            self.probability_blocker, self.threshold) else "non_blocker"

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability_blocker": self.probability_blocker,
            "label": self.label,
            "threshold": self.threshold.value,
            "threshold_source": self.threshold.source.value,
            "model_id": self.model_id,
        }


@dataclass(frozen=True, slots=True)
class Tox21AssayPrediction:
    task: str
    probability_activity: float
    threshold: ResolvedThreshold

    def __post_init__(self) -> None:
        if self.task not in TOX21_TASKS:
            raise ValueError(f"unknown Tox21 task: {self.task!r}")
        object.__setattr__(
            self, "probability_activity",
            _probability(self.probability_activity, "probability_activity"),
        )

    @property
    def active(self) -> bool:
        return apply_threshold(self.probability_activity, self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability_activity": self.probability_activity,
            "active": self.active,
            "threshold": self.threshold.value,
            "threshold_source": self.threshold.source.value,
        }


@dataclass(frozen=True, slots=True)
class Tox21Prediction:
    """Twelve independent assay activities. Deliberately not summed into a score.

    The replaced code exposed ``assay_hits`` and a ``mechanistic_alert`` boolean,
    which invited counting hits across chemically unrelated assays as if it were
    a severity measure. Consumers that want a count can count.
    """

    assays: tuple[Tox21AssayPrediction, ...]
    model_id: str

    def __post_init__(self) -> None:
        order = tuple(a.task for a in self.assays)
        if order != TOX21_TASKS:
            raise ValueError(
                f"Tox21 assays must be in {TOX21_TASK_ORDER_VERSION} order.\n"
                f"  got:      {order}\n  expected: {TOX21_TASKS}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_order_version": TOX21_TASK_ORDER_VERSION,
            "assays": {a.task: a.to_dict() for a in self.assays},
            "model_id": self.model_id,
        }


@dataclass(frozen=True, slots=True)
class ApplicabilityAssessment:
    """Rule-based input-domain check. Not a learned OOD detector.

    ``method`` is spelled out in the payload so a reader cannot mistake an
    element whitelist for a distributional test.
    """

    status: str
    method: str
    reasons: tuple[str, ...] = ()

    _ALLOWED = ("ok", "limited", "out_of_domain")

    def __post_init__(self) -> None:
        if self.status not in self._ALLOWED:
            raise ValueError(f"status must be one of {self._ALLOWED}, got {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "method": self.method, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class PredictionResult:
    input_smiles: str
    canonical_smiles: str
    applicability: ApplicabilityAssessment
    provenance: Mapping[str, Any]
    clintox: ClinToxPrediction | None = None
    herg: HergPrediction | None = None
    tox21: Tox21Prediction | None = None

    def to_dict(self) -> dict[str, Any]:
        predictions: dict[str, Any] = {}
        if self.clintox is not None:
            predictions[Endpoint.CLINTOX.value] = self.clintox.to_dict()
        if self.herg is not None:
            predictions[Endpoint.HERG.value] = self.herg.to_dict()
        if self.tox21 is not None:
            predictions[Endpoint.TOX21.value] = self.tox21.to_dict()
        return {
            "input_smiles": self.input_smiles,
            "canonical_smiles": self.canonical_smiles,
            "predictions": predictions,
            "applicability": self.applicability.to_dict(),
            "provenance": dict(self.provenance),
        }
