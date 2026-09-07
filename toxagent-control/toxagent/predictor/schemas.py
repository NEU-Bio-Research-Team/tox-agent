"""Validation of what the predictor returned.

A response that does not match this shape is a ``predictor_protocol_error``,
never a partially-understood payload that gets stored anyway. The models are
lenient about *additional* provenance keys — the predictor may learn to report
more — and strict about everything a claim can later cite.

Nothing here reshapes the payload. The validated response is stored losslessly;
these models exist to refuse, not to transform.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .contract import (
    APPLICABILITY_METHOD_PREFIX,
    APPLICABILITY_STATUSES,
    LABELS,
    TOX21_TASK_ORDER_VERSION,
    TOX21_TASKS,
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")


class HergPrediction(_Base):
    probability_blocker: float = Field(ge=0.0, le=1.0)
    label: Literal["blocker", "non_blocker"]
    threshold: float = Field(ge=0.0, le=1.0)
    threshold_source: str
    model_id: str


class ClinToxPrediction(_Base):
    probability_clinical_toxicity: float = Field(ge=0.0, le=1.0)
    label: Literal["positive", "negative"]
    threshold: float = Field(ge=0.0, le=1.0)
    threshold_source: str
    model_id: str


class Tox21Assay(_Base):
    probability_activity: float = Field(ge=0.0, le=1.0)
    active: bool
    threshold: float = Field(ge=0.0, le=1.0)
    threshold_source: str


class Tox21Prediction(_Base):
    task_order_version: str
    assays: dict[str, Tox21Assay]
    model_id: str

    @field_validator("task_order_version")
    @classmethod
    def _pinned_order(cls, value: str) -> str:
        if value != TOX21_TASK_ORDER_VERSION:
            raise ValueError(
                f"Tox21 task order version {value!r} is not the pinned "
                f"{TOX21_TASK_ORDER_VERSION!r}; the column-to-assay mapping may have moved "
                "and cannot be resolved at runtime"
            )
        return value

    @model_validator(mode="after")
    def _known_assays(self) -> "Tox21Prediction":
        unknown = sorted(set(self.assays) - set(TOX21_TASKS))
        if unknown:
            raise ValueError(f"unknown Tox21 assay(s): {unknown}")
        return self


class Applicability(_Base):
    status: Literal["ok", "limited", "out_of_domain"]
    method: str
    reasons: list[str] = []

    @field_validator("method")
    @classmethod
    def _rule_based(cls, value: str) -> str:
        """SCI-07. If the predictor ever ships a learned detector, the method
        string changes and this fails loudly — because the required limitation
        attached downstream ("rule based, not a distributional test") would
        become false, and a silent pass would carry that falsehood into answers.
        """
        if not value.startswith(APPLICABILITY_METHOD_PREFIX):
            raise ValueError(
                f"applicability method {value!r} is not the pinned "
                f"{APPLICABILITY_METHOD_PREFIX!r} family"
            )
        return value


class Predictions(_Base):
    clintox: ClinToxPrediction | None = None
    herg: HergPrediction | None = None
    tox21: Tox21Prediction | None = None


class PredictionResponse(_Base):
    input_smiles: str
    canonical_smiles: str
    predictions: Predictions
    applicability: Applicability
    provenance: dict[str, Any] = {}

    #: The bytes the predictor actually sent. Validation must not be able to
    #: change what gets stored: a model dump would add ``"clintox": null`` for an
    #: endpoint the predictor simply did not serve, and "present but null" is a
    #: different statement from "absent" once something reads it back.
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def parse_lossless(cls, payload: dict[str, Any]) -> "PredictionResponse":
        parsed = cls.model_validate(payload)
        parsed._raw = payload
        return parsed

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw or self.model_dump(mode="json", exclude_none=True)

    @model_validator(mode="after")
    def _no_cross_endpoint_leakage(self) -> "PredictionResponse":
        """SCI-01 at the boundary: a hERG payload carrying a clinical
        probability field, or the reverse, is refused rather than stored."""
        herg, clintox = self.predictions.herg, self.predictions.clintox
        if herg is not None and hasattr(herg, "probability_clinical_toxicity"):
            raise ValueError("hERG payload carries a clinical-toxicity field")
        if clintox is not None and hasattr(clintox, "probability_blocker"):
            raise ValueError("ClinTox payload carries an hERG blockade field")
        return self

    def served_endpoints(self) -> tuple[str, ...]:
        return tuple(
            name for name in ("clintox", "herg", "tox21")
            if getattr(self.predictions, name) is not None
        )


class BatchItemError(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    input_smiles: str
    error: str
    detail: str = ""


class BatchPredictionResponse(_Base):
    results: list[PredictionResponse]
    errors: list[BatchItemError] = []
    count: int


class AttributionTokenScore(_Base):
    token: str
    score: float


class AttributionResponse(_Base):
    """Attribution is an explanation of one endpoint/task, never a mechanism.

    ``status`` may be ``partial`` — the predictor reports a slow backward pass
    rather than silently returning a truncated result — and the control plane
    passes that through instead of presenting partial scores as complete.
    """

    status: Literal["completed", "partial", "failed"]
    input_smiles: str
    canonical_smiles: str
    endpoint: Literal["clintox", "herg", "tox21"]
    task: str | None = None
    probability: float | None = None
    tokens: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    error: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def _tox21_names_its_task(self) -> "AttributionResponse":
        if self.endpoint == "tox21" and not self.task:
            raise ValueError("a Tox21 attribution must name its assay (SCI-09)")
        if self.task and self.task not in TOX21_TASKS:
            raise ValueError(f"unknown Tox21 task {self.task!r}")
        return self


class ExplanationResponse(_Base):
    """Atom-level explanation from ToxPred ``POST /v1/explanations``.

    Passed through to the client near-verbatim (the control plane only appends
    the ``attribution_not_causality`` limitation). ``extra="allow"`` so a newer
    predictor field is carried, not dropped.
    """

    status: Literal["completed", "partial", "failed"]
    endpoint: Literal["clintox", "herg", "tox21"]
    task: str | None = None
    input_smiles: str
    canonical_smiles: str | None = None
    atom_order_version: str | None = None
    probability: float | None = None
    atoms: list[dict[str, Any]] = []
    unmapped_importance: float | None = None
    tokens: list[dict[str, Any]] = []
    method: str | None = None
    metadata: dict[str, Any] = {}


class ModelInfo(_Base):
    model_id: str
    capabilities: list[str] = []
    loaded: bool
    required: bool = True
    detail: str = ""
    blocked_reason: str | None = None


class ModelsResponse(_Base):
    models: list[ModelInfo]
    served_endpoints: list[str]


class ReadinessResponse(_Base):
    ready: bool
    reasons: list[str] = []
    served_endpoints: list[str] = []
