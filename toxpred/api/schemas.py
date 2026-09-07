"""Request and response models for the v1 prediction API.

Requests forbid unknown fields. The legacy ``/predict`` accepted a field named
``threshold`` and silently discarded anything else, so a caller sending
``clinical_threshold`` got the default while believing they had set the
operating point. Here that is a 422 instead.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..domain.endpoints import TOX21_TASKS

EndpointName = Literal["clintox", "herg", "tox21"]


class ThresholdOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    herg: float | None = Field(default=None, ge=0.0, le=1.0)
    clintox: float | None = Field(default=None, ge=0.0, le=1.0)
    tox21: dict[str, float] | None = None

    @field_validator("tox21")
    @classmethod
    def _known_tasks_only(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        if value is None:
            return None
        unknown = sorted(set(value) - set(TOX21_TASKS))
        if unknown:
            raise ValueError(f"unknown Tox21 task(s): {unknown}")
        for task, threshold in value.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"threshold for {task} must lie in [0, 1], got {threshold}")
        return value


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smiles: str = Field(..., min_length=1, description="A single SMILES string.")
    endpoints: list[EndpointName] | None = Field(
        default=None,
        description="Endpoints to evaluate. Defaults to every endpoint this build serves.",
    )
    threshold_overrides: ThresholdOverrides | None = None


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smiles: list[str] = Field(..., min_length=1)
    endpoints: list[EndpointName] | None = None
    threshold_overrides: ThresholdOverrides | None = None


class AttributionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smiles: str = Field(..., min_length=1)
    endpoint: EndpointName = Field(
        ..., description="Attribution is always for one endpoint; it is never aggregated."
    )
    task: str | None = Field(
        default=None,
        description="Required when endpoint is 'tox21': which of the 12 assays to attribute.",
    )

    @field_validator("task")
    @classmethod
    def _known_task(cls, value: str | None) -> str | None:
        if value is not None and value not in TOX21_TASKS:
            raise ValueError(f"unknown Tox21 task: {value!r}")
        return value


class ExplainRequest(BaseModel):
    """``POST /v1/explanations`` (plan section 5.1).

    One assay per call: ``endpoint='tox21'`` requires ``task``. Running all 12
    is 12 backward passes and a combined tox21 attribution is meaningless.
    """

    model_config = ConfigDict(extra="forbid")

    smiles: str = Field(..., min_length=1)
    endpoint: Literal["herg", "tox21"]
    task: str | None = None

    @model_validator(mode="after")
    def _task_rules(self) -> "ExplainRequest":
        if self.endpoint == "tox21" and not self.task:
            raise ValueError(
                "attributing the tox21 endpoint requires a task; the twelve assays "
                "are independent and a combined attribution would not mean anything"
            )
        if self.endpoint != "tox21" and self.task is not None:
            raise ValueError(f"task is only meaningful for tox21, not {self.endpoint}")
        if self.task is not None and self.task not in TOX21_TASKS:
            raise ValueError(f"unknown Tox21 task: {self.task!r}")
        return self


# --- responses -------------------------------------------------------------
# Kept as plain dicts rather than mirrored models: the domain layer already
# owns the payload shape and its invariants, and a second definition here would
# be free to drift from it.

class ModelInfo(BaseModel):
    model_id: str
    capabilities: list[str]
    loaded: bool
    required: bool
    detail: str = ""
    blocked_reason: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    served_endpoints: list[str]


class ReadinessResponse(BaseModel):
    ready: bool
    reasons: list[str] = []
    served_endpoints: list[str] = []
    device: str = "cpu"


class BatchItemErrorOut(BaseModel):
    index: int
    input_smiles: str
    error: str
    detail: str


class BatchPredictionResponse(BaseModel):
    results: list[dict[str, Any]]
    errors: list[BatchItemErrorOut]
    count: int
