"""Strict wire contract expected from a future Activity Predictor service.

There is intentionally no route or network client yet.  These models let the
control plane and synthetic tests agree on semantics without claiming that a
SAIR model or artifact is available.
"""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActivityTargetRequest(_Strict):
    target_id: str = Field(min_length=1)
    uniprot_id: str | None = Field(default=None, min_length=1)


class ActivityPredictionRequest(_Strict):
    """A request names the assay context it wants a pIC50 for.

    The response carries a required ``assay_context``, so leaving it out of
    the request meant the server picked one the caller never chose and could
    not check.  A threshold is calibrated per context, so the caller has to
    state it and the application boundary compares it against the response.
    """

    smiles: str = Field(min_length=1, max_length=4000)
    target: ActivityTargetRequest
    assay_context: str = Field(min_length=1)
    model_id: str | None = Field(default=None, min_length=1)


class ActivityTarget(_Strict):
    target_id: str = Field(min_length=1)
    uniprot_id: str | None = None
    sequence_sha256: str | None = None

    @field_validator("sequence_sha256")
    @classmethod
    def _digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.lower()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("sequence_sha256 must be a 64-character hexadecimal digest")
        return value


class ActivityEstimatePayload(_Strict):
    predicted_pic50: float
    uncertainty: float | None = Field(default=None, ge=0)
    prediction_interval: tuple[float, float] | None = None
    model_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_numbers(self) -> "ActivityEstimatePayload":
        values = [self.predicted_pic50]
        if self.uncertainty is not None:
            values.append(self.uncertainty)
        if self.prediction_interval is not None:
            low, high = self.prediction_interval
            values.extend((low, high))
            if low > high:
                raise ValueError("prediction interval lower bound exceeds upper bound")
            if not low <= self.predicted_pic50 <= high:
                raise ValueError("prediction interval must contain predicted_pic50")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("activity estimate values must be finite")
        return self


ApplicabilityValue = Literal["in_domain", "limited", "out_of_domain", "unknown"]


class ActivityApplicabilityPayload(_Strict):
    ligand: ApplicabilityValue
    target: ApplicabilityValue
    joint: ApplicabilityValue
    reasons: list[str] = Field(default_factory=list)


class ActivityProvenancePayload(_Strict):
    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    split_id: str | None = None
    artifact_hashes: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class ActivityPredictionResponse(_Strict):
    input_smiles: str = Field(min_length=1)
    canonical_smiles: str = Field(min_length=1)
    target: ActivityTarget
    assay_context: str = Field(min_length=1)
    prediction: ActivityEstimatePayload
    applicability: ActivityApplicabilityPayload
    provenance: ActivityProvenancePayload

    @model_validator(mode="after")
    def _same_model(self) -> "ActivityPredictionResponse":
        if self.prediction.model_id != self.provenance.model_id:
            raise ValueError("prediction and provenance model_id must match")
        return self
