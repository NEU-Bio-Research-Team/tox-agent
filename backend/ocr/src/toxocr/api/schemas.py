from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StructureRecognitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    data_base64: str = Field(min_length=1)


class StructureRecognitionResponse(BaseModel):
    smiles: str
    canonical_smiles: str
    confidence: float | None = None


class ReadinessResponse(BaseModel):
    ready: bool
    device: str = "unloaded"
    checkpoint_fingerprint: str | None = None
