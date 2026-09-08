"""Wire models for the product API (plan section 6).

Requests forbid unknown fields. A caller who misspells ``threshold_overrides``
gets a 400 rather than the default operating point plus the belief they changed
it — the same lesson the predictor's own request models encode.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..predictor.contract import TOX21_TASKS


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSessionRequest(_Request):
    preferred_language: Literal["vi", "en"] = "en"
    title: str | None = Field(default=None, max_length=200)
    client_session_id: str | None = Field(default=None, max_length=255)


class UpdateSessionRequest(_Request):
    title: str = Field(min_length=1, max_length=120)
    expected_version: int = Field(ge=1)


class SessionSettingsRequest(_Request):
    ai_profile_id: str | None = Field(default=None, max_length=40)
    predictor_bindings: dict[Literal["clintox", "herg", "tox21"], str] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    status: str
    preferred_language: str
    title: str | None = None
    created_at: str
    version: int
    title_source: str | None = None
    title_status: str = "pending"


class TextPart(_Request):
    type: Literal["text"] = "text"
    text: str


class MoleculeInput(_Request):
    smiles: str | None = Field(default=None, min_length=1, max_length=4000)
    batch_smiles: list[str] | None = Field(default=None, max_length=256)


class ImageInput(_Request):
    """A structure-recognition attempt: image in, SMILES out through the
    ``toxocr`` service (ADR 0006), then the same deterministic analysis
    pipeline a typed SMILES already goes through. When no OCR service is
    configured, ``SubmitMessage`` answers ``capability_unavailable`` instead —
    see ``SubmitMessage.structure_recognition_available``. ``data_base64`` is
    decoded and size-checked at the route boundary (``routes.py``); only its
    byte count survives past that point."""

    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    data_base64: str = Field(min_length=1)


class AnalysisOptions(_Request):
    endpoints: list[Literal["clintox", "herg", "tox21"]] | None = None
    model_selection: dict[Literal["clintox", "herg", "tox21"], str] | None = None
    threshold_overrides: dict[str, Any] | None = None
    include_attribution: bool = False
    explanation_mode: Literal["required", "on_demand", "none"] = "on_demand"
    explanation_targets: list["ExplanationTarget"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _explanation_rules(self) -> "AnalysisOptions":
        targets = {(target.endpoint, target.task) for target in self.explanation_targets}
        if len(targets) != len(self.explanation_targets):
            raise ValueError("explanation targets must be unique")
        if self.explanation_mode == "required" and "tox21" in (self.endpoints or []) and not any(t.endpoint == "tox21" for t in self.explanation_targets):
            raise ValueError("required Tox21 explanations need one explicitly selected assay")
        return self


class ExplanationTarget(_Request):
    endpoint: Literal["herg", "tox21"]
    task: str | None = None

    @model_validator(mode="after")
    def _task_rules(self) -> "ExplanationTarget":
        if self.endpoint == "tox21" and self.task not in TOX21_TASKS:
            raise ValueError("a Tox21 explanation target requires a known assay")
        if self.endpoint == "herg" and self.task is not None:
            raise ValueError("task is only meaningful for Tox21")
        return self


class SendMessageRequest(_Request):
    client_message_id: str | None = Field(default=None, max_length=255)
    content: list[TextPart] = Field(default_factory=list)
    intent_hint: Literal[
        "auto", "analyze", "ask_report", "research_evidence", "request_attribution"
    ] = "auto"
    molecule: MoleculeInput | None = None
    image: ImageInput | None = None
    analysis_options: AnalysisOptions | None = None
    analysis_id: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(part.text for part in self.content).strip()


class PredictRequest(_Request):
    """Body for the stateless ``POST /v1/predict`` (plan section 3.1).

    ``extra="forbid"`` is inherited: a misspelled ``threshold_overrides`` is a
    400, never the default operating point plus the belief it was changed.
    """

    smiles: str = Field(min_length=1, max_length=4000)
    endpoints: list[Literal["clintox", "herg", "tox21"]] | None = None
    model_selection: dict[Literal["clintox", "herg", "tox21"], str] | None = None
    threshold_overrides: dict[str, Any] | None = None
    #: Convenience for API callers: additionally attach per-endpoint token
    #: attributions. The UI should prefer the explicit ``POST /v1/predict/explain``
    #: call. Tox21 is skipped here because it needs a named assay.
    include_attribution: bool = False
    explanation_mode: Literal["required", "on_demand", "none"] = "on_demand"
    explanation_targets: list[ExplanationTarget] = Field(default_factory=list)


class PredictBatchRequest(_Request):
    """Body for ``POST /v1/predict:batch``. The length cap is enforced by the
    limiter (422 when over ``PredictorSettings.max_batch_size``)."""

    smiles: list[str] = Field(min_length=1)
    endpoints: list[Literal["clintox", "herg", "tox21"]] | None = None
    model_selection: dict[Literal["clintox", "herg", "tox21"], str] | None = None
    threshold_overrides: dict[str, Any] | None = None
    explanation_mode: Literal["required", "on_demand", "none"] = "none"
    explanation_targets: list[ExplanationTarget] = Field(default_factory=list)


class ExplainRequest(_Request):
    """Body for ``POST /v1/predict/explain`` (plan section 5.2).

    Mirrors the ToxPred ``/v1/explanations`` contract so the same request is
    refused at this boundary rather than one hop later: ``endpoint='tox21'``
    needs a ``task``, and ``task`` is meaningless for hERG.
    """

    smiles: str = Field(min_length=1, max_length=4000)
    endpoint: Literal["herg", "tox21"]
    task: str | None = None

    @model_validator(mode="after")
    def _task_rules(self) -> "ExplainRequest":
        if self.endpoint == "tox21" and not self.task:
            raise ValueError(
                "attributing tox21 requires a task; the twelve assays are independent"
            )
        if self.endpoint != "tox21" and self.task is not None:
            raise ValueError("task is only meaningful for tox21")
        if self.task is not None and self.task not in TOX21_TASKS:
            raise ValueError(f"unknown Tox21 task: {self.task!r}")
        return self


class RecognizeRequest(_Request):
    """Body for the stateless ``POST /v1/predict/recognize`` (plan section 4.1).

    Same wire shape as ``ImageInput``, but the proxy never persists the bytes:
    they are decoded, size/mime/magic-byte checked, passed to OCR, discarded.
    """

    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    data_base64: str = Field(min_length=1)


class RecognizedStructure(BaseModel):
    smiles: str
    canonical_smiles: str
    confidence: float | None = None


class AcceptedResponse(BaseModel):
    message_id: str
    run_id: str
    run_status: str
    selected_intent: str
    lane: str
    events_url: str
    clarification: dict[str, Any] | None = None
    duplicate_of_message_id: str | None = None


class CancelResponse(BaseModel):
    run_id: str
    requested: bool
    runtime_cancel_supported: bool
    action: str


class CreateModelConnectionRequest(_Request):
    provider_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=128)
    display_name: str | None = Field(default=None, max_length=160)
    auth_mode: Literal["chatgpt_subscription", "api_key", "local", "none"]
    base_url: str | None = Field(default=None, max_length=2048)
    credential: str | None = Field(default=None, min_length=1, max_length=16_384)

    @model_validator(mode="after")
    def _credential_rules(self) -> "CreateModelConnectionRequest":
        if self.auth_mode == "api_key" and not self.credential:
            raise ValueError("api_key auth requires credential")
        if self.auth_mode in {"local", "none"} and self.credential:
            raise ValueError("local/none auth cannot include credential")
        return self
