# model_server/schemas.py
import os
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

from backend.workspace_mode import resolve_default_clinical_threshold


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


DEFAULT_CLINICAL_THRESHOLD = _env_float("CLINICAL_THRESHOLD", 0.35)
DEFAULT_MECHANISM_THRESHOLD = _env_float("MECHANISM_THRESHOLD", 0.5)

try:
    DEFAULT_CLINICAL_THRESHOLD = float(resolve_default_clinical_threshold())
except Exception:
    # Keep env fallback if workspace config cannot be loaded.
    DEFAULT_CLINICAL_THRESHOLD = _env_float("CLINICAL_THRESHOLD", 0.35)

class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    threshold: float = Field(DEFAULT_CLINICAL_THRESHOLD, ge=0.0, le=1.0)
    inference_backend: str = Field(
        default="xsmiles",
        description="Inference backend: xsmiles, chemberta, pubchem, or molformer",
    )

class PredictResponse(BaseModel):
    smiles: str
    canonical_smiles: Optional[str]
    p_toxic: float
    label: str 
    confidence: float
    threshold_used: float

class BatchPredictRequest(BaseModel):
    smiles_list: List[str]
    threshold: float = DEFAULT_CLINICAL_THRESHOLD
    inference_backend: str = Field(
        default="xsmiles",
        description="Inference backend: xsmiles, chemberta, pubchem, or molformer",
    )

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    n_toxic: int
    n_non_toxic: int
    n_errors: int

class ExplainRequest(BaseModel):
    smiles: str
    epochs: int = Field(200, ge=50, le=500)
    explainer_timeout_ms: int = Field(30000, ge=1000, le=300000)
    target_class: Optional[int] = None  # None = auto-detect from prediction

class AtomImportance(BaseModel):
    atom_idx: int
    element: str
    importance: float
    is_in_ring: bool
    is_aromatic: Optional[bool] = None

class BondImportance(BaseModel):
    bond_idx: int
    atom_pair: str
    bond_type: str
    importance: float

class ExplainResponse(BaseModel):
    smiles: str
    p_toxic: float
    label: str
    top_atoms: List[AtomImportance]
    top_bonds: List[BondImportance]
    heatmap_base64: str     # PNG image encoded as base64
    molecule_png_base64: Optional[str] = None
    chemical_interpretation: str
    explainer_note: str     # Document known limitations


class ClinicalToxicityOutput(BaseModel):
    label: str
    is_toxic: bool
    confidence: float
    p_toxic: float
    threshold_used: float


class MechanismToxicityOutput(BaseModel):
    task_scores: Dict[str, float]
    active_tasks: List[str]
    highest_risk_task: str
    highest_risk_score: float
    assay_hits: int
    threshold_used: float
    task_thresholds: Dict[str, float]


class ToxicityExplanationOutput(BaseModel):
    target_task: str
    target_task_score: float
    top_atoms: List[AtomImportance]
    top_bonds: List[BondImportance]
    heatmap_base64: Optional[str] = None
    molecule_png_base64: Optional[str] = None
    explainer_note: str


class AnalyzeRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    clinical_threshold: float = Field(DEFAULT_CLINICAL_THRESHOLD, ge=0.0, le=1.0)
    mechanism_threshold: float = Field(DEFAULT_MECHANISM_THRESHOLD, ge=0.0, le=1.0)
    inference_backend: str = Field(
        default="xsmiles",
        description="Inference backend: xsmiles, chemberta, pubchem, or molformer",
    )
    return_all_scores: bool = True
    explain_only_if_alert: bool = True
    explainer_epochs: int = Field(200, ge=50, le=500)
    explainer_timeout_ms: int = Field(30000, ge=1000, le=300000)
    target_task: Optional[str] = None
    binary_tox_model: str = Field(
        default="pretrained_2head_herg_chemberta_model",
        description="Model key for binary toxicity prediction (e.g. dualhead_ensemble6_simple, dualhead_ensemble3_weighted, dualhead_ensemble3_simple, dualhead_ensemble5_simple, pretrained_2head_herg_chemberta_model)",
    )
    tox_type_model: str = Field(
        default="tox21_ensemble_3_best",
        description="Model key for toxicity-type prediction (e.g. dualhead_ensemble6_simple, dualhead_ensemble3_weighted, dualhead_ensemble3_simple, dualhead_ensemble5_simple, tox21_pretrained_gin_model, tox21_gatv2_model)",
    )


class OodAssessmentOutput(BaseModel):
    ood_risk: str
    flag: bool
    reason: str
    rare_elements: List[str] = Field(default_factory=list)
    high_risk_elements: List[str] = Field(default_factory=list)
    recommendation: Optional[str] = None


class InferenceContextOutput(BaseModel):
    workspace_mode: str
    inference_backend: str = Field(default="xsmiles")
    inference_backend_loaded: bool = False
    threshold_policy: Optional[str] = None
    clinical_threshold_applied: Optional[float] = None
    clinical_model_loaded: bool
    tox21_model_loaded: bool
    explainer_used: bool
    explanation_available: bool
    tox21_threshold_source: Optional[str] = None
    clinical_reference_metrics: Dict[str, float] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    smiles: str
    canonical_smiles: Optional[str]
    clinical: ClinicalToxicityOutput
    mechanism: MechanismToxicityOutput
    explanation: Optional[ToxicityExplanationOutput]
    ood_assessment: OodAssessmentOutput
    reliability_warning: Optional[str] = None
    inference_context: InferenceContextOutput
    final_verdict: str


class AgentAnalyzeRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session id. If omitted, a new one is generated.",
    )
    user_id: str = Field(default="default_user", description="Logical user id")
    max_literature_results: int = Field(default=5, ge=1, le=20)
    clinical_threshold: float = Field(DEFAULT_CLINICAL_THRESHOLD, ge=0.0, le=1.0)
    mechanism_threshold: float = Field(DEFAULT_MECHANISM_THRESHOLD, ge=0.0, le=1.0)
    inference_backend: str = Field(
        default="xsmiles",
        description="Inference backend for screening: xsmiles, chemberta, pubchem, or molformer",
    )
    binary_tox_model: str = Field(
        default="pretrained_2head_herg_chemberta_model",
        description="Model key for binary toxicity prediction (e.g. dualhead_ensemble6_simple, dualhead_ensemble3_weighted, dualhead_ensemble3_simple, dualhead_ensemble5_simple, pretrained_2head_herg_chemberta_model)",
    )
    tox_type_model: str = Field(
        default="tox21_ensemble_3_best",
        description="Model key for toxicity-type prediction (e.g. dualhead_ensemble6_simple, dualhead_ensemble3_weighted, dualhead_ensemble3_simple, dualhead_ensemble5_simple, tox21_pretrained_gin_model, tox21_gatv2_model)",
    )
    include_agent_events: bool = Field(
        default=True,
        description="Include agent/tool calling event trace for debugging",
    )
    language: str = Field(
        default="vi",
        description="Report language: vi or en",
    )
    molrag_enabled: bool = Field(
        default=False,
        description="Enable MolRAG retrieval + reasoning augmentation in screening output.",
    )
    molrag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Top-k similar molecules retrieved for MolRAG evidence.",
    )
    molrag_min_similarity: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum Tanimoto similarity threshold for MolRAG retrieval.",
    )


class AgentEventRecord(BaseModel):
    type: Optional[str] = None
    author: Optional[str] = None
    function_calls: List[Dict[str, Any]] = Field(default_factory=list)
    function_responses: List[Dict[str, Any]] = Field(default_factory=list)
    is_final: bool = False
    text_preview: Optional[str] = None


class AgentAnalyzeResponse(BaseModel):
    session_id: str
    chat_session_id: Optional[str] = Field(
        default=None,
        description="Report-chat session id used by /agent/chat for follow-up QA.",
    )
    adk_available: bool
    runtime_mode: str = Field(
        default="adk",
        description="Runtime path used by /agent/analyze: adk or deterministic_fallback",
    )
    runtime_note: Optional[str] = Field(
        default=None,
        description="Optional note describing fallback cause when runtime_mode is deterministic_fallback",
    )
    validation_status: Optional[str] = None
    final_report: Dict[str, Any] = Field(default_factory=dict)
    evidence_qa_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Evidence quality payload used by report chat for confidence calibration.",
    )
    final_text: Optional[str] = None
    agent_events: List[AgentEventRecord] = Field(default_factory=list)
    state_keys: List[str] = Field(default_factory=list)


class AgentChatRequest(BaseModel):
    message: str = Field(..., description="User message for report-level QA chat.")
    chat_session_id: Optional[str] = Field(
        default=None,
        description="Session id returned by /agent/analyze as chat_session_id.",
    )
    analysis_session_id: Optional[str] = Field(
        default=None,
        description="Fallback analyze session id to resolve chat session mapping on server.",
    )
    report_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional report grounding payload used to rehydrate a chat session when the runtime "
            "is load-balanced across stateless instances."
        ),
    )


class AgentChatResponse(BaseModel):
    chat_session_id: str
    response: str
