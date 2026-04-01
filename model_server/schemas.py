# model_server/schemas.py
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    threshold: float = Field(0.5, ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    smiles: str
    canonical_smiles: Optional[str]
    p_toxic: float
    label: str 
    confidence: float
    threshold_used: float

class BatchPredictRequest(BaseModel):
    smiles_list: List[str]
    threshold: float = 0.5

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
    explainer_note: str


class AnalyzeRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    clinical_threshold: float = Field(0.5, ge=0.0, le=1.0)
    mechanism_threshold: float = Field(0.5, ge=0.0, le=1.0)
    return_all_scores: bool = True
    explain_only_if_alert: bool = True
    explainer_epochs: int = Field(200, ge=50, le=500)
    explainer_timeout_ms: int = Field(30000, ge=1000, le=300000)
    target_task: Optional[str] = None


class AnalyzeResponse(BaseModel):
    smiles: str
    canonical_smiles: Optional[str]
    clinical: ClinicalToxicityOutput
    mechanism: MechanismToxicityOutput
    explanation: Optional[ToxicityExplanationOutput]
    final_verdict: str


class AgentAnalyzeRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session id. If omitted, a new one is generated.",
    )
    user_id: str = Field(default="default_user", description="Logical user id")
    max_literature_results: int = Field(default=5, ge=1, le=20)
    include_agent_events: bool = Field(
        default=True,
        description="Include agent/tool calling event trace for debugging",
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
    adk_available: bool
    validation_status: Optional[str] = None
    final_report: Dict[str, Any] = Field(default_factory=dict)
    final_text: Optional[str] = None
    agent_events: List[AgentEventRecord] = Field(default_factory=list)
    state_keys: List[str] = Field(default_factory=list)

