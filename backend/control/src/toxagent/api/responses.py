"""Response shapes of the product API, as the OpenAPI schema publishes them.

The frontend's types are generated from this schema (``make openapi``), so a
field renamed here is a type error in the browser code rather than a silent
``undefined``.

These models *document* responses: routes declare them through
``responses={200: {"model": ...}}`` and keep returning plain dicts, so a
mismatch can never turn a working response into a 500 in production. The
contract is enforced where it is cheap to fail — ``tests/conftest.py``
validates every JSON response the test suite produces against the model its
route declares, with unknown fields refused unless a model says otherwise.

Conventions: a field that is always present is required; ``X | None`` without
a default when it can be null. A field that only some responses carry has a
default: ``x: X | None = None`` when it may also be null, ``x: X = None`` when
it is either absent or a real value, never null. ``extra="allow"`` marks the
few shapes that are open by design.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Endpoint = Literal["clintox", "herg", "tox21"]
RunStatus = Literal["queued", "running", "validating", "completed", "failed", "cancelled"]
Lane = Literal["deterministic", "agentic", "mixed"]
PreferredLanguage = Literal["vi", "en"]


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Open(BaseModel):
    """A shape that carries fields beyond the ones named here, by design."""

    model_config = ConfigDict(extra="allow")


# -- sessions --------------------------------------------------------------------


class RunSummary(_Response):
    run_id: str
    status: RunStatus
    intent: str


class SessionListRow(_Response):
    session_id: str
    title: str | None
    status: str
    preferred_language: PreferredLanguage
    created_at: str
    updated_at: str
    active_run: RunSummary | None
    run_count: int
    last_message_preview: str | None
    title_source: Literal["deterministic", "model", "manual"] | None = None


class SessionListResponse(_Response):
    sessions: list[SessionListRow]
    next_offset: int | None


class ConfigurationSnapshot(_Open):
    ai_profile_id: str | None
    predictor_bindings: dict[str, str]
    created_at: str


class RunProjection(_Response):
    run_id: str
    status: RunStatus
    lane: Lane
    intent: str
    trigger_message_id: str
    runtime_binding_id: str | None
    recovery_of_run_id: str | None
    failure_code: str | None
    potentially_billed: bool
    deadline_at: str
    created_at: str
    started_at: str | None
    ended_at: str | None
    configuration_snapshot: ConfigurationSnapshot | None = None


class SessionSettings(_Response):
    ai_profile_id: str | None
    predictor_bindings: dict[str, str]


# -- analysis --------------------------------------------------------------------


class Applicability(_Open):
    status: str
    method: str
    reasons: list[str]


class AnalysisProvenance(_Open):
    content_sha256: str


class HergSection(_Open):
    measurement: str
    label: str
    threshold: float
    threshold_source: str
    model_id: str
    probability_blocker: float


class ClintoxSection(_Open):
    measurement: str
    label: str
    threshold: float
    threshold_source: str
    model_id: str
    probability_clinical_toxicity: float


class Tox21Assay(_Open):
    probability_activity: float
    active: bool
    threshold: float
    threshold_source: str


class Tox21Section(_Open):
    measurement: str
    task_order_version: str
    model_id: str
    #: A mapping, never a count (SCI-05).
    assays: dict[str, Tox21Assay]


class AnalysisSections(_Response):
    herg: HergSection | None = None
    clintox: ClintoxSection | None = None
    tox21: Tox21Section | None = None


class _AnalysisBody(_Response):
    input_smiles: str
    canonical_smiles: str
    requested_endpoints: list[Endpoint]
    served_endpoints: list[Endpoint]
    unavailable_endpoints: list[Endpoint]
    sections: AnalysisSections
    applicability: Applicability
    provenance: AnalysisProvenance
    policy_snapshot: dict[str, Any]
    required_limitations: list[str]
    created_at: str


class AnalysisProjection(_AnalysisBody):
    analysis_id: str


class QuickPredictResult(_AnalysisBody):
    """``POST /v1/predict``: the session path's analysis shape, never persisted."""

    analysis_id: None
    persisted: Literal[False]
    attributions: list[Any] = None


class QuickPredictBatchError(_Response):
    index: int
    input_smiles: str
    error: str
    detail: str


class QuickPredictBatchResult(_Response):
    results: list[QuickPredictResult]
    errors: list[QuickPredictBatchError]
    count: int


class QuickPredictComparison(_Response):
    endpoint: Endpoint
    model_id: str
    result: QuickPredictResult


class QuickPredictCompareResult(_Response):
    persisted: Literal[False]
    input_smiles: str
    comparisons: list[QuickPredictComparison]


class PredictModelInfo(_Open):
    model_id: str
    capabilities: list[str]
    loaded: bool
    required: bool
    detail: str
    blocked_reason: str | None


class PredictEndpointCapability(_Open):
    id: Endpoint
    display_name: str
    enabled: bool
    model_id: str | None
    supports_explanation: bool
    explanation_target_required: bool
    tasks: list[str]
    blocked_reason: str | None
    models: list[PredictModelInfo] = None


class PredictCapabilities(_Open):
    capability_version: str = None
    default_endpoints: list[Endpoint] = None
    served_endpoints: list[Endpoint]
    endpoints: list[PredictEndpointCapability] = None
    models: list[PredictModelInfo]
    predictor_id: str
    ocr_available: bool


class AtomImportance(_Open):
    atom_index: int
    symbol: str
    importance: float
    relative_importance: float


class BondImportance(_Open):
    bond_index: int
    begin_atom_index: int
    end_atom_index: int
    bond_type: str
    importance: float
    relative_importance: float
    display_importance: float = None
    source: Literal["explicit_token", "adjacent_atom_derived"]


class ExplainToken(_Open):
    token: str
    position: int = None
    importance: float
    relative_importance: float = None
    offsets: tuple[int, int] = None


class Depiction(_Open):
    numeric_content_sha256: str
    renderer_version: str
    palette_version: str


class AtomAttribution(_Open):
    """``POST /v1/predict/explain``: token attribution projected onto atoms."""

    status: Literal["completed", "partial", "failed"]
    endpoint: Endpoint
    task: str | None
    input_smiles: str
    canonical_smiles: str | None
    atom_order_version: str | None
    probability: float | None
    atoms: list[AtomImportance]
    bonds: list[BondImportance] = None
    structure_order_version: str | None = None
    depiction_svg: str | None = None
    depiction: Depiction | None = None
    unmapped_importance: float | None
    tokens: list[ExplainToken]
    method: str | None
    metadata: dict[str, Any]
    limitations: list[str]


class AttributionToken(_Response):
    token: str
    score: float


class AttributionProjection(_Response):
    observation_id: str
    run_id: str
    created_at: str
    content_sha256: str
    required_limitations: list[str]
    analysis_id: str
    endpoint: Endpoint
    task: str | None
    status: Literal["completed", "partial"]
    method: str | None
    model_id: str | None
    top_tokens: list[AttributionToken]


class AttributionListResponse(_Response):
    attributions: list[AttributionProjection]


class ObservationResponse(_Response):
    observation_id: str
    run_id: str
    producer: str
    kind: str
    schema_version: str
    model_projection: dict[str, Any]
    provenance: dict[str, Any]
    required_limitations: list[str]
    content_sha256: str
    created_at: str
    canonical_payload: dict[str, Any] | None = None


class SessionProjection(_Response):
    session_id: str
    status: str
    preferred_language: PreferredLanguage
    title: str | None
    title_source: Literal["deterministic", "model", "manual"] | None = None
    title_status: Literal["pending", "ready", "failed"] | None = None
    version: int
    created_at: str
    updated_at: str
    latest_event_sequence: int
    active_run: RunProjection | None
    recent_runs: list[RunProjection]
    active_analysis: AnalysisProjection | None


# -- messages --------------------------------------------------------------------


class MessagePart(_Response):
    part_id: str
    index: int
    type: str
    #: Shape depends on ``type``; the browser narrows it with type guards.
    content: dict[str, Any]


class Message(_Response):
    message_id: str
    role: Literal["user", "assistant", "system_event"]
    sequence: int
    created_at: str
    client_message_id: str | None
    parts: list[MessagePart]


class MessageListResponse(_Response):
    messages: list[Message]
    count: int


# -- answers and evidence --------------------------------------------------------


ClaimKind = Literal[
    "numeric", "classification", "scientific", "comparison", "limitation", "recommendation",
]
LimitationCode = Literal[
    "uncalibrated_probability", "applicability_is_rule_based", "attribution_not_causality",
    "endpoint_unavailable", "evidence_scope_limited", "screening_not_safety_assessment",
]


class Claim(_Response):
    claim_id: str
    kind: ClaimKind
    text: str
    transform: str
    citation_ids: list[str]
    observation_id: str = None
    field_path: str = None
    source_value: float | int | str | bool = None
    rendered_value: str = None
    input_claim_ids: list[str] = None


class Limitation(_Response):
    code: LimitationCode
    text: str


class RecommendedNextStep(_Response):
    text: str
    basis_claim_ids: list[str]


class GroundedAnswer(_Response):
    schema_version: str
    answer_id: str
    run_id: str
    answer_markdown: str
    claims: list[Claim]
    limitations: list[Limitation]
    recommended_next_steps: list[RecommendedNextStep]
    candidate_generation: int
    is_fallback: bool
    content_sha256: str
    created_at: str


class EvidenceRecordView(_Response):
    """Bounded, normalised external material. ``abstract_or_excerpt`` is
    untrusted content, never instructions."""

    evidence_id: str
    title: str
    authors: list[str]
    published_at: str | None
    source_type: str
    source_quality_tier: str
    identifier: dict[str, str | None]
    canonical_url: str | None
    abstract_or_excerpt: str | None
    normalized_facts: dict[str, Any]
    status: str
    rejection_reason: str | None
    provider: str
    retrieved_at: str
    content_sha256: str
    untrusted_external_content: Literal[True]


class EvidenceListResponse(_Response):
    evidence: list[EvidenceRecordView]
    count: int


# -- events ----------------------------------------------------------------------


class ToxAgentEvent(_Response):
    event_id: str
    session_id: str
    sequence: int
    type: str
    entity_type: str
    entity_id: str
    entity_version: int
    run_id: str | None
    occurred_at: str
    payload: dict[str, Any]


class EventListResponse(_Response):
    events: list[ToxAgentEvent]
    count: int
    latest_sequence: int


# -- runs ------------------------------------------------------------------------


class ToolCallView(_Response):
    call_id: str
    tool_name: str
    status: str
    error_code: str | None
    duration_ms: int | None
    arguments_sha256: str | None
    observation_ids: list[str]
    started_at: str | None
    ended_at: str | None


class RuntimeManifest(_Response):
    runtime_binding_id: str
    runtime_kind: str
    runtime_version: str
    provider_id: str
    model_id: str
    auth_mode: str
    connection_id: str | None = None
    profile_hash: str
    tool_schema_hash: str
    system_prompt_hash: str


class UsageTokens(_Response):
    input: int | None
    output: int | None
    reasoning: int | None
    cache_read: int | None
    cache_write: int | None
    total: int | None


class UsageCost(_Response):
    amount: str | None
    currency: str | None


class RuntimeUsageEvent(_Open):
    """One immutable provider report; absent fields stay null, never zero or summed."""

    usage_event_id: str
    runtime_binding_id: str
    provider_id: str
    model_id: str
    reported_at: str
    tokens: UsageTokens
    cost: UsageCost


class RuntimeUsage(_Response):
    status: Literal["unknown", "reported"]
    events: list[RuntimeUsageEvent]


class RunDetail(RunProjection):
    runtime: RuntimeManifest | None
    usage: RuntimeUsage
    tool_calls: list[ToolCallView]


# -- reports ---------------------------------------------------------------------


ReportSourceClass = Literal[
    "structure_fact", "predictor_fact", "explanation_fact",
    "external_evidence", "agent_synthesis", "recommendation",
]


class ReportSection(_Response):
    section_id: str
    heading: str
    body_markdown: str
    claim_ids: list[str]
    figure_ids: list[str]
    table_ids: list[str]
    gap_ids: list[str]
    source_classes: list[ReportSourceClass] = None


class ReportClaim(_Open):
    claim_id: str
    kind: str
    text: str
    observation_id: str | None
    field_path: str | None
    rendered_value: str | None
    transform: str = None
    citation_ids: list[str]


class ReportFigure(_Open):
    figure_id: str
    attachment_id: str
    media_type: str
    caption: str
    alt_text: str
    content_sha256: str
    renderer_version: str
    endpoint: str | None
    task: str | None
    observation_id: str | None


class ReportTable(_Open):
    table_id: str
    title: str
    columns: list[str]
    rows: list[list[str]]
    source_class: ReportSourceClass
    row_claim_ids: list[list[str]] = None


class ReportReference(_Open):
    """REP-01: the immutable snapshot of one cited source."""

    evidence_id: str
    number: int
    title: str
    provider: str
    canonical_url: str | None
    link_url: str | None
    authors: list[str]
    published_at: str | None
    identifier: dict[str, str]
    source_type: str | None
    source_quality_tier: str | None
    retrieved_at: str | None
    unresolved_reason: str | None
    short_form: str


class ExtractedHighlights(_Open):
    positive_contributors: list[dict[str, Any]]
    negative_contributors: list[dict[str, Any]]
    unmapped_importance: float | None


class ReportExplanation(_Open):
    explanation_id: str
    observation_id: str
    endpoint: str
    task: str | None
    method: str | None
    status: Literal["completed", "partial", "failed"]
    figure: ReportFigure | None
    extracted_highlights: ExtractedHighlights
    failure_reason: str | None


class ReportSubstanceProfile(_Open):
    canonical_smiles: str
    structure_figure_id: str | None
    preferred_name: str | None
    synonyms: list[str]
    identifiers: dict[str, Any]
    properties: list[dict[str, Any]]
    source_refs: dict[str, str]


class ReportEvidenceSynthesis(_Open):
    synthesis_id: str
    proposition: str
    relation: Literal["supports", "contradicts", "contextualizes", "insufficient"]
    evidence_ids: list[str]
    endpoint: str | None
    assay: str | None
    organism: str | None
    dose_context: str | None
    quality_notes: list[str]
    conflict_id: str | None


class ReportConclusion(_Open):
    conclusion_id: str
    text: str
    basis_claim_ids: list[str]
    endpoint: str | None
    task: str | None
    is_integrated: bool


class ReportRecommendation(_Open):
    recommendation_id: str
    text: str
    basis_claim_ids: list[str]
    action_category: str
    priority: str
    rationale: str
    conditions: str


class ReportGap(_Open):
    gap_id: str
    reason: str
    detail: str
    section_id: str
    endpoint: str | None = None
    task: str | None = None


class ReportLimitation(_Open):
    code: str
    text: str


class ReportRenderingRef(_Open):
    format: Literal["markdown", "markdown_bundle", "html", "pdf"]
    size_bytes: int
    media_type: str = None


class ReportArtifact(_Open):
    """The report (``domain/report/artifact.py``). v1-v3 are all served; the
    fields every version carries are named, the rest pass through."""

    schema_version: Literal["toxagent-report-v1", "toxagent-report-v2", "toxagent-report-v3"]
    report_id: str
    report_build_id: str
    analysis_id: str
    title: str
    status: Literal["completed", "completed_with_gaps"]
    version: int
    subject: ReportSubstanceProfile
    sections: list[ReportSection]
    #: Empty in a v3 report, which is compiled from a fact bundle.
    tables: list[ReportTable]
    figures: list[ReportFigure]
    #: Empty in a v3 report.
    claims: list[ReportClaim]
    explanations: list[ReportExplanation]
    evidence_synthesis: list[ReportEvidenceSynthesis]
    conclusions: list[ReportConclusion]
    recommendations: list[ReportRecommendation]
    #: v2 and later.
    references: list[ReportReference] = None
    gaps: list[ReportGap]
    limitations: list[ReportLimitation]
    provenance: dict[str, Any]
    renderings: list[ReportRenderingRef]
    content_sha256: str
    created_at: str


# -- model connections -----------------------------------------------------------


class ModelConnectionCapabilities(_Open):
    streaming: bool
    tool_calls: bool
    structured_output: bool
    context_size: int | None


class ModelConnection(_Open):
    connection_id: str
    provider_id: str
    model_id: str
    display_name: str
    auth_mode: str
    base_url: str | None
    has_credential: bool
    capabilities: ModelConnectionCapabilities
    status: Literal["untested", "ready", "failed"]


class ModelConnectionList(_Response):
    connections: list[ModelConnection]


class SupportedProvider(_Response):
    provider_id: str
    display_name: str
    protocol: str
    default_base_url: str | None
    base_url_required: bool
    auth_modes: list[str]
    note: str


class SupportedProviderList(_Response):
    providers: list[SupportedProvider]


# -- scientific cases (ADR 0012) -------------------------------------------------


class CaseCoverage(_Response):
    hypotheses: int
    with_any_source: int
    with_independent_direct_evidence: int
    with_counterevidence_considered: int
    open_uncertainties: int
    blocking_uncertainties: int


class CaseDataScope(_Response):
    external_search: bool
    reason: str
    run_id: str | None


Actor = Literal["user", "model", "server"]


class CaseContextItem(_Open):
    id: str
    key: str
    value: str
    note: str
    actor: Actor
    run_id: str | None


class CaseHypothesis(_Open):
    id: str
    statement: str
    kind: str
    refutation_condition: str
    status: Literal["open", "supported", "weakened", "refuted", "unresolvable"]
    status_reason: str
    actor: str
    run_id: str | None


class CaseEvidenceEntry(_Open):
    id: str
    claim: str
    source_class: str
    source_ref: str
    stance: Literal["supports", "contradicts", "contextual", "insufficient"]
    directness: str
    hypothesis_ids: list[str]
    locator: str | None
    scope: dict[str, str]
    actor: str
    run_id: str | None


class CaseUncertainty(_Open):
    id: str
    kind: str
    description: str
    severity: Literal["low", "medium", "high", "blocking"]
    hypothesis_ids: list[str]
    status: Literal["open", "resolved"]
    resolution: str
    resolving_refs: list[str]
    actor: str
    run_id: str | None


class CaseNextTest(_Open):
    id: str
    test: str
    rationale: str
    discriminates: list[str]
    expected_readouts: list[str]
    actor: str
    run_id: str | None


class CaseAction(_Open):
    id: str
    action: str
    purpose: str
    decision: str
    outcome: str
    run_id: str | None


class CanSay(_Open):
    text: str
    evidence_ids: list[str]


class CaseConclusion(_Open):
    can_say: list[CanSay]
    cannot_say: list[str]
    what_would_change: list[str]
    approver: str
    run_id: str | None


class CaseRun(_Open):
    run_id: str
    goal: str
    stop_reason: str | None
    answer_id: str | None


class ScientificCase(_Open):
    """The case as ``ScientificCaseV1.to_dict`` writes it."""

    schema_version: str
    case_id: str
    session_id: str
    subject_key: str
    question: str
    decision_context: str
    subject_refs: list[str]
    requester: str
    data_scope: CaseDataScope
    status: Literal["open", "closed"]
    context: list[CaseContextItem]
    hypotheses: list[CaseHypothesis]
    evidence: list[CaseEvidenceEntry]
    uncertainties: list[CaseUncertainty]
    actions: list[CaseAction]
    next_tests: list[CaseNextTest]
    conclusion: CaseConclusion
    runs: list[CaseRun]
    coverage: CaseCoverage
    revision: int
    created_at: str
    updated_at: str


class ScientificCaseSummary(_Response):
    case_id: str
    question: str
    subject_key: str
    subject_refs: list[str]
    status: Literal["open", "closed"]
    revision: int
    hypotheses: int
    evidence: int
    open_uncertainties: int
    runs: int
    coverage: CaseCoverage
    updated_at: str
    requester: str
    external_search: bool


class ScientificCaseListResponse(_Response):
    cases: list[ScientificCaseSummary]


class ScientificCaseEvent(_Response):
    op: str
    payload: dict[str, Any]
    actor: Literal["user", "model", "server"]
    at: str
    run_id: str | None
    revision: int


class ScientificCaseEventsResponse(_Response):
    case_id: str
    events: list[ScientificCaseEvent]


class DossierHypothesis(CaseHypothesis):
    evidence_for: list[CaseEvidenceEntry]
    evidence_against: list[CaseEvidenceEntry]


class DossierCanSay(CanSay):
    sources: list[str]


class DossierConclusion(CaseConclusion):
    can_say: list[DossierCanSay]


class DecisionDossier(_Open):
    """``DecisionDossierV1``: what one run concluded over its case."""

    schema_version: str
    case_id: str
    case_revision: int
    run_id: str
    answer_id: str | None
    question: str
    data_scope: CaseDataScope
    hypotheses: list[DossierHypothesis]
    open_uncertainties: list[CaseUncertainty]
    next_tests: list[CaseNextTest]
    conclusion: DossierConclusion
    coverage: CaseCoverage
    stop_reason: str | None


# -- service and health ----------------------------------------------------------


class ServiceInfo(_Response):
    name: str
    version: str
    docs: str


class LiveStatus(_Response):
    status: str


class Capability(_Open):
    """One capability's state (``application/capabilities.py``). Gate UI on
    ``available``; ``configured`` is only what the wiring declared."""

    configured: bool
    available: bool
    checked_at: str
    reason: str | None = None


class HealthReady(_Open):
    ready: bool
    mode: Literal["predictor_only", "agent_enabled"] | None = None
    database: dict[str, Any] | None = None
    predictor: dict[str, Any] | None = None
    explainer: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    capabilities: dict[str, Capability] | None = None


class EffectiveProduct(_Open):
    """What this deployment actually composes (``api/effective_product.py``);
    secret-free by construction."""

    schema_version: str
    toxagent_version: str
    flags: dict[str, Any]
    expired_flags: list[Any]
    intents: dict[str, Any]
    tool_profiles: dict[str, Any]
    runtime: dict[str, Any]
    topology: dict[str, Any]
    providers: dict[str, Any]
    scientific_skills: dict[str, Any]
    hashes: dict[str, Any]
    capabilities: dict[str, Any]
    effective_product_hash: str


# -- reports: listing and builds -------------------------------------------------


class ReportListItem(_Response):
    """Metadata only; the full artifact is one ``GET`` away."""

    report_id: str
    report_build_id: str
    analysis_id: str
    title: str
    status: str
    report_language: str | None
    content_sha256: str
    version: int
    supersedes_report_id: str | None
    created_at: str


class ReportListResponse(_Response):
    reports: list[ReportListItem]
    count: int


class ReportBuildView(_Response):
    report_build_id: str
    session_id: str
    run_id: str
    analysis_id: str
    request: dict[str, Any]
    stage: str
    report_id: str | None
    correction_attempts: int
    failure_code: str | None
    failure_detail: str | None
    deadline_at: str | None
    created_at: str
    updated_at: str


# -- runs: decision state, evidence relations, queue -----------------------------


class DecisionState(_Open):
    """``DecisionSupportStateV1`` for one run (``domain/decision_state.py``)."""

    schema_version: str
    session_id: str
    run_id: str
    goal: str
    subject_refs: list[str]
    propositions: list[dict[str, Any]]
    coverage: dict[str, Any]
    budget_snapshot: dict[str, Any]
    usage: dict[str, Any]
    available_refs: list[str]
    found_refs: list[str]
    stop_reason: str | None
    answer_outcome: str | None = None
    revision: int


class EvidenceRelationAssessment(_Open):
    id: str
    proposition_id: str
    source_ref: dict[str, Any]
    relation: str
    directness: str
    applicability: str
    strength: str
    reason_codes: list[str]
    scope: dict[str, Any]
    input_refs: list[Any]
    assessor: str
    method_version: str
    created_at: str


class EvidenceRelationList(_Response):
    evidence_relations: list[EvidenceRelationAssessment]


class RunQueueStatus(_Response):
    run_id: str
    status: str
    state: Literal["not_queued", "claimed", "deferred", "waiting"]
    queue_name: str | None = None
    position_estimate: int | None = None
    retry_after_s: int | None = None
    waiting_on: str | None = None
    attempts: int | None = None


# -- skill drafts ----------------------------------------------------------------


class SkillDraftAuthor(_Open):
    actor: str
    subject_id: str
    session_id: str | None
    run_id: str | None


class SkillDraftReview(_Open):
    reviewer: str
    decision: str
    note: str
    at: str


class SkillDraft(_Open):
    """A proposed skill package awaiting expert review; never active until promoted."""

    draft_id: str
    skill_id: str
    version: str
    status: str
    author: SkillDraftAuthor
    rationale: str
    manifest: dict[str, Any]
    content_sha256: str
    catalog_sha256: str
    review: SkillDraftReview | None
    created_at: str
    updated_at: str
    skill_md: str | None = None
    references: dict[str, Any] | None = None


class SkillDraftList(_Response):
    drafts: list[SkillDraft]


class SkillDraftPackage(_Response):
    """An approved draft as files, the input to ``scripts/promote_skill_draft.py``."""

    draft_id: str
    skill_id: str
    version: str
    review: SkillDraftReview | None
    files: dict[str, str]
    package_sha256: str
