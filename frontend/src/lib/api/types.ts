/**
 * Hand-typed against the real control-plane contract (toxagent-control),
 * not generated from OpenAPI: most GET routes return ad-hoc dicts with no
 * pydantic response_model, so a generated schema would type them as `unknown`.
 * Every shape here traces to a specific route/dataclass — see
 * docs/spec/TOXAGENT_AGENTIC_LAYER_REBUILD_PLAN_VI.md section 26.4 for the
 * file/line provenance of each field.
 */

export type PreferredLanguage = 'vi' | 'en';

// -- intent / lane -----------------------------------------------------------

/** What a client may ask for. Distinct from `SelectedIntent` below. */
export type IntentHint =
  | 'auto'
  | 'analyze'
  | 'ask_report'
  | 'research_evidence'
  | 'request_attribution';

/** What the router actually decided. NOT the same enum as IntentHint:
 * "analyze" (hint) becomes "analysis" (selected), "ask_report" becomes
 * "report_qa", "request_attribution" becomes "attribution". */
export type SelectedIntent =
  | 'analysis'
  | 'analysis_batch'
  | 'report_qa'
  | 'evidence_research'
  | 'attribution'
  | 'structure_recognition'
  | 'clarification_required'
  | 'out_of_scope';

export type Lane = 'deterministic' | 'agentic' | 'mixed';

export type RunStatus = 'queued' | 'running' | 'validating' | 'completed' | 'failed' | 'cancelled';

export type Endpoint = 'clintox' | 'herg' | 'tox21';

// -- sessions -----------------------------------------------------------------

export interface SessionResponse {
  session_id: string;
  status: string;
  preferred_language: PreferredLanguage;
  title: string | null;
  created_at: string;
  version: number;
}

export interface RunSummary {
  run_id: string;
  status: RunStatus;
  intent: SelectedIntent;
}

export interface SessionListRow {
  session_id: string;
  title: string | null;
  status: string;
  preferred_language: PreferredLanguage;
  created_at: string;
  updated_at: string;
  active_run: RunSummary | null;
  run_count: number;
  last_message_preview: string | null;
}

export interface SessionListResponse {
  sessions: SessionListRow[];
  next_offset: number | null;
}

export interface RunProjection {
  run_id: string;
  status: RunStatus;
  lane: Lane;
  intent: SelectedIntent;
  trigger_message_id: string;
  runtime_binding_id: string | null;
  recovery_of_run_id: string | null;
  failure_code: string | null;
  potentially_billed: boolean;
  deadline_at: string;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
}

// -- analysis -------------------------------------------------------------

export interface EndpointSectionCommon {
  measurement: string;
  label: string;
  threshold: number;
  threshold_source: string;
  model_id: string;
}

export interface HergSection extends EndpointSectionCommon {
  probability_blocker: number;
}

export interface ClintoxSection extends EndpointSectionCommon {
  probability_clinical_toxicity: number;
}

export interface Tox21Assay {
  probability_activity: number;
  active: boolean;
  threshold: number;
  threshold_source: string;
}

export interface Tox21Section {
  measurement: string;
  task_order_version: string;
  model_id: string;
  /** A mapping, never a count (SCI-05). Never reduce this to a number. */
  assays: Record<string, Tox21Assay>;
}

export interface AnalysisSections {
  herg?: HergSection;
  clintox?: ClintoxSection;
  tox21?: Tox21Section;
}

export interface Applicability {
  status: string;
  method: string;
  reasons: string[];
}

export interface AnalysisProjection {
  analysis_id: string;
  input_smiles: string;
  canonical_smiles: string;
  requested_endpoints: Endpoint[];
  served_endpoints: Endpoint[];
  unavailable_endpoints: Endpoint[];
  sections: AnalysisSections;
  applicability: Applicability;
  provenance: {
    predictor_base_url_id?: string;
    predictor_service_version?: string;
    predictor_git_commit?: string;
    artifact_hashes?: string[];
    content_sha256: string;
    [k: string]: unknown;
  };
  policy_snapshot: Record<string, unknown>;
  required_limitations: LimitationCode[];
  created_at: string;
}

// -- quick predict (stateless, no session) ----------------------------------

/** Body for `POST /v1/predict`. No session, no run, nothing persisted. */
export interface QuickPredictRequest {
  smiles: string;
  endpoints?: Endpoint[];
  threshold_overrides?: Record<string, number | Record<string, number>> | null;
  include_attribution?: boolean;
}

/** `POST /v1/predict` returns the same `AnalysisProjection` shape the session
 * path returns, plus these two markers. `analysis_id` is always null here. */
export interface QuickPredictResult extends Omit<AnalysisProjection, 'analysis_id'> {
  analysis_id: null;
  persisted: false;
  attributions?: unknown[];
}

export interface QuickPredictBatchRequest {
  smiles: string[];
  endpoints?: Endpoint[];
  threshold_overrides?: Record<string, number | Record<string, number>> | null;
}

export interface QuickPredictBatchResult {
  results: QuickPredictResult[];
  errors: Array<{ index: number; input_smiles: string; error: string; detail: string }>;
  count: number;
}

export interface PredictModelInfo {
  model_id: string;
  capabilities: string[];
  loaded: boolean;
  required: boolean;
  detail: string;
  blocked_reason: string | null;
}

/** `GET /v1/predict/capabilities`. */
export interface PredictCapabilities {
  served_endpoints: Endpoint[];
  models: PredictModelInfo[];
  predictor_id: string;
  ocr_available: boolean;
}

/** `POST /v1/predict/recognize` — image → SMILES, stateless. */
export interface RecognizedStructure {
  smiles: string;
  canonical_smiles: string;
  /** The OCR service may omit this; when present it is in [0, 1]. */
  confidence: number | null;
}

/** Body for `POST /v1/predict/explain`. One assay per call for tox21. */
export interface ExplainRequest {
  smiles: string;
  endpoint: 'herg' | 'tox21';
  task?: string;
}

export interface AtomImportance {
  atom_index: number;
  symbol: string;
  importance: number;
  relative_importance: number;
}

export interface ExplainToken {
  token: string;
  position?: number;
  importance: number;
  relative_importance?: number;
  offsets?: [number, number];
}

/** `POST /v1/predict/explain` response: token attribution projected onto
 * heavy-atom indices. Never a mechanism, never causal. */
export interface AtomAttribution {
  status: 'completed' | 'partial' | 'failed';
  endpoint: Endpoint;
  task: string | null;
  input_smiles: string;
  /** The FE MUST depict this exact string for `atom_index` to line up. */
  canonical_smiles: string | null;
  atom_order_version: string | null;
  probability: number | null;
  atoms: AtomImportance[];
  /** Normalised fraction of importance that landed on bonds/topology, not atoms. */
  unmapped_importance: number | null;
  tokens: ExplainToken[];
  method: string | null;
  metadata: Record<string, unknown>;
  limitations: string[];
}

export interface AttributionToken {
  token: string;
  score: number;
}

/** One endpoint/task-specific attribution observation. Never aggregate these
 * cards: an attribution only describes what moved that model score. */
export interface AttributionProjection {
  observation_id: string;
  run_id: string;
  created_at: string;
  content_sha256: string;
  required_limitations: string[];
  analysis_id: string;
  endpoint: Endpoint;
  task: string | null;
  status: 'completed' | 'partial';
  method: string | null;
  model_id: string | null;
  top_tokens: AttributionToken[];
}

export interface AttributionListResponse {
  attributions: AttributionProjection[];
}

// -- observation --------------------------------------------------------------

export interface ObservationResponse {
  observation_id: string;
  run_id: string;
  producer: string;
  kind: string;
  schema_version: string;
  model_projection: Record<string, unknown> & { observation_id: string };
  provenance: Record<string, unknown>;
  required_limitations: string[];
  content_sha256: string;
  created_at: string;
  canonical_payload?: Record<string, unknown>;
}

// -- session projection -------------------------------------------------------

export interface SessionProjection {
  session_id: string;
  status: string;
  preferred_language: PreferredLanguage;
  title: string | null;
  version: number;
  created_at: string;
  updated_at: string;
  latest_event_sequence: number;
  active_run: RunProjection | null;
  recent_runs: RunProjection[];
  active_analysis: AnalysisProjection | null;
}

// -- messages -------------------------------------------------------------

export type PartType = 'text' | 'analysis_ref' | 'answer_ref' | 'tool_call' | 'error' | 'image_ref';

/** A gateway-produced answer message: `{text: answer_markdown}` — see
 * harness/gateway.py `_commit_answer_message`. */
export interface AssistantTextContent {
  text: string;
}

/** A clarification/out-of-scope message built with no runtime at all — see
 * application/submit_message.py `_answer_without_a_runtime`. Same PartType
 * ("text") as AssistantTextContent, but a different, mutually-exclusive
 * shape: there is no `.text` key here, only `.question`/`.message`. */
export interface ClarificationTextContent {
  reason: string;
  code: string;
  question: string;
  options?: string[];
  message?: string;
  [key: string]: unknown;
}

/** A durable hand-off from toxocr to the deterministic predictor. This is a
 * recognition suggestion, not a toxicity score or a safety assessment. */
export interface StructureRecognitionTextContent extends Record<string, unknown> {
  code: 'structure_recognized';
  smiles: string;
  canonical_smiles: string;
  /** The OCR service may omit confidence; when present it is in [0, 1]. */
  confidence?: number;
}

export function isStructureRecognitionContent(
  content: Record<string, unknown>,
): content is StructureRecognitionTextContent {
  return (
    content.code === 'structure_recognized' &&
    typeof content.smiles === 'string' &&
    typeof content.canonical_smiles === 'string' &&
    (content.confidence === undefined ||
      (typeof content.confidence === 'number' &&
        Number.isFinite(content.confidence) &&
        content.confidence >= 0 &&
        content.confidence <= 1))
  );
}

export type TextPartContent = AssistantTextContent | ClarificationTextContent | StructureRecognitionTextContent;

export function isClarificationContent(content: Record<string, unknown>): content is ClarificationTextContent {
  return typeof content.code === 'string' && !isStructureRecognitionContent(content);
}

export interface MessagePart {
  part_id: string;
  index: number;
  type: PartType;
  content: Record<string, unknown>;
}

export type MessageRole = 'user' | 'assistant' | 'system_event';

export interface Message {
  message_id: string;
  role: MessageRole;
  sequence: number;
  created_at: string;
  client_message_id: string | null;
  parts: MessagePart[];
}

export interface MessageListResponse {
  messages: Message[];
  count: number;
}

// -- answers ----------------------------------------------------------------

export type ClaimKind = 'numeric' | 'classification' | 'scientific' | 'comparison';
export type Transform = 'identity' | `round:${number}` | `percent:${number}` | 'difference' | 'ratio';

export interface Claim {
  claim_id: string;
  kind: ClaimKind;
  text: string;
  transform: Transform | string;
  citation_ids: string[];
  observation_id?: string;
  field_path?: string;
  source_value?: number | string | boolean;
  rendered_value?: string;
  input_claim_ids?: string[];
}

export type LimitationCode =
  | 'uncalibrated_probability'
  | 'applicability_is_rule_based'
  | 'attribution_not_causality'
  | 'endpoint_unavailable'
  | 'evidence_scope_limited'
  | 'screening_not_safety_assessment';

export interface Limitation {
  code: LimitationCode;
  text: string;
}

export interface RecommendedNextStep {
  text: string;
  basis_claim_ids: string[];
}

export interface GroundedAnswer {
  schema_version: string;
  answer_id: string;
  run_id: string;
  answer_markdown: string;
  claims: Claim[];
  limitations: Limitation[];
  recommended_next_steps: RecommendedNextStep[];
  candidate_generation: number;
  is_fallback: boolean;
  content_sha256: string;
  created_at: string;
}

// -- evidence -----------------------------------------------------------------

/** Bounded, normalized external material. `abstract_or_excerpt` is external
 * untrusted content, never instructions or a source of browser authority. */
export interface EvidenceRecordView {
  evidence_id: string;
  title: string;
  authors: string[];
  published_at: string | null;
  source_type: string;
  source_quality_tier: string;
  identifier: Record<string, string | null>;
  canonical_url: string | null;
  abstract_or_excerpt: string | null;
  normalized_facts: Record<string, unknown>;
  status: string;
  rejection_reason: string | null;
  provider: string;
  retrieved_at: string;
  content_sha256: string;
  untrusted_external_content: true;
}

export interface EvidenceListResponse {
  evidence: EvidenceRecordView[];
  count: number;
}

// -- events ---------------------------------------------------------------

export type EventType =
  | 'session.created'
  | 'message.created'
  | 'run.queued'
  | 'run.started'
  | 'run.validating'
  | 'run.completed'
  | 'run.failed'
  | 'run.cancelled'
  | 'part.created'
  | 'part.updated'
  | 'tool.started'
  | 'tool.completed'
  | 'tool.failed'
  | 'observation.created'
  | 'analysis.created'
  | 'evidence.created'
  | 'answer.accepted'
  | 'answer.rejected'
  | 'runtime.recovery_started'
  | 'runtime.usage_reported';

export interface Violation {
  code: string;
  message: string;
  path?: string;
  expected?: unknown;
  actual?: unknown;
}

export interface ToxAgentEvent {
  event_id: string;
  session_id: string;
  sequence: number;
  type: EventType;
  entity_type: string;
  entity_id: string;
  entity_version: number;
  run_id: string | null;
  occurred_at: string;
  payload: Record<string, unknown>;
}

export interface EventListResponse {
  events: ToxAgentEvent[];
  count: number;
  latest_sequence: number;
}

// -- accepted / clarification -------------------------------------------------

export interface Clarification {
  code: string;
  question: string;
  options: string[];
}

export interface AcceptedResponse {
  message_id: string;
  run_id: string;
  run_status: RunStatus;
  selected_intent: SelectedIntent;
  lane: Lane;
  events_url: string;
  clarification?: Clarification;
  duplicate_of_message_id?: string;
}

export interface CancelResponse {
  run_id: string;
  requested: boolean;
  runtime_cancel_supported: boolean;
  action: string;
}

// -- tool calls (run detail) --------------------------------------------------

export interface ToolCallView {
  call_id: string;
  tool_name: string;
  status: string;
  error_code: string | null;
  duration_ms: number | null;
  started_at: string | null;
  ended_at: string | null;
}

export interface RuntimeManifest {
  runtime_binding_id: string;
  runtime_kind: string;
  runtime_version: string;
  provider_id: string;
  model_id: string;
  profile_hash: string;
  tool_schema_hash: string;
  system_prompt_hash: string;
}

/** One immutable provider report. Fields absent from a report are deliberately
 * null/unknown, never coerced to zero or summed across events: providers may
 * emit incompatible delta and cumulative accounting records. */
export interface RuntimeUsageEvent {
  usage_event_id: string;
  runtime_binding_id: string;
  provider_id: string;
  model_id: string;
  reported_at: string;
  tokens: {
    input: number | null;
    output: number | null;
    reasoning: number | null;
    cache_read: number | null;
    cache_write: number | null;
    total: number | null;
  };
  cost: {
    amount: string | null;
    currency: string | null;
  };
}

export interface RuntimeUsage {
  status: 'unknown' | 'reported';
  events: RuntimeUsageEvent[];
}

export interface RunDetail extends RunProjection {
  runtime: RuntimeManifest | null;
  usage: RuntimeUsage;
  tool_calls: ToolCallView[];
}

// -- error envelope -------------------------------------------------------

export interface ErrorBody {
  error: {
    code: string;
    message: string;
    retryable: boolean;
    details: Record<string, unknown>;
  };
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;
  readonly details: Record<string, unknown>;

  constructor(status: number, body: ErrorBody) {
    super(body.error.message);
    this.name = 'ApiError';
    this.status = status;
    this.code = body.error.code;
    this.retryable = body.error.retryable;
    this.details = body.error.details;
  }
}
