/**
 * The control-plane API contract, for the browser.
 *
 * Response shapes are generated: `openapi.d.ts` comes from the control plane's
 * OpenAPI document (`backend/control/src/toxagent/api/responses.py` ->
 * `backend/control/scripts/export_openapi.py` -> `openapi.json` ->
 * `npm run openapi:types`), and each response type below is an alias into it.
 * A field renamed on the server is a type error here, not a silent undefined.
 * CI regenerates both files and fails on drift.
 *
 * What stays hand-written: the small vocabularies the UI switches on, request
 * bodies, the text-part content variants and their type guards, and ApiError.
 */
import type { components } from './openapi';

type Schemas = components['schemas'];

export type PreferredLanguage = 'vi' | 'en';

// -- intent / lane -----------------------------------------------------------

/** What a client may ask for. Distinct from `SelectedIntent` below. */
export type IntentHint =
  | 'auto'
  | 'analyze'
  | 'ask_report'
  | 'research_evidence'
  | 'request_attribution'
  | 'build_report';

/** What the router actually decided. NOT the same enum as IntentHint:
 * "analyze" (hint) becomes "analysis". As of ADR 0010, "ask_report",
 * "research_evidence" and "request_attribution" all become
 * "decision_support" — they no longer name distinct destinations, only a
 * `requested_hint`/reason code kept for audit. "report_qa",
 * "evidence_research" and "attribution" remain valid values only on
 * historical runs created before this change. */
export type SelectedIntent =
  | 'analysis'
  | 'analysis_batch'
  | 'decision_support'
  | 'report_qa'
  | 'evidence_research'
  | 'attribution'
  | 'build_report'
  | 'structure_recognition'
  | 'clarification_required'
  | 'out_of_scope';

export type Lane = 'deterministic' | 'agentic' | 'mixed';

export type RunStatus = 'queued' | 'running' | 'validating' | 'completed' | 'failed' | 'cancelled';

export type Endpoint = 'clintox' | 'herg' | 'tox21';

// -- sessions -----------------------------------------------------------------

export type SessionResponse = Schemas['SessionResponse'];

export type RunSummary = Schemas['RunSummary'];

export type SessionListRow = Schemas['SessionListRow'];

export type SessionListResponse = Schemas['SessionListResponse'];

export type RunProjection = Schemas['RunProjection'];

// -- analysis -------------------------------------------------------------

export interface EndpointSectionCommon {
  measurement: string;
  label: string;
  threshold: number;
  threshold_source: string;
  model_id: string;
}

export type HergSection = Schemas['HergSection'];

export type ClintoxSection = Schemas['ClintoxSection'];

export type Tox21Assay = Schemas['Tox21Assay'];

export type Tox21Section = Schemas['Tox21Section'];

export type AnalysisSections = Schemas['AnalysisSections'];

export type Applicability = Schemas['Applicability'];

export type AnalysisProjection = Schemas['AnalysisProjection'];

// -- quick predict (stateless, no session) ----------------------------------

/** Body for `POST /v1/predict`. No session, no run, nothing persisted. */
export interface QuickPredictRequest {
  smiles: string;
  endpoints?: Endpoint[];
  /** Endpoint -> explicitly selected admitted model. Omit for safe Auto mode. */
  model_selection?: Partial<Record<Endpoint, string>>;
  threshold_overrides?: Record<string, number | Record<string, number>> | null;
  include_attribution?: boolean;
  explanation_mode?: 'required' | 'on_demand' | 'none';
  explanation_targets?: ExplanationTarget[];
}

/** `POST /v1/predict` returns the same `AnalysisProjection` shape the session
 * path returns, plus these two markers. `analysis_id` is always null here. */
export type QuickPredictResult = Schemas['QuickPredictResult'];

export interface QuickPredictBatchRequest {
  smiles: string[];
  endpoints?: Endpoint[];
  model_selection?: Partial<Record<Endpoint, string>>;
  threshold_overrides?: Record<string, number | Record<string, number>> | null;
  explanation_mode?: 'required' | 'on_demand' | 'none';
  explanation_targets?: ExplanationTarget[];
}

export type QuickPredictBatchResult = Schemas['QuickPredictBatchResult'];

export interface QuickPredictCompareRequest {
  smiles: string;
  /** Endpoint -> every explicitly selected admitted model to compare. */
  model_selection: Partial<Record<Endpoint, string[]>>;
  threshold_overrides?: Record<string, number | Record<string, number>> | null;
}

export type QuickPredictCompareResult = Schemas['QuickPredictCompareResult'];

export type PredictModelInfo = Schemas['PredictModelInfo'];

export type PredictEndpointCapability = Schemas['PredictEndpointCapability'];

export interface ActivityLive {
  activity_id: string;
  phase: 'planning' | 'retrieval' | 'reading' | 'prediction' | 'analysis' | 'synthesis';
  kind: string;
  label_key: string;
  status: 'started' | 'progress' | 'completed' | 'failed';
  progress?: { current?: number; total?: number };
}

export type ModelConnection = Schemas['ModelConnection'];

export interface ExplanationTarget {
  endpoint: 'herg' | 'tox21';
  task?: string;
}

/** `GET /v1/predict/capabilities`. */
export type PredictCapabilities = Schemas['PredictCapabilities'];

/** `POST /v1/predict/recognize` — image → SMILES, stateless. */
export type RecognizedStructure = Schemas['RecognizedStructure'];

/** Body for `POST /v1/predict/explain`. One assay per call for tox21. */
export interface ExplainRequest {
  smiles: string;
  endpoint: 'herg' | 'tox21';
  task?: string;
}

export type AtomImportance = Schemas['AtomImportance'];

export type BondImportance = Schemas['BondImportance'];

export type ExplainToken = Schemas['ExplainToken'];

/** `POST /v1/predict/explain` response: token attribution projected onto
 * heavy-atom indices. Never a mechanism, never causal. */
export type AtomAttribution = Schemas['AtomAttribution'];

export type AttributionToken = Schemas['AttributionToken'];

/** One endpoint/task-specific attribution observation. Never aggregate these
 * cards: an attribution only describes what moved that model score. */
export type AttributionProjection = Schemas['AttributionProjection'];

export type AttributionListResponse = Schemas['AttributionListResponse'];

// -- observation --------------------------------------------------------------

export type ObservationResponse = Schemas['ObservationResponse'];

// -- session projection -------------------------------------------------------

export type SessionProjection = Schemas['SessionProjection'];

// -- messages -------------------------------------------------------------

export type PartType = 'text' | 'analysis_ref' | 'answer_ref' | 'report_ref' | 'dossier_ref' | 'tool_call' | 'error' | 'image_ref';

/** A predictor number, an explanation, an external record and an agent's own
 * synthesis are four different kinds of claim about the world, and the report is
 * required never to blur them (spec 3.4). The UI carries the distinction into
 * the visual treatment rather than only into the data. */
export type ReportSourceClass =
  | 'structure_fact'
  | 'predictor_fact'
  | 'explanation_fact'
  | 'external_evidence'
  | 'agent_synthesis'
  | 'recommendation';

export type ReportSection = Schemas['ReportSection'];

export type ReportClaim = Schemas['ReportClaim'];

export type ReportFigure = Schemas['ReportFigure'];

export type ReportTable = Schemas['ReportTable'];

/** REP-01: the resolved, immutable snapshot of one cited source. `link_url` is
 * null when the recorded URL is missing or not HTTPS — the metadata stays so a
 * reader can still identify the source, only the anchor is withheld. */
export type ReportReference = Schemas['ReportReference'];

export type ReportExplanation = Schemas['ReportExplanation'];

export type ReportSubstanceProfile = Schemas['ReportSubstanceProfile'];

export type ReportEvidenceSynthesis = Schemas['ReportEvidenceSynthesis'];

export type ReportConclusion = Schemas['ReportConclusion'];

export type ReportRecommendation = Schemas['ReportRecommendation'];

export type ReportFormat = 'markdown' | 'markdown_bundle' | 'html' | 'pdf';

export type ReportArtifact = Schemas['ReportArtifact'];

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

export type MessagePart = Schemas['MessagePart'];

export type MessageRole = 'user' | 'assistant' | 'system_event';

export type Message = Schemas['Message'];

export type MessageListResponse = Schemas['MessageListResponse'];

// -- answers ----------------------------------------------------------------

/** Every kind `domain/answer.py::ClaimKind` defines. This listed four of
 *  the six; `limitation` and `recommendation` claims exist and were
 *  outside the type, so anything switching on kind was incomplete without
 *  TypeScript being able to say so. */
export type ClaimKind =
  | 'numeric'
  | 'classification'
  | 'scientific'
  | 'comparison'
  | 'limitation'
  | 'recommendation';
export type Transform = 'identity' | `round:${number}` | `percent:${number}` | 'difference' | 'ratio';

export type Claim = Schemas['Claim'];

export type LimitationCode =
  | 'uncalibrated_probability'
  | 'applicability_is_rule_based'
  | 'attribution_not_causality'
  | 'endpoint_unavailable'
  | 'evidence_scope_limited'
  | 'screening_not_safety_assessment';

export type Limitation = Schemas['Limitation'];

export type RecommendedNextStep = Schemas['RecommendedNextStep'];

export type GroundedAnswer = Schemas['GroundedAnswer'];

// -- evidence -----------------------------------------------------------------

/** Bounded, normalized external material. `abstract_or_excerpt` is external
 * untrusted content, never instructions or a source of browser authority. */
export type EvidenceRecordView = Schemas['EvidenceRecordView'];

export type EvidenceListResponse = Schemas['EvidenceListResponse'];

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
  | 'activity.started'
  | 'activity.progress'
  | 'activity.completed'
  | 'activity.failed'
  | 'observation.created'
  | 'analysis.created'
  | 'evidence.created'
  | 'answer.accepted'
  | 'answer.rejected'
  | 'runtime.recovery_started'
  | 'runtime.usage_reported'
  | 'report.build_started'
  | 'report.stage_changed'
  | 'report.figure_created'
  | 'report.draft_saved'
  | 'report.draft_patched'
  | 'report.validation_failed'
  | 'report.completed'
  | 'report.completed_with_gaps'
  | 'report.failed'
  | 'report.cancelled';

export interface Violation {
  code: string;
  message: string;
  path?: string;
  expected?: unknown;
  actual?: unknown;
}

export type ToxAgentEvent = Schemas['ToxAgentEvent'];

export type EventListResponse = Schemas['EventListResponse'];

// -- accepted / clarification -------------------------------------------------

export interface Clarification {
  code: string;
  question: string;
  options: string[];
}

export type AcceptedResponse = Schemas['AcceptedResponse'];

export type CancelResponse = Schemas['CancelResponse'];

// -- tool calls (run detail) --------------------------------------------------

export type ToolCallView = Schemas['ToolCallView'];

export type RuntimeManifest = Schemas['RuntimeManifest'];

/** One immutable provider report. Fields absent from a report are deliberately
 * null/unknown, never coerced to zero or summed across events: providers may
 * emit incompatible delta and cumulative accounting records. */
export type RuntimeUsageEvent = Schemas['RuntimeUsageEvent'];

export type RuntimeUsage = Schemas['RuntimeUsage'];

export type RunDetail = Schemas['RunDetail'];

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

// --- scientific cases (ADR 0012) ---------------------------------------------

export type HypothesisStatus = 'open' | 'supported' | 'weakened' | 'refuted' | 'unresolvable';
export type EvidenceStance = 'supports' | 'contradicts' | 'contextual' | 'insufficient';

export type CaseContextItem = Schemas['CaseContextItem'];

export type CaseHypothesis = Schemas['CaseHypothesis'];

export type CaseEvidenceEntry = Schemas['CaseEvidenceEntry'];

export type CaseUncertainty = Schemas['CaseUncertainty'];

export type CaseNextTest = Schemas['CaseNextTest'];

export type CaseConclusion = Schemas['CaseConclusion'];

/** Ref coverage and quality coverage side by side; never one score. */
export type CaseCoverage = Schemas['CaseCoverage'];

/** What the case may reach (W9-07); set only by the researcher. */
export type CaseDataScope = Schemas['CaseDataScope'];

export type ScientificCase = Schemas['ScientificCase'];

export type ScientificCaseSummary = Schemas['ScientificCaseSummary'];

export type ScientificCaseListResponse = Schemas['ScientificCaseListResponse'];

/** DecisionDossierV1 (ADR 0012): what one run concluded over its case. The
 * chat answer and the investigation board are views of it (W9-05). Only the
 * fields the transcript renders are typed here. */
export type DossierHypothesis = Schemas['DossierHypothesis'];

export type DecisionDossier = Schemas['DecisionDossier'];

/** One entry of the case's append-only log: the case is the fold of these. */
export type ScientificCaseEvent = Schemas['ScientificCaseEvent'];

export type ScientificCaseEventsResponse = Schemas['ScientificCaseEventsResponse'];

export interface SetCaseScopeInput {
  external_search: boolean;
  reason?: string;
}

export interface AddCaseContextInput {
  key: string;
  value: string;
  note?: string;
}
