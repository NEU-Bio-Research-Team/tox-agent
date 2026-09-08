import { apiRequest } from './client';
import { collectSequencedPages } from './pagination';
import type {
  AcceptedResponse,
  AnalysisProjection,
  AtomAttribution,
  AttributionListResponse,
  CancelResponse,
  EventListResponse,
  EvidenceListResponse,
  EvidenceRecordView,
  ExplainRequest,
  GroundedAnswer,
  Message,
  MessageListResponse,
  ModelConnection,
  ObservationResponse,
  PredictCapabilities,
  PreferredLanguage,
  QuickPredictBatchRequest,
  QuickPredictBatchResult,
  QuickPredictCompareRequest,
  QuickPredictCompareResult,
  QuickPredictRequest,
  QuickPredictResult,
  RecognizedStructure,
  RunDetail,
  SessionListResponse,
  SessionProjection,
  SessionResponse,
  ToxAgentEvent,
} from './types';

export interface CreateSessionInput {
  preferred_language?: PreferredLanguage;
  title?: string;
  client_session_id?: string;
}

export function createSession(input: CreateSessionInput = {}): Promise<SessionResponse> {
  return apiRequest('/v1/sessions', { method: 'POST', body: input });
}

export interface SessionSettings {
  ai_profile_id: string | null;
  predictor_bindings: Partial<Record<'clintox' | 'herg' | 'tox21', string>>;
}

export function getSessionSettings(sessionId: string): Promise<SessionSettings> {
  return apiRequest(`/v1/sessions/${sessionId}/settings`);
}

export function updateSessionSettings(sessionId: string, input: SessionSettings): Promise<SessionSettings> {
  return apiRequest(`/v1/sessions/${sessionId}/settings`, { method: 'PATCH', body: input });
}

export function renameSession(sessionId: string, input: { title: string; expected_version: number }): Promise<SessionResponse> {
  return apiRequest(`/v1/sessions/${sessionId}`, { method: 'PATCH', body: input });
}

export function listSessions(params: { limit?: number; offset?: number } = {}): Promise<SessionListResponse> {
  return apiRequest('/v1/sessions', { query: params });
}

export function getSession(sessionId: string): Promise<SessionProjection> {
  return apiRequest(`/v1/sessions/${sessionId}`);
}

export function listMessages(
  sessionId: string,
  params: { after_sequence?: number; limit?: number } = {},
): Promise<MessageListResponse> {
  return apiRequest(`/v1/sessions/${sessionId}/messages`, { query: params });
}

/** Fetches the complete durable transcript, not merely the first REST page.
 * The server orders message sequences monotonically; a short page is the
 * endpoint's explicit end-of-history signal. */
export async function listAllMessages(sessionId: string): Promise<MessageListResponse> {
  const messages = await collectSequencedPages<Message>(
    async (after_sequence, limit) => (await listMessages(sessionId, { after_sequence, limit })).messages,
    { pageSize: 500 },
  );
  return { messages, count: messages.length };
}

export interface SendMessageInput {
  client_message_id?: string;
  content?: Array<{ type: 'text'; text: string }>;
  intent_hint?: 'auto' | 'analyze' | 'ask_report' | 'research_evidence' | 'request_attribution';
  molecule?: { smiles?: string; batch_smiles?: string[] };
  /** Recognised through the toxocr/ service (ADR 0006) into a SMILES, then the
   * same deterministic analysis pipeline a typed SMILES goes through. A
   * deployment with no OCR service configured answers `capability_unavailable`
   * instead (submit_message.py). */
  image?: { mime_type: 'image/png' | 'image/jpeg' | 'image/webp'; data_base64: string };
  analysis_options?: {
    endpoints?: Array<'clintox' | 'herg' | 'tox21'>;
    model_selection?: Partial<Record<'clintox' | 'herg' | 'tox21', string>>;
    threshold_overrides?: Record<string, number> | null;
    include_attribution?: boolean;
    explanation_mode?: 'required' | 'on_demand' | 'none';
    explanation_targets?: Array<{ endpoint: 'herg' | 'tox21'; task?: string }>;
  };
  analysis_id?: string;
}

export function sendMessage(sessionId: string, input: SendMessageInput): Promise<AcceptedResponse> {
  return apiRequest(`/v1/sessions/${sessionId}/messages`, { method: 'POST', body: input });
}

// -- quick predict (stateless, no session) --------------------------------

/** SMILES in, numbers out. Bypasses the session lifecycle: nothing is
 * persisted, so there is no history entry and no audit trail server-side. */
export function quickPredict(input: QuickPredictRequest): Promise<QuickPredictResult> {
  return apiRequest('/v1/predict', { method: 'POST', body: input });
}

export function quickPredictBatch(
  input: QuickPredictBatchRequest,
): Promise<QuickPredictBatchResult> {
  return apiRequest('/v1/predict:batch', { method: 'POST', body: input });
}

/** Explicit multi-model comparison; results stay separate, never ensembled. */
export function quickPredictCompare(input: QuickPredictCompareRequest): Promise<QuickPredictCompareResult> {
  return apiRequest('/v1/predict:compare', { method: 'POST', body: input });
}

export function quickPredictCapabilities(): Promise<PredictCapabilities> {
  return apiRequest('/v1/predict/capabilities');
}

export function listModelConnections(): Promise<{ connections: ModelConnection[] }> {
  return apiRequest('/v1/model-connections');
}

export function createModelConnection(input: {
  provider_id: string; model_id: string; display_name?: string; auth_mode: ModelConnection['auth_mode'];
  base_url?: string; credential?: string;
}): Promise<ModelConnection> {
  return apiRequest('/v1/model-connections', { method: 'POST', body: input });
}

export function testModelConnection(connectionId: string): Promise<ModelConnection> {
  return apiRequest(`/v1/model-connections/${connectionId}:test`, { method: 'POST' });
}

export function deleteModelConnection(connectionId: string): Promise<void> {
  return apiRequest(`/v1/model-connections/${connectionId}`, { method: 'DELETE' });
}

/** Image → SMILES through the stateless toxocr proxy. Two-step by design:
 * the caller confirms (and may edit) the SMILES before calling quickPredict. */
export function recognizeStructure(input: {
  mime_type: 'image/png' | 'image/jpeg' | 'image/webp';
  data_base64: string;
}): Promise<RecognizedStructure> {
  return apiRequest('/v1/predict/recognize', { method: 'POST', body: input });
}

/** Atom-level attribution for one served endpoint (one tox21 assay at a time).
 * `atom_index` is into `canonical_smiles`; depict that exact string. */
export function explainPrediction(input: ExplainRequest): Promise<AtomAttribution> {
  return apiRequest('/v1/predict/explain', { method: 'POST', body: input });
}

export function getRun(sessionId: string, runId: string): Promise<RunDetail> {
  return apiRequest(`/v1/sessions/${sessionId}/runs/${runId}`);
}

export function cancelRun(sessionId: string, runId: string): Promise<CancelResponse> {
  return apiRequest(`/v1/sessions/${sessionId}/runs/${runId}:cancel`, { method: 'POST' });
}

export function getAnalysis(
  sessionId: string,
  analysisId: string,
  params: { include_raw?: boolean } = {},
): Promise<AnalysisProjection> {
  return apiRequest(`/v1/sessions/${sessionId}/analyses/${analysisId}`, { query: params });
}

export function listAttributions(sessionId: string, analysisId: string): Promise<AttributionListResponse> {
  return apiRequest(`/v1/sessions/${sessionId}/analyses/${analysisId}/attributions`);
}

export function getAnswer(sessionId: string, answerId: string): Promise<GroundedAnswer> {
  return apiRequest(`/v1/sessions/${sessionId}/answers/${answerId}`);
}

export function getObservation(sessionId: string, observationId: string): Promise<ObservationResponse> {
  return apiRequest(`/v1/sessions/${sessionId}/observations/${observationId}`);
}

export function listEvidence(
  sessionId: string,
  params: { status?: string; limit?: number; offset?: number } = {},
): Promise<EvidenceListResponse> {
  return apiRequest(`/v1/sessions/${sessionId}/evidence`, { query: params });
}

export async function listAllEvidence(
  sessionId: string,
  params: { status?: string } = {},
): Promise<EvidenceRecordView[]> {
  const evidence: EvidenceRecordView[] = [];
  const limit = 200;
  let offset = 0;
  while (true) {
    const page = await listEvidence(sessionId, { ...params, limit, offset });
    evidence.push(...page.evidence);
    if (page.evidence.length < limit) return evidence;
    offset += page.evidence.length;
  }
}

export function getEvidence(sessionId: string, evidenceId: string): Promise<EvidenceRecordView> {
  return apiRequest(`/v1/sessions/${sessionId}/evidence/${evidenceId}`);
}

export function listEventsOnce(
  sessionId: string,
  params: { after_sequence?: number; limit?: number; run_id?: string } = {},
): Promise<EventListResponse> {
  return apiRequest(`/v1/sessions/${sessionId}/events:list`, { query: params });
}

/** Replays outbox rows through the immutable cursor returned by GET session.
 * Events committed after that snapshot are intentionally not included: the
 * event bus was opened from the same cursor and owns those live updates. */
export function listEventsThroughSnapshot(
  sessionId: string,
  latestSequence: number,
): Promise<ToxAgentEvent[]> {
  return collectSequencedPages<ToxAgentEvent>(
    async (after_sequence, limit) => (await listEventsOnce(sessionId, { after_sequence, limit })).events,
    { pageSize: 500, throughSequence: latestSequence },
  );
}

/** Validation history is a durable outbox projection too. Keep following its
 * per-run cursor instead of silently dropping candidate rejections after the
 * endpoint's 500-row cap. */
export function listAllEventsForRun(sessionId: string, runId: string): Promise<ToxAgentEvent[]> {
  return collectSequencedPages<ToxAgentEvent>(
    async (after_sequence, limit) => (
      await listEventsOnce(sessionId, { after_sequence, limit, run_id: runId })
    ).events,
    { pageSize: 500 },
  );
}

/** One capability's state, as `application/capabilities.py` reports it.
 *
 * `configured` is what the deployment's wiring declared; `available` is what
 * a request would actually meet. They are separate because they disagreed
 * (I01/I05): a stack advertised a runtime kind it never constructed. Read
 * `available` for anything that gates UI, and show `reason` when it is false
 * — it is written for the reader and carries no secrets.
 */
export interface Capability {
  configured: boolean;
  available: boolean;
  checked_at: string;
  reason?: string;
}

export interface HealthReady {
  ready: boolean;
  /** `predictor_only` or `agent_enabled`, derived from what is bound. */
  mode?: 'predictor_only' | 'agent_enabled';
  database?: { ready: boolean; reason?: string; checked_at?: string };
  predictor?: { ready: boolean; served_endpoints?: string[]; reason?: string };
  runtime?: { configured_kind: string; bound: boolean; healthy?: boolean; reason?: string };
  capabilities?: Record<string, Capability>;
}

export function getHealthReady(): Promise<HealthReady> {
  return apiRequest('/health/ready');
}
