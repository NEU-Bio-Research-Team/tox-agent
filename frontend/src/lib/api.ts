const configuredBaseUrl = (
	(import.meta as { env?: Record<string, string | undefined> }).env?.VITE_API_BASE_URL ??
	''
).trim();

function resolveBaseUrl(baseUrl: string): string {
	if (!baseUrl) {
		return '';
	}

	const normalized = baseUrl.replace(/\/$/, '');

	if (typeof window === 'undefined') {
		return normalized;
	}

	const hostname = window.location.hostname;
	const isHostedEnvironment = hostname !== 'localhost' && hostname !== '127.0.0.1';
	const isLocalApiUrl = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(normalized);

	// Safety net: if production bundle accidentally includes localhost API URL,
	// use relative path so Firebase Hosting rewrites route API calls to Cloud Run.
	if (isHostedEnvironment && isLocalApiUrl) {
		return '';
	}

	return normalized;
}

export const BASE_URL = resolveBaseUrl(configuredBaseUrl);

export type RiskLevelCode = 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW' | 'UNKNOWN';
export interface RiskLevelDetail {
	level?: RiskLevelCode | string | null;
	description?: string | null;
}
export type RiskLevel = RiskLevelCode | RiskLevelDetail;
export type InferenceBackend = 'xsmiles' | 'chemberta' | 'pubchem' | 'molformer';

export interface AgentEventRecord {
	type?: string | null;
	author?: string | null;
	function_calls: Array<Record<string, unknown>>;
	function_responses: Array<Record<string, unknown>>;
	is_final: boolean;
	text_preview?: string | null;
}

export interface ClinicalSection {
	verdict?: string | null;
	probability?: number | null;
	confidence?: number | null;
	threshold_used?: number | null;
	interpretation?: string | null;
}

export interface MechanismSection {
	active_tox21_tasks?: string[];
	highest_risk?: string | null;
	assay_hits?: number | null;
	task_scores?: Record<string, number>;
}

export interface StructuralAtom {
	atom_idx: number;
	element: string;
	importance: number;
	is_in_ring?: boolean;
	is_aromatic?: boolean;
}

export interface StructuralBond {
	bond_idx?: number;
	atom_pair?: string;
	bond_type?: string;
	importance: number;
}

export interface StructuralSection {
	top_atoms?: StructuralAtom[];
	top_bonds?: StructuralBond[];
	heatmap_base64?: string | null;
	molecule_png_base64?: string | null;
	target_task?: string | null;
	target_task_score?: number | null;
	explainer_note?: string | null;
}

export interface MolragRetrievedExample {
	entry_id?: string;
	name?: string;
	smiles?: string;
	canonical_smiles?: string;
	similarity?: number;
	label?: string;
	source?: string;
	notes?: string;
	is_exact_match?: boolean;
}

export interface MolragSection {
	enabled?: boolean;
	strategy?: string;
	retrieval_db_size?: number | null;
	retrieval_error?: string | null;
	retrieved_examples?: MolragRetrievedExample[];
	evidence_summary?: string | null;
	reasoning_summary?: string | null;
	suggested_label?: string | null;
	confidence?: number | null;
	prompt_preview?: string | null;
	reasoning_mode?: string | null;
	error?: string | null;
}

export interface FusionResultSection {
	mode?: string;
	baseline_label?: string;
	baseline_score?: number | null;
	baseline_confidence?: number | null;
	molrag_label?: string;
	molrag_confidence?: number | null;
	final_label?: string;
	final_confidence?: number | null;
	agreement?: boolean | null;
	decision_note?: string;
}

export interface OodAssessmentSection {
	ood_risk?: string;
	flag?: boolean;
	reason?: string;
	rare_elements?: string[];
	high_risk_elements?: string[];
	recommendation?: string | null;
}

export interface FailureRegistryEntry {
	id?: string;
	canonical_smiles?: string;
	common_name?: string;
	true_label?: number;
	p_toxic_at_default_threshold?: number;
	default_threshold?: number;
	ood_flag?: boolean;
	ood_reason?: string;
	threshold_at_which_correctly_classified?: number | null;
	recommended_action?: string;
	source?: string;
}

export interface FailureRegistrySection {
	matched?: boolean;
	entry?: FailureRegistryEntry | null;
	registry_size?: number;
}

export interface InferenceContextSection {
	workspace_mode?: string;
	threshold_policy?: string;
	clinical_threshold_applied?: number;
	clinical_model_loaded?: boolean;
	tox21_model_loaded?: boolean;
	explainer_used?: boolean;
	explanation_available?: boolean;
	tox21_threshold_source?: string | null;
	clinical_reference_metrics?: Record<string, number>;
}

export interface LiteraturePaper {
	pmid?: string;
	title?: string;
	authors?: string | string[];
	year?: string;
	journal?: string;
	snippet?: string;
	abstract_snippet?: string;
	pubmed_url?: string;
}

export interface BioassayItem {
	aid?: number | string;
	assay_name?: string;
	activity_outcome?: string;
}

export interface LiteratureSection {
	compound_id?: {
		cid?: number | null;
		pubchem_url?: string | null;
	};
	query_name_used?: string;
	total_found?: number;
	relevant_papers?: LiteraturePaper[];
	bioassay_evidence?: {
		cid?: number;
		active_assays?: BioassayItem[];
		total_assays_tested?: number;
		tox21_active_count?: number;
		error?: string | null;
	} | null;
	bioassay_explanation?: string;
}

export interface RecommendationSection {
	title?: string | null;
	content?: string | null;
}

export interface FinalReport {
	report_metadata: {
		smiles: string;
		canonical_smiles?: string | null;
		compound_name?: string | null;
		common_name?: string | null;
		iupac_name?: string | null;
		language?: string | null;
		analysis_timestamp?: string | null;
		timestamp?: string | null;
		report_version?: string | null;
		report_id?: string | null;
	};
	executive_summary: string;
	risk_level: RiskLevel;
	sections: {
		clinical_toxicity: ClinicalSection;
		mechanism_toxicity: MechanismSection;
		structural_explanation: StructuralSection;
		molrag_evidence?: MolragSection;
		fusion_result?: FusionResultSection;
		literature_context: LiteratureSection;
		ood_assessment?: OodAssessmentSection;
		inference_context?: InferenceContextSection;
		reliability_warning?: string | null;
		recommendation_source?: 'llm' | 'deterministic' | string;
		recommendation_source_detail?: string;
		failure_registry?: FailureRegistrySection;
		recommendations?: string[] | RecommendationSection;
	};
}

export interface AgentAnalyzeResponse {
	session_id: string;
	chat_session_id?: string | null;
	adk_available: boolean;
	runtime_mode?: 'adk' | 'deterministic_fallback' | string;
	runtime_note?: string | null;
	validation_status: string | null;
	final_report: FinalReport;
	final_text: string | null;
	agent_events: AgentEventRecord[];
	state_keys: string[];
}

export interface AgentChatResponse {
	chat_session_id: string;
	response: string;
}

export interface SmilesImageExtractionResponse {
	smiles: string | null;
	canonical_smiles: string | null;
	confidence: number | null;
	warnings: string[];
	error_code: string | null;
	message: string | null;
}

export class SmilesImageExtractionError extends Error {
	status: number;
	code: string;

	constructor(status: number, code: string, message: string) {
		super(message);
		this.name = 'SmilesImageExtractionError';
		this.status = status;
		this.code = code;
	}
}

function toErrorMessage(status: number, bodyText: string): string {
	if (!bodyText) {
		return `API error ${status}`;
	}

	try {
		const parsed = JSON.parse(bodyText) as {
			error?: string;
			message?: string;
			detail?: { message?: string; error?: string } | string;
		};
		if (parsed.error && parsed.message) {
			return `API error ${status} (${parsed.error}): ${parsed.message}`;
		}
		if (parsed.message) {
			return `API error ${status}: ${parsed.message}`;
		}
		if (parsed.error) {
			return `API error ${status}: ${parsed.error}`;
		}
		if (typeof parsed.detail === 'string') {
			return `API error ${status}: ${parsed.detail}`;
		}
		if (parsed.detail?.message) {
			return `API error ${status}: ${parsed.detail.message}`;
		}
		if (parsed.detail?.error) {
			return `API error ${status}: ${parsed.detail.error}`;
		}
	} catch {
		// Keep plain text fallback below.
	}

	return `API error ${status}: ${bodyText}`;
}

export interface AgentAnalyzeOptions {
	language?: 'en';
	clinicalThreshold?: number;
	mechanismThreshold?: number;
	maxLiteratureResults?: number;
	inferenceBackend?: InferenceBackend;
	binaryToxModel?: string;
	toxTypeModel?: string;
	molragEnabled?: boolean;
	molragTopK?: number;
	molragMinSimilarity?: number;
}

export async function agentAnalyze(
	smiles: string,
	options: AgentAnalyzeOptions = {},
): Promise<AgentAnalyzeResponse> {
	const payload = {
		smiles,
		include_agent_events: true,
		language: options.language ?? 'en',
		clinical_threshold: options.clinicalThreshold ?? 0.35,
		mechanism_threshold: options.mechanismThreshold ?? 0.5,
		max_literature_results: options.maxLiteratureResults ?? 5,
		inference_backend: options.inferenceBackend ?? 'chemberta',
		binary_tox_model: options.binaryToxModel ?? 'pretrained_2head_herg_chemberta_model',
		tox_type_model: options.toxTypeModel ?? 'tox21_gatv2_model',
		molrag_enabled: options.molragEnabled ?? true,
		molrag_top_k: options.molragTopK ?? 5,
		molrag_min_similarity: options.molragMinSimilarity ?? 0.15,
	};

	const res = await fetch(`${BASE_URL}/agent/analyze`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(payload),
	});

	if (!res.ok) {
		const bodyText = await res.text();
		throw new Error(toErrorMessage(res.status, bodyText));
	}

	return (await res.json()) as AgentAnalyzeResponse;
}

export interface AgentChatOptions {
	chatSessionId?: string | null;
	analysisSessionId?: string | null;
	reportState?: {
		smiles_input?: string;
		final_report?: FinalReport | Record<string, unknown>;
		evidence_qa_result?: Record<string, unknown>;
	} | null;
}

export async function agentChat(
	message: string,
	options: AgentChatOptions = {},
): Promise<AgentChatResponse> {
	const payload = {
		message,
		chat_session_id: options.chatSessionId ?? undefined,
		analysis_session_id: options.analysisSessionId ?? undefined,
		report_state: options.reportState ?? undefined,
	};

	const res = await fetch(`${BASE_URL}/agent/chat`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(payload),
	});

	if (!res.ok) {
		const bodyText = await res.text();
		throw new Error(toErrorMessage(res.status, bodyText));
	}

	return (await res.json()) as AgentChatResponse;
}

export async function extractSmilesFromImage(
	file: File,
): Promise<SmilesImageExtractionResponse> {
	const formData = new FormData();
	formData.append('file', file);

	const res = await fetch(`${BASE_URL}/extract-smiles-from-image`, {
		method: 'POST',
		body: formData,
	});

	if (!res.ok) {
		const bodyText = await res.text();
		let code = 'extraction_service_unavailable';
		let message = toErrorMessage(res.status, bodyText);

		if (bodyText) {
			try {
				const parsed = JSON.parse(bodyText) as {
					error?: string;
					message?: string;
					detail?: { error?: string; message?: string } | string;
				};

				if (typeof parsed.error === 'string' && parsed.error.trim()) {
					code = parsed.error;
				}
				if (typeof parsed.message === 'string' && parsed.message.trim()) {
					message = parsed.message;
				}
				if (typeof parsed.detail === 'object' && parsed.detail) {
					if (typeof parsed.detail.error === 'string' && parsed.detail.error.trim()) {
						code = parsed.detail.error;
					}
					if (typeof parsed.detail.message === 'string' && parsed.detail.message.trim()) {
						message = parsed.detail.message;
					}
				}
			} catch {
				// Keep fallback error message/code.
			}
		}

		throw new SmilesImageExtractionError(res.status, code, message);
	}

	return (await res.json()) as SmilesImageExtractionResponse;
}
