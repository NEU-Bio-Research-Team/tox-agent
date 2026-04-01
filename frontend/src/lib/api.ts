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

export type RiskLevel = 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW' | 'UNKNOWN';

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
}

export interface FinalReport {
	report_metadata: {
		smiles: string;
		canonical_smiles?: string | null;
		compound_name?: string | null;
		analysis_timestamp: string;
		report_version: string;
	};
	executive_summary: string;
	risk_level: RiskLevel;
	sections: {
		clinical_toxicity: ClinicalSection;
		mechanism_toxicity: MechanismSection;
		structural_explanation: StructuralSection;
		literature_context: LiteratureSection;
		recommendations: string[];
	};
}

export interface AgentAnalyzeResponse {
	session_id: string;
	adk_available: boolean;
	validation_status: string | null;
	final_report: FinalReport;
	final_text: string | null;
	agent_events: AgentEventRecord[];
	state_keys: string[];
}

function toErrorMessage(status: number, bodyText: string): string {
	if (!bodyText) {
		return `API error ${status}`;
	}

	try {
		const parsed = JSON.parse(bodyText) as {
			detail?: { message?: string; error?: string } | string;
		};
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

export async function agentAnalyze(smiles: string): Promise<AgentAnalyzeResponse> {
	const res = await fetch(`${BASE_URL}/agent/analyze`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ smiles, include_agent_events: true }),
	});

	if (!res.ok) {
		const bodyText = await res.text();
		throw new Error(toErrorMessage(res.status, bodyText));
	}

	return (await res.json()) as AgentAnalyzeResponse;
}
