import { Brain, CheckCircle, Database, Loader2, Search } from 'lucide-react';

export interface ThinkingStep {
	id: string;
	type: 'reasoning' | 'tool_call' | 'tool_result';
	status: 'running' | 'done';
	text?: string;
	tool?: string;
	input?: string;
}

const TOOL_LABELS: Record<string, string> = {
	check_claim_support: 'Verifying report claim support',
	get_article_detail: 'Loading curated article detail',
	get_report_section: 'Expanding report section context',
	rerun_screening: 'Re-running toxicity screening',
	query_molrag_live: 'Retrieving similar compounds',
	compare_with_analogs: 'Comparing nearest analog compounds',
	lookup_structural_alerts: 'Checking structural alerts',
	explain_mechanism: 'Resolving mechanism detail',
	fetch_pubmed_context: 'Fetching additional PubMed context',
};

function ToolIcon({ tool }: { tool?: string }) {
	if (tool === 'check_claim_support') {
		return <Search className="h-3 w-3" />;
	}
	if (tool === 'get_article_detail' || tool === 'get_report_section') {
		return <Database className="h-3 w-3" />;
	}
	return <Brain className="h-3 w-3" />;
}

function truncateInput(input?: string, max = 90): string {
	const text = String(input ?? '').trim();
	if (text.length <= max) {
		return text;
	}
	return `${text.slice(0, max)}...`;
}

export function AgentThinkingLog({ steps }: { steps: ThinkingStep[] }) {
	if (!Array.isArray(steps) || steps.length === 0) {
		return null;
	}

	return (
		<div
			className="mb-2 rounded-2xl border px-3 py-2 text-xs"
			style={{
				backgroundColor: 'var(--surface-alt)',
				borderColor: 'var(--border)',
				color: 'var(--text-muted)',
			}}
		>
			<div className="mb-1 flex items-center gap-1.5">
				<Brain className="h-3 w-3" style={{ color: 'var(--accent-blue)' }} />
				<span className="text-xs font-medium" style={{ color: 'var(--text-faint)' }}>
					Agent reasoning
				</span>
			</div>

			<div className="space-y-1.5">
				{steps.map((step) => (
					<div key={step.id} className="flex items-start gap-2">
						<div className="mt-[1px] shrink-0">
							{step.status === 'running' ? (
								<Loader2 className="h-3 w-3 animate-spin" style={{ color: 'var(--accent-blue)' }} />
							) : (
								<CheckCircle className="h-3 w-3" style={{ color: 'var(--accent-green)' }} />
							)}
						</div>

						{step.type === 'reasoning' && <span>{step.text}</span>}

						{step.type === 'tool_call' && (
							<div className="flex min-w-0 items-center gap-1.5">
								<ToolIcon tool={step.tool} />
								<span className="font-mono" style={{ color: 'var(--accent-blue)' }}>
									{step.tool || 'tool'}
								</span>
								<span>{TOOL_LABELS[step.tool || ''] || 'Calling tool'}</span>
								{step.input && (
									<span className="truncate opacity-70">({truncateInput(step.input)})</span>
								)}
							</div>
						)}

						{step.type === 'tool_result' && (
							<span className="opacity-80">{step.tool || 'tool'} completed</span>
						)}
					</div>
				))}
			</div>
		</div>
	);
}
