import { Copy, MessageCircleQuestion } from 'lucide-react';
import type { AnalysisProjection } from '../../lib/api/types';

/** The session path passes a persisted `AnalysisProjection`; the stateless
 * Quick Predict path passes a shape whose `analysis_id` is `null`. Everything
 * else the panel reads is identical. */
export type PanelAnalysis = Omit<AnalysisProjection, 'analysis_id'> & {
  analysis_id: string | null;
};
import { MoleculeDepiction } from './MoleculeDepiction';
import { EndpointCard } from './EndpointCard';
import { EndpointUnavailableCard } from './EndpointUnavailableCard';
import { Tox21AssayTable } from './Tox21AssayTable';
import { ApplicabilityChip } from './ApplicabilityChip';
import { Button } from '../ui/button';
import { toast } from 'sonner';
import { AttributionPanel } from './AttributionPanel';
import { ExplainPanel } from './ExplainPanel';

const ALL_ENDPOINTS = ['herg', 'clintox', 'tox21'] as const;

export function AnalysisPanel({
  sessionId,
  analysis,
  onAskAboutAnalysis,
}: {
  /** Absent on the stateless Quick Predict page: there is no session to scope
   * a persisted attribution list to, so that panel is simply not rendered. */
  sessionId?: string;
  analysis: PanelAnalysis | null;
  /** Section 8.2.1: picking this sets the composer's analysis context chip
   * and sends `analysis_id` on the next message — viewing an analysis never
   * changes what the next send targets on its own. */
  onAskAboutAnalysis?: (analysisId: string) => void;
}) {
  if (!analysis) {
    return (
      <div
        className="flex h-full min-h-[240px] items-center justify-center rounded-xl border border-dashed p-6 text-center text-sm"
        style={{ borderColor: 'var(--border)', color: 'var(--text-faint)' }}
      >
        Chưa có phân tích nào trong session này. Nhập SMILES ở ô bên dưới để bắt đầu.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="rounded-xl border p-4" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
            Phân tử
          </h2>
          <button
            type="button"
            className="rounded-md p-1 hover:opacity-70"
            onClick={() => {
              void navigator.clipboard.writeText(analysis.canonical_smiles);
              toast.success('Đã copy canonical SMILES');
            }}
            aria-label="Copy canonical SMILES"
          >
            <Copy className="h-3.5 w-3.5" style={{ color: 'var(--text-faint)' }} />
          </button>
        </div>
        <div className="flex justify-center">
          <MoleculeDepiction smiles={analysis.canonical_smiles} />
        </div>
        <p className="mt-3 break-all font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          {analysis.canonical_smiles}
        </p>
        {analysis.analysis_id && (
          <p className="mt-1 font-mono text-xs" style={{ color: 'var(--text-faint)' }}>
            {analysis.analysis_id}
          </p>
        )}
        {onAskAboutAnalysis && analysis.analysis_id && (
          <Button
            variant="outline"
            size="sm"
            className="mt-3 w-full gap-1.5"
            onClick={() => onAskAboutAnalysis(analysis.analysis_id as string)}
          >
            <MessageCircleQuestion className="h-3.5 w-3.5" />
            Hỏi về phân tích này
          </Button>
        )}
      </div>

      {analysis.sections.herg && (
        <EndpointCard
          title="hERG"
          section={analysis.sections.herg}
          probability={analysis.sections.herg.probability_blocker}
        />
      )}
      {analysis.sections.tox21 && <Tox21AssayTable section={analysis.sections.tox21} />}
      {analysis.sections.clintox && (
        <EndpointCard
          title="ClinTox"
          section={analysis.sections.clintox}
          probability={analysis.sections.clintox.probability_clinical_toxicity}
        />
      )}

      {ALL_ENDPOINTS.filter((endpoint) => analysis.unavailable_endpoints.includes(endpoint)).map((endpoint) => (
        <EndpointUnavailableCard key={endpoint} endpoint={endpoint} />
      ))}

      <ApplicabilityChip applicability={analysis.applicability} />

      {analysis.served_endpoints
        .filter((endpoint) => endpoint === 'herg' || endpoint === 'tox21')
        .map((endpoint) => (
          <ExplainPanel
            key={endpoint}
            canonicalSmiles={analysis.canonical_smiles}
            endpoint={endpoint as 'herg' | 'tox21'}
            tox21Tasks={
              endpoint === 'tox21' && analysis.sections.tox21
                ? Object.keys(analysis.sections.tox21.assays)
                : undefined
            }
          />
        ))}

      {sessionId && analysis.analysis_id && (
        <AttributionPanel sessionId={sessionId} analysisId={analysis.analysis_id} />
      )}
    </div>
  );
}
