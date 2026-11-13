import { Copy, MessageCircleQuestion } from 'lucide-react';
import type { AnalysisProjection } from '../../shared/api/types';

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
import { Button } from '../../shared/ui/button';
import { Callout } from '../../shared/ui/callout';
import { LIMITATION_LABEL_VI } from '../../shared/lib/labels';
import { toast } from 'sonner';
import { AttributionPanel } from './AttributionPanel';
import { ExplainPanel } from './ExplainPanel';

const ALL_ENDPOINTS = ['herg', 'clintox', 'tox21'] as const;

/**
 * One analysis: the structure, the applicability note, one card per measurement
 * (each followed by its own explanation), and the limitations the server
 * requires. Used by the session workbench and by Phân tích nhanh. Nothing here
 * combines measurements (ADR-0002).
 */
export function AnalysisPanel({
  sessionId,
  analysis,
  onAskAboutAnalysis,
  hideMolecule = false,
}: {
  /** Absent on the stateless Quick Predict page: there is no session to scope
   * a persisted attribution list to, so that panel is simply not rendered. */
  sessionId?: string;
  analysis: PanelAnalysis | null;
  /** Section 8.2.1: picking this sets the composer's analysis context chip
   * and sends `analysis_id` on the next message — viewing an analysis never
   * changes what the next send targets on its own. */
  onAskAboutAnalysis?: (analysisId: string) => void;
  /** Phân tích nhanh already shows the structure beside the results. */
  hideMolecule?: boolean;
}) {
  if (!analysis) {
    return (
      <div className="flex h-full min-h-[240px] items-center justify-center rounded-xl border border-dashed border-line-strong p-6 text-center text-sm text-ink-tertiary">
        Chưa có phân tích nào trong session này. Nhập SMILES ở ô bên dưới để bắt đầu.
      </div>
    );
  }

  const explain = (endpoint: 'herg' | 'tox21') =>
    analysis.served_endpoints.includes(endpoint) ? (
      <ExplainPanel
        canonicalSmiles={analysis.canonical_smiles}
        endpoint={endpoint}
        tox21Tasks={endpoint === 'tox21' && analysis.sections.tox21 ? Object.keys(analysis.sections.tox21.assays) : undefined}
        modelId={endpoint === 'herg' ? analysis.sections.herg?.model_id : analysis.sections.tox21?.model_id}
      />
    ) : null;

  return (
    <div className="space-y-4">
      {hideMolecule ? null : (
        <div className="rounded-xl border border-line bg-card p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-ink">Phân tử</h2>
            <button
              type="button"
              className="rounded-md p-1 text-ink-tertiary hover:text-ink"
              onClick={() => {
                void navigator.clipboard.writeText(analysis.canonical_smiles);
                toast.success('Đã copy canonical SMILES');
              }}
              aria-label="Copy canonical SMILES"
            >
              <Copy className="size-3.5" />
            </button>
          </div>
          <div className="flex justify-center">
            <MoleculeDepiction smiles={analysis.canonical_smiles} />
          </div>
          <p className="mt-3 break-all font-mono text-xs text-ink-secondary">{analysis.canonical_smiles}</p>
          {analysis.analysis_id && <p className="mt-1 break-all font-mono text-xs text-ink-tertiary">{analysis.analysis_id}</p>}
          {onAskAboutAnalysis && analysis.analysis_id && (
            <Button
              variant="outline"
              size="sm"
              className="mt-3 w-full gap-1.5"
              onClick={() => onAskAboutAnalysis(analysis.analysis_id as string)}
            >
              <MessageCircleQuestion className="size-3.5" />
              Hỏi về phân tích này
            </Button>
          )}
        </div>
      )}

      <ApplicabilityChip applicability={analysis.applicability} />

      {analysis.sections.herg && (
        <>
          <EndpointCard
            title="hERG"
            section={analysis.sections.herg}
            probability={analysis.sections.herg.probability_blocker}
            probabilityLabel="xác suất blocker"
          />
          {explain('herg')}
        </>
      )}
      {analysis.sections.tox21 && (
        <>
          <Tox21AssayTable section={analysis.sections.tox21} />
          {explain('tox21')}
        </>
      )}
      {analysis.sections.clintox && (
        <EndpointCard
          title="ClinTox"
          section={analysis.sections.clintox}
          probability={analysis.sections.clintox.probability_clinical_toxicity}
          probabilityLabel="xác suất độc tính lâm sàng"
        />
      )}

      {ALL_ENDPOINTS.filter((endpoint) => analysis.unavailable_endpoints.includes(endpoint)).map((endpoint) => (
        <EndpointUnavailableCard key={endpoint} endpoint={endpoint} />
      ))}

      {analysis.required_limitations.length > 0 && (
        <Callout tone="warning" title="Giới hạn">
          <ul className="list-disc space-y-1 pl-4">
            {analysis.required_limitations.map((code) => (
              <li key={code}>{(LIMITATION_LABEL_VI as Record<string, string>)[code] ?? code}</li>
            ))}
          </ul>
        </Callout>
      )}

      {sessionId && analysis.analysis_id && <AttributionPanel sessionId={sessionId} analysisId={analysis.analysis_id} />}
    </div>
  );
}
