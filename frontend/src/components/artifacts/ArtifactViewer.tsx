import { useQuery } from '@tanstack/react-query';
import { getAnalysis } from '../../lib/api/endpoints';
import { errorMessageVi } from '../../lib/labels';
import { ApiError } from '../../lib/api/types';
import type { ArtifactSelection } from '../../hooks/useArtifactSelection';
import { AnalysisPanel } from '../workbench/AnalysisPanel';
import { RunInspectorContent } from '../inspector/RunInspectorContent';
import { AnswerAuditViewer } from './AnswerAuditViewer';
import { ObservationArtifact } from './ObservationArtifact';
import { EvidenceArtifact } from './EvidenceArtifact';

function AnalysisArtifact({
  sessionId,
  analysisId,
  onAskAboutAnalysis,
}: {
  sessionId: string;
  analysisId: string;
  onAskAboutAnalysis?: (analysisId: string) => void;
}) {
  const query = useQuery({
    queryKey: ['analysis', sessionId, analysisId],
    queryFn: () => getAnalysis(sessionId, analysisId),
  });

  if (query.isLoading) {
    return <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Đang tải…</p>;
  }
  if (query.isError || !query.data) {
    return (
      <p className="text-sm" style={{ color: 'var(--accent-red)' }}>
        {query.error instanceof ApiError
          ? errorMessageVi(query.error.code, query.error.message)
          : 'Không tải được phân tích này.'}
      </p>
    );
  }

  return <AnalysisPanel sessionId={sessionId} analysis={query.data} onAskAboutAnalysis={onAskAboutAnalysis} />;
}

/**
 * Switches on artifact kind (plan section 8.2.1's viewer table). One viewer
 * renders at a time — the panel never grows a fourth column for a picker;
 * the selector living in `ArtifactsPanel`'s header is what changes which
 * kind/id this receives.
 */
export function ArtifactViewer({
  sessionId,
  selection,
  onAskAboutAnalysis,
}: {
  sessionId: string;
  selection: ArtifactSelection;
  onAskAboutAnalysis?: (analysisId: string) => void;
}) {
  switch (selection.kind) {
    case 'analysis':
      return <AnalysisArtifact sessionId={sessionId} analysisId={selection.entityId} onAskAboutAnalysis={onAskAboutAnalysis} />;
    case 'run':
      return <RunInspectorContent sessionId={sessionId} runId={selection.entityId} />;
    case 'answer':
      return <AnswerAuditViewer sessionId={sessionId} answerId={selection.entityId} />;
    case 'observation':
      return <ObservationArtifact sessionId={sessionId} observationId={selection.entityId} fieldPath={selection.fieldPath} />;
    case 'evidence':
      return <EvidenceArtifact sessionId={sessionId} evidenceId={selection.entityId} />;
    default:
      return null;
  }
}
