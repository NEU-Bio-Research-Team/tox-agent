import { useMemo } from 'react';
import { useNavigate } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { PackageOpen, X } from 'lucide-react';
import { Button } from '../ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { ArtifactViewer } from './ArtifactViewer';
import { InvestigationBoard } from '../workbench/InvestigationBoard';
import { artifactPath, type ArtifactKind, type ArtifactSelection } from '../../hooks/useArtifactSelection';
import { buildArtifactPickerOptions } from '../../lib/artifacts';
import { listAllEvidence, listScientificCases } from '../../lib/api/endpoints';
import type { SessionProjection } from '../../lib/api/types';

/**
 * Right-region content (plan section 8.2.1): header, result picker, the
 * selected result, and the session's sources under it. This is pure content —
 * `WorkbenchPage` decides whether it sits in a resizable desktop column or a
 * tablet/mobile Sheet (section 8.2.2), so this component never renders its
 * own overlay.
 *
 * One scrolling panel rather than Sources/Results/Điều tra tabs: a selected
 * source opens in the same viewer as any other result, and the investigation
 * board appears only for a session that has a scientific case — with
 * `scientific_case_v1` off, which is the default, it had nothing to show.
 */
export function ArtifactsPanel({
  sessionId,
  session,
  selection,
  onClose,
  onAskAboutAnalysis,
}: {
  sessionId: string;
  session: SessionProjection;
  selection: ArtifactSelection | null;
  onClose: () => void;
  onAskAboutAnalysis?: (analysisId: string) => void;
}) {
  const navigate = useNavigate();
  const evidenceQuery = useQuery({
    queryKey: ['evidence', sessionId],
    queryFn: () => listAllEvidence(sessionId, { status: 'all' }),
  });
  // Same key as InvestigationBoard's own list, so opening the board reuses it.
  const casesQuery = useQuery({
    queryKey: ['cases', sessionId],
    queryFn: () => listScientificCases(sessionId),
  });
  const caseCount = casesQuery.data?.cases.length ?? 0;
  const options = useMemo(() => buildArtifactPickerOptions(session), [session]);
  const currentValue = selection && selection.kind !== 'evidence' ? `${selection.kind}:${selection.entityId}` : '';
  const evidence = evidenceQuery.data ?? [];

  return (
    <div className="flex h-full flex-col" style={{ backgroundColor: 'var(--surface)' }}>
      <div className="flex h-14 shrink-0 items-center gap-2 border-b px-3" style={{ borderColor: 'var(--border)' }}>
        <p className="flex-1 truncate text-sm font-semibold" style={{ color: 'var(--text)' }}>
          Kết quả
        </p>
        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose} aria-label="Đóng artifacts">
          <X className="h-4 w-4" />
        </Button>
      </div>

      {options.length > 0 && (
        <div className="border-b px-3 py-2" style={{ borderColor: 'var(--border)' }}>
          <Select
            value={currentValue || undefined}
            aria-label="Chọn artifact để xem"
            onValueChange={(value) => {
              const separatorIndex = value.indexOf(':');
              const kind = value.slice(0, separatorIndex) as ArtifactKind;
              const entityId = value.slice(separatorIndex + 1);
              navigate(artifactPath(sessionId, { kind, entityId }));
            }}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue placeholder="Chọn kết quả…" />
            </SelectTrigger>
            <SelectContent>
              {options.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      <div className="min-h-0 flex-1 space-y-6 overflow-y-auto p-4">
        {selection ? (
          <ArtifactViewer sessionId={sessionId} selection={selection} onAskAboutAnalysis={onAskAboutAnalysis} />
        ) : (
          <div
            className="flex min-h-[200px] flex-col items-center justify-center gap-2 rounded-xl border border-dashed p-6 text-center"
            style={{ borderColor: 'var(--border)' }}
          >
            <PackageOpen className="h-6 w-6" style={{ color: 'var(--text-faint)' }} />
            <p className="text-sm" style={{ color: 'var(--text-faint)' }}>
              Chưa có kết quả nào để xem. Phân tích một phân tử hoặc nhận đáp án để kết quả xuất hiện ở đây.
            </p>
          </div>
        )}

        {(evidence.length > 0 || evidenceQuery.isError) && (
          <section aria-labelledby="artifacts-sources-heading" className="space-y-2">
            <h2 id="artifacts-sources-heading" className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
              Nguồn
            </h2>
            {evidenceQuery.isError && (
              <p className="text-xs" style={{ color: 'var(--accent-red)' }}>
                Không tải được danh sách nguồn.
              </p>
            )}
            {evidence.map((record, index) => {
              const selected = selection?.kind === 'evidence' && selection.entityId === record.evidence_id;
              return (
                <button
                  key={record.evidence_id}
                  type="button"
                  aria-current={selected ? 'true' : undefined}
                  className="w-full rounded-lg border p-2 text-left"
                  style={{ borderColor: selected ? 'var(--accent-blue)' : 'var(--border)' }}
                  onClick={() => navigate(artifactPath(sessionId, { kind: 'evidence', entityId: record.evidence_id }))}
                >
                  <p className="line-clamp-2 text-xs font-medium" style={{ color: 'var(--text)' }}>
                    <sup>{index + 1}</sup> {record.title}
                  </p>
                  <p className="mt-1 text-[11px]" style={{ color: 'var(--text-faint)' }}>
                    {record.source_type} · {record.published_at ?? 'n.d.'}
                  </p>
                </button>
              );
            })}
          </section>
        )}

        {caseCount > 0 && (
          <details className="rounded-lg border p-3" style={{ borderColor: 'var(--border)' }}>
            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
              Điều tra khoa học ({caseCount})
            </summary>
            <div className="mt-3">
              <InvestigationBoard sessionId={sessionId} />
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
