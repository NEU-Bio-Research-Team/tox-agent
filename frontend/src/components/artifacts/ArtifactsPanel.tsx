import { useMemo } from 'react';
import { useNavigate } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { PackageOpen, X } from 'lucide-react';
import { Button } from '../ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { ArtifactViewer } from './ArtifactViewer';
import { artifactPath, type ArtifactKind, type ArtifactSelection } from '../../hooks/useArtifactSelection';
import { buildArtifactPickerOptions } from '../../lib/artifacts';
import { listAllEvidence } from '../../lib/api/endpoints';
import type { SessionProjection } from '../../lib/api/types';

/**
 * Right-region content (plan section 8.2.1): header, kind/time selector,
 * one viewer at a time. This is pure content — `WorkbenchPage` decides
 * whether it sits in a resizable desktop column or a tablet/mobile Sheet
 * (section 8.2.2), so this component never renders its own overlay.
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
  const options = useMemo(() => [
    ...buildArtifactPickerOptions(session),
    ...(evidenceQuery.data ?? []).map((record) => ({
      value: `evidence:${record.evidence_id}`,
      label: `Evidence · ${record.title}`,
    })),
  ], [session, evidenceQuery.data]);
  const currentValue = selection ? `${selection.kind}:${selection.entityId}` : '';

  return (
    <div className="flex h-full flex-col" style={{ backgroundColor: 'var(--surface)' }}>
      <div className="flex h-14 shrink-0 items-center gap-2 border-b px-3" style={{ borderColor: 'var(--border)' }}>
        <p className="flex-1 truncate text-sm font-semibold" style={{ color: 'var(--text)' }}>
          Artifacts
        </p>
        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose} aria-label="Đóng artifacts">
          <X className="h-4 w-4" />
        </Button>
      </div>

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
        {options.length > 0 && (
          <p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
            Đã tải {options.length} kết quả gần nhất — answer/observation mở qua link trong hội thoại.
          </p>
        )}
        {evidenceQuery.isError && (
          <p className="mt-1 text-xs" style={{ color: 'var(--accent-red)' }}>
            Không tải được danh sách evidence.
          </p>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {selection ? (
          <ArtifactViewer sessionId={sessionId} selection={selection} onAskAboutAnalysis={onAskAboutAnalysis} />
        ) : (
          <div
            className="flex h-full min-h-[200px] flex-col items-center justify-center gap-2 rounded-xl border border-dashed p-6 text-center"
            style={{ borderColor: 'var(--border)' }}
          >
            <PackageOpen className="h-6 w-6" style={{ color: 'var(--text-faint)' }} />
            <p className="text-sm" style={{ color: 'var(--text-faint)' }}>
              Chưa có kết quả nào để xem. Phân tích một phân tử hoặc nhận đáp án để kết quả xuất hiện ở đây.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
