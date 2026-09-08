import { useMemo } from 'react';
import { useNavigate } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { FileText, FlaskConical, PackageOpen, X } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
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
          Sources &amp; Results
        </p>
        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose} aria-label="Đóng artifacts">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <Tabs defaultValue="results" className="flex min-h-0 flex-1 flex-col">
      <div className="border-b px-3 py-2" style={{ borderColor: 'var(--border)' }}>
        <TabsList className="mb-2 grid w-full grid-cols-2"><TabsTrigger value="sources" className="gap-1"><FileText className="h-3.5 w-3.5" />Sources</TabsTrigger><TabsTrigger value="results" className="gap-1"><FlaskConical className="h-3.5 w-3.5" />Results</TabsTrigger></TabsList>
        <TabsContent value="results" className="m-0">
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
        </TabsContent>
        <TabsContent value="sources" className="m-0 space-y-2">
          {evidenceQuery.isLoading && <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Đang tải nguồn…</p>}
          {(evidenceQuery.data ?? []).length === 0 && !evidenceQuery.isLoading && <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Chưa có nguồn evidence cho phiên này.</p>}
          {(evidenceQuery.data ?? []).map((record, index) => <button key={record.evidence_id} type="button" className="w-full rounded-lg border p-2 text-left" style={{ borderColor: 'var(--border)' }} onClick={() => navigate(artifactPath(sessionId, { kind: 'evidence', entityId: record.evidence_id }))}><p className="line-clamp-2 text-xs font-medium" style={{ color: 'var(--text)' }}><sup>{index + 1}</sup> {record.title}</p><p className="mt-1 text-[11px]" style={{ color: 'var(--text-faint)' }}>{record.source_type} · {record.published_at ?? 'n.d.'}</p></button>)}
        </TabsContent>
      </div>

      <TabsContent value="results" className="m-0 flex-1 overflow-y-auto p-4">
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
      </TabsContent>
      <TabsContent value="sources" className="m-0 flex-1 overflow-y-auto p-4"><div className="flex h-full items-center justify-center text-center text-xs" style={{ color: 'var(--text-faint)' }}>Chọn một source để xem passage, metadata và provenance.</div></TabsContent>
      </Tabs>
    </div>
  );
}
