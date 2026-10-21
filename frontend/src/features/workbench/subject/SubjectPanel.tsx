import { useQuery } from '@tanstack/react-query';
import { getScreening, type SubjectRef } from '../../../shared/api/endpoints';
import { ApiError } from '../../../shared/api/types';
import { ScreeningCard } from './ScreeningCard';
import { SubjectSelector, type SubjectOption } from './SubjectSelector';

/**
 * Vùng chọn chủ thể đặt ngay trên composer (AG-UI, §10.1 bước 1–3).
 *
 * Khi chủ thể đang chọn là một activity screening, thẻ artifact hiện ngay các
 * dữ kiện đã có — người dùng không phải chờ một lượt LLM chỉ để thấy pIC50 hay
 * trạng thái "độc tính chưa chạy".
 */
export function SubjectPanel({
  sessionId,
  activeAnalysis,
  selected,
  onSelect,
  runPending,
}: {
  sessionId: string;
  activeAnalysis?: { analysis_id: string; canonical_smiles?: string | null } | null;
  selected?: SubjectOption;
  onSelect: (option: SubjectOption | undefined) => void;
  runPending?: boolean;
}) {
  const screeningId = screeningIdOf(selected?.ref);
  const detail = useQuery({
    queryKey: ['screening', sessionId, screeningId],
    queryFn: () => getScreening(sessionId, screeningId as string),
    enabled: Boolean(screeningId),
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 2,
  });

  return (
    <div className="space-y-2">
      <SubjectSelector
        sessionId={sessionId}
        activeAnalysis={activeAnalysis}
        selected={selected?.key}
        onSelect={(option) => onSelect(option.key === selected?.key ? undefined : option)}
        runPending={runPending}
      />
      {screeningId && detail.data && (
        <div
          className="rounded-md border px-3 py-2"
          style={{ borderColor: 'var(--border-subtle)', backgroundColor: 'var(--surface)' }}
        >
          <ScreeningCard screening={detail.data} />
        </div>
      )}
    </div>
  );
}

export function screeningIdOf(ref: SubjectRef | undefined): string | undefined {
  return ref?.kind === 'activity_screening' ? (ref.screening_id ?? undefined) : undefined;
}
