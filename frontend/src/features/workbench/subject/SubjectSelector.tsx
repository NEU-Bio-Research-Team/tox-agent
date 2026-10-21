import { useQuery } from '@tanstack/react-query';
import { listScreenings, type ScreeningSummary, type SubjectRef } from '../../../shared/api/endpoints';
import { ApiError } from '../../../shared/api/types';

export interface SubjectOption {
  /** `analysis:<id>` hoặc `screening:<id>` — khóa ổn định để so sánh lựa chọn. */
  key: string;
  ref: SubjectRef;
  label: string;
  detail: string;
}

/**
 * Chọn chủ thể cho câu hỏi tiếp theo (AG-UI, §10.2).
 *
 * Phân biệt rõ ligand – target – bối cảnh: cùng một phân tử trên hai target là
 * hai chủ thể khác nhau, nên nhãn luôn nói đủ cả ba. Lựa chọn là tường minh;
 * khi session có nhiều chủ thể mà chưa chọn, hệ thống hỏi lại chứ không tự lấy
 * cái đầu tiên, nên component này không bao giờ tự chọn hộ.
 *
 * Đổi lựa chọn trong lúc một run đang chờ không đổi chủ thể của run đó: nó đã
 * được bind khi message được nhận. Dòng chú thích dưới nói điều đó cho người
 * dùng, vì nếu không thì việc đổi selection trông như đã đổi cả run đang chạy.
 */
export function SubjectSelector({
  sessionId,
  activeAnalysis,
  selected,
  onSelect,
  runPending = false,
}: {
  sessionId: string;
  activeAnalysis?: { analysis_id: string; canonical_smiles?: string | null } | null;
  selected?: string;
  onSelect: (option: SubjectOption) => void;
  runPending?: boolean;
}) {
  const screenings = useQuery({
    queryKey: ['screenings', sessionId],
    queryFn: () => listScreenings(sessionId),
    // 404 = deployment không bật activity replay: không có screening nào, không
    // phải lỗi để hiện cho người dùng.
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 2,
  });

  const options = subjectOptions(activeAnalysis, screenings.data?.screenings ?? []);
  if (options.length === 0) return null;

  return (
    <div className="space-y-1" aria-label="Chọn chủ thể">
      <div className="flex flex-wrap gap-1.5">
        {options.map((option) => {
          const isSelected = option.key === selected;
          return (
            <button
              key={option.key}
              type="button"
              aria-pressed={isSelected}
              onClick={() => onSelect(option)}
              className="rounded-full border px-2.5 py-1 text-left text-xs"
              style={{
                borderColor: isSelected ? 'var(--accent-blue)' : 'var(--border)',
                color: isSelected ? 'var(--accent-blue)' : 'var(--text-muted)',
              }}
            >
              <span className="font-semibold">{option.label}</span>
              <span className="ml-1">{option.detail}</span>
            </button>
          );
        })}
      </div>
      {options.length > 1 && !selected && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Session này có nhiều chủ thể. Hãy chọn một trước khi hỏi — hệ thống không tự chọn hộ.
        </p>
      )}
      {runPending && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Run đang chạy vẫn trả lời về chủ thể đã chọn lúc gửi; đổi lựa chọn chỉ áp dụng cho câu hỏi sau.
        </p>
      )}
    </div>
  );
}

export function subjectOptions(
  activeAnalysis: { analysis_id: string; canonical_smiles?: string | null } | null | undefined,
  screenings: ScreeningSummary[],
): SubjectOption[] {
  const options: SubjectOption[] = [];
  if (activeAnalysis?.analysis_id) {
    options.push({
      key: `analysis:${activeAnalysis.analysis_id}`,
      ref: { kind: 'analysis', analysis_id: activeAnalysis.analysis_id },
      label: 'Phân tích độc tính',
      detail: activeAnalysis.canonical_smiles ?? activeAnalysis.analysis_id,
    });
  }
  for (const screening of screenings) {
    const target = (screening.target ?? {}) as Record<string, unknown>;
    const targetId = String(target.target_id ?? 'không rõ target');
    const variant = target.variant ? ` (${String(target.variant)})` : '';
    options.push({
      key: `screening:${screening.screening_id}`,
      ref: { kind: 'activity_screening', screening_id: screening.screening_id },
      label: `${targetId}${variant}`,
      detail: `${screening.assay_context ?? 'không rõ bối cảnh'} · ${
        screening.toxicity_run ? 'độc tính đã chạy' : 'độc tính chưa chạy'
      }`,
    });
  }
  return options;
}
