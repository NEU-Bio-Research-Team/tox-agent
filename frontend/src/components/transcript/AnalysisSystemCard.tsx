import { Link } from 'react-router';
import { CheckCircle2, Loader2, XCircle } from 'lucide-react';
import type { RunProjection } from '../../lib/api/types';
import { ERROR_CODE_LABEL_VI } from '../../lib/labels';

/** Lane D produces no assistant message (E2E doc section 1) — the only
 * visible trace of an `analysis`/`analysis_batch` run in the transcript is
 * this card. Without it, submitting a SMILES for analysis would look like
 * nothing happened. `analysisId` is only known once `analysis.created` has
 * actually landed (run completed); earlier statuses have nothing to link to
 * yet. */
export function AnalysisSystemCard({
  sessionId,
  run,
  analysisId,
}: {
  sessionId: string;
  run: RunProjection;
  analysisId?: string;
}) {
  const isStructureRecognition = run.intent === 'structure_recognition';
  const durationMs =
    run.started_at && run.ended_at
      ? new Date(run.ended_at).getTime() - new Date(run.started_at).getTime()
      : null;

  if (run.status === 'failed') {
    return (
      <div id={`run-anchor-${run.run_id}`} className="my-2 flex items-center gap-2 rounded-lg px-3 py-2 text-xs" style={{ backgroundColor: 'var(--surface-alt)' }} role="status" aria-live="polite" aria-atomic="true">
        <XCircle className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--accent-red)' }} />
        <span style={{ color: 'var(--text-muted)' }}>
          {isStructureRecognition ? 'Nhận diện/phân tích cấu trúc thất bại' : 'Phân tích thất bại'} — {run.failure_code ? (ERROR_CODE_LABEL_VI[run.failure_code] ?? run.failure_code) : 'lỗi không xác định'}
        </span>
      </div>
    );
  }

  if (run.status === 'cancelled') {
    return (
      <div id={`run-anchor-${run.run_id}`} className="my-2 flex items-center gap-2 rounded-lg px-3 py-2 text-xs" style={{ backgroundColor: 'var(--surface-alt)' }} role="status" aria-live="polite" aria-atomic="true">
        <XCircle className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--text-faint)' }} />
        <span style={{ color: 'var(--text-muted)' }}>{isStructureRecognition ? 'Nhận diện cấu trúc đã bị huỷ' : 'Phân tích đã bị huỷ'}</span>
      </div>
    );
  }

  if (run.status === 'completed') {
    const label = (
      <span style={{ color: 'var(--text-muted)' }}>
        {isStructureRecognition ? 'Đã nhận diện và tạo phân tích' : 'Đã tạo phân tích'}{durationMs !== null ? ` · ${durationMs} ms` : ''} — xem ở cột bên phải
      </span>
    );
    return analysisId ? (
      <Link
        id={`run-anchor-${run.run_id}`}
        to={`/s/${sessionId}/analyses/${analysisId}`}
        className="my-2 flex items-center gap-2 rounded-lg px-3 py-2 text-xs hover:opacity-80"
        style={{ backgroundColor: 'var(--surface-alt)' }}
      >
        <CheckCircle2 className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--accent-green)' }} />
        {label}
      </Link>
    ) : (
      <div id={`run-anchor-${run.run_id}`} className="my-2 flex items-center gap-2 rounded-lg px-3 py-2 text-xs" style={{ backgroundColor: 'var(--surface-alt)' }} role="status" aria-live="polite" aria-atomic="true">
        <CheckCircle2 className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--accent-green)' }} />
        {label}
      </div>
    );
  }

  return (
    <div className="my-2 flex items-center gap-2 rounded-lg px-3 py-2 text-xs" style={{ backgroundColor: 'var(--surface-alt)' }} role="status" aria-live="polite" aria-atomic="true">
      <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" style={{ color: 'var(--accent-blue)' }} />
      <span style={{ color: 'var(--text-muted)' }}>
        {isStructureRecognition
          ? run.status === 'queued'
            ? 'Đang nhận diện cấu trúc từ ảnh…'
            : 'Đang phân tích SMILES đã nhận diện…'
          : 'Đang gọi predictor…'}
      </span>
    </div>
  );
}
