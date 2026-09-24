import { Check, Loader2, Minus, RotateCcw, X } from 'lucide-react';
import {
  reportStageLabel,
  type ReportProgress,
  type ReportStageStatus,
} from '../../lib/store/reportProgress';

const STATUS_TEXT: Record<ReportStageStatus, string> = {
  pending: 'chưa bắt đầu',
  running: 'đang chạy',
  completed: 'xong',
  skipped: 'bỏ qua theo yêu cầu',
  failed: 'lỗi',
};

function StageIcon({ status }: { status: ReportStageStatus }) {
  const common = 'h-3.5 w-3.5 shrink-0';
  if (status === 'running') {
    return <Loader2 aria-hidden className={`${common} animate-spin motion-reduce:animate-none`} style={{ color: 'var(--purple-600)' }} />;
  }
  if (status === 'completed') return <Check aria-hidden className={common} style={{ color: 'var(--accent-green)' }} />;
  if (status === 'failed') return <X aria-hidden className={common} style={{ color: 'var(--accent-red)' }} />;
  if (status === 'skipped') return <Minus aria-hidden className={common} style={{ color: 'var(--text-faint)' }} />;
  return <span aria-hidden className={`${common} inline-block rounded-full border`} style={{ borderColor: 'var(--text-faint)' }} />;
}

/** A report build's stages as the server recorded them (WS09).
 *
 * Status is carried in text as well as in the icon, so it is not conveyed by
 * colour alone. The live region announces only the current stage, not every
 * event: a screen reader reading eight stage transitions aloud is noise.
 */
export function ReportProgressTimeline({ progress }: { progress: ReportProgress }) {
  if (progress.stages.length === 0) {
    return (
      <p className="my-2 text-sm" role="status" aria-live="polite" style={{ color: 'var(--text-muted)' }}>
        {reportStageLabel(progress.currentStage)}…
      </p>
    );
  }
  const current = progress.outcome ? null : progress.currentStage;
  return (
    <div className="my-2 space-y-1.5" data-testid="report-progress">
      <p className="text-sm" role="status" aria-live="polite" style={{ color: 'var(--text-muted)' }}>
        {current
          ? `${reportStageLabel(current)} (${progress.completed}/${progress.total})`
          : progress.outcome === 'failed'
            ? 'Không tạo được báo cáo.'
            : progress.outcome === 'cancelled'
              ? 'Đã dừng tạo báo cáo.'
              : progress.outcome
                ? 'Đã tạo xong báo cáo.'
                : `Đã xong ${progress.completed}/${progress.total} bước`}
        {progress.retried && (
          <span className="ml-2 inline-flex items-center gap-1 text-xs">
            <RotateCcw aria-hidden className="h-3 w-3" /> có bước phải chạy lại
          </span>
        )}
      </p>
      <ol className="space-y-1 text-xs" aria-label="Các bước tạo báo cáo">
        {progress.stages.map((stage) => (
          <li
            key={stage.stage}
            className="flex min-w-0 items-center gap-2"
            aria-current={stage.stage === current ? 'step' : undefined}
            style={{ color: stage.status === 'pending' || stage.status === 'skipped' ? 'var(--text-faint)' : 'var(--text-muted)' }}
          >
            <StageIcon status={stage.status} />
            <span className="truncate">{reportStageLabel(stage.stage)}</span>{' '}
            <span className="shrink-0">
              — {STATUS_TEXT[stage.status]}
              {stage.attempt > 1 ? ` (lần thử ${stage.attempt})` : ''}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
