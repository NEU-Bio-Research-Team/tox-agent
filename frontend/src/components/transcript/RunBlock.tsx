import { Link } from 'react-router';
import { CheckCircle2, CircleDot, Loader2, XCircle } from 'lucide-react';
import type { RunProjection } from '../../lib/api/types';
import type { ToolCallLive } from '../../hooks/useSessionEvents';
import {
  ERROR_CODE_LABEL_VI,
  INTENT_LABEL_VI,
  LANE_LABEL_VI,
  RUN_STATUS_LABEL_VI,
} from '../../lib/labels';

function ToolTick({ call }: { call: ToolCallLive }) {
  const icon =
    call.state === 'running' ? (
      <Loader2 className="h-3 w-3 animate-spin" style={{ color: 'var(--accent-blue)' }} />
    ) : call.state === 'completed' ? (
      <CheckCircle2 className="h-3 w-3" style={{ color: 'var(--accent-green)' }} />
    ) : (
      <XCircle className="h-3 w-3" style={{ color: 'var(--accent-red)' }} />
    );
  return (
    <div className="flex items-center gap-1.5 text-xs">
      {icon}
      <span className="font-mono" style={{ color: 'var(--text-muted)' }}>
        {call.tool_name}
      </span>
      {call.duration_ms !== undefined && (
        <span style={{ color: 'var(--text-faint)' }}>{call.duration_ms}ms</span>
      )}
    </div>
  );
}

export function RunBlock({
  sessionId,
  run,
  liveToolCalls,
}: {
  sessionId: string;
  run: RunProjection;
  liveToolCalls: ToolCallLive[];
}) {
  const statusColor =
    run.status === 'completed'
      ? 'var(--accent-green)'
      : run.status === 'failed'
        ? 'var(--accent-red)'
        : run.status === 'cancelled'
          ? 'var(--text-faint)'
          : 'var(--accent-blue)';

  // W5-09: a failure_code is an internal enum. Render the fixed Vietnamese
  // sentence for it (deadline / predictor unavailable / runtime lost read
  // clearly) and keep the raw code as a mono suffix for audit. `cancelled`
  // is its own terminal state, not a failure — never show a red code for it.
  const failureLabel =
    run.failure_code && run.status !== 'cancelled'
      ? ERROR_CODE_LABEL_VI[run.failure_code] ?? null
      : null;

  return (
    <div
      id={`run-anchor-${run.run_id}`}
      className="my-2 rounded-xl border p-3"
      style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
      role="status"
      aria-live="polite"
      aria-atomic="true"
      aria-label={`Run ${INTENT_LABEL_VI[run.intent] ?? run.intent}: ${RUN_STATUS_LABEL_VI[run.status]}`}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-faint)' }}>
          <CircleDot className="h-3 w-3" style={{ color: statusColor }} />
          <span>
            run · {INTENT_LABEL_VI[run.intent] ?? run.intent} · {LANE_LABEL_VI[run.lane] ?? run.lane}
          </span>
        </div>
        <span className="text-xs font-medium" style={{ color: statusColor }}>
          {RUN_STATUS_LABEL_VI[run.status]}
        </span>
      </div>

      {liveToolCalls.length > 0 ? (
        <div className="space-y-1 border-t pt-2" style={{ borderColor: 'var(--border)' }}>
          {liveToolCalls.map((call) => (
            <ToolTick key={call.call_id} call={call} />
          ))}
        </div>
      ) : (
        run.status !== 'queued' && (
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
            {run.status === 'running' || run.status === 'validating'
              ? 'Model đang xử lý…'
              : 'Mở chi tiết để xem tool call.'}
          </p>
        )
      )}

      {run.status === 'cancelled' && (
        <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
          Run đã huỷ theo yêu cầu.
        </p>
      )}

      {run.failure_code && run.status !== 'cancelled' && (
        <p className="mt-2 text-xs" style={{ color: 'var(--accent-red)' }}>
          {failureLabel ?? 'Run thất bại.'}{' '}
          <span className="font-mono" style={{ color: 'var(--text-faint)' }}>
            ({run.failure_code})
          </span>
        </p>
      )}

      {run.recovery_of_run_id && (
        <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
          Đây là run khôi phục cho{' '}
          <Link
            to={`/s/${sessionId}/runs/${run.recovery_of_run_id}`}
            className="font-mono underline"
            style={{ color: 'var(--accent-blue)' }}
          >
            run trước đó
          </Link>
          .
        </p>
      )}

      {run.potentially_billed && (
        <p className="mt-2 text-xs" style={{ color: 'var(--accent-yellow)' }}>
          Nhà cung cấp có thể đã tính phí; mở chi tiết runtime để xem usage được báo.
        </p>
      )}

      <Link
        to={`/s/${sessionId}/runs/${run.run_id}`}
        className="mt-2 inline-block text-xs font-medium underline"
        style={{ color: 'var(--accent-blue)' }}
      >
        xem chi tiết run →
      </Link>
    </div>
  );
}
