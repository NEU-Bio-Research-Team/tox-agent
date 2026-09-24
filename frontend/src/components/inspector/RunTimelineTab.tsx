import { CheckCircle2, Loader2, XCircle } from 'lucide-react';
import type { RunDetail } from '../../lib/api/types';
import { RUN_STATUS_LABEL_VI } from '../../lib/labels';

function msBetween(a: string, b: string): number {
  return new Date(b).getTime() - new Date(a).getTime();
}

export function RunTimelineTab({ run }: { run: RunDetail }) {
  const anchor = run.started_at ?? run.created_at;
  const end = run.ended_at ?? new Date().toISOString();
  const totalMs = Math.max(msBetween(anchor, end), 1);
  const hasTimestamps = run.tool_calls.every((call) => call.started_at);

  const totalToolMs = run.tool_calls.reduce((sum, call) => sum + (call.duration_ms ?? 0), 0);
  const thinkingPct = totalMs > 0 ? Math.max(0, ((totalMs - totalToolMs) / totalMs) * 100) : 0;

  return (
    <div className="space-y-4">
      <div
        className="rounded-lg p-3 text-xs"
        style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}
      >
        {(totalMs / 1000).toFixed(1)}s tổng · {totalToolMs}ms trong tool · {thinkingPct.toFixed(2)}% là model/runtime đang xử lý
      </div>

      {!hasTimestamps && (
        <p className="text-xs italic" style={{ color: 'var(--text-faint)' }}>
          Run này không có mốc thời gian tool call đầy đủ — thứ tự dưới đây đúng, tỉ lệ độ dài có thể không chính xác.
        </p>
      )}

      <div className="space-y-2">
        {run.tool_calls.map((call) => {
          const offsetPct = call.started_at ? (msBetween(anchor, call.started_at) / totalMs) * 100 : 0;
          const widthPct = call.duration_ms ? Math.max((call.duration_ms / totalMs) * 100, 0.5) : 1;
          const icon =
            call.status === 'completed' ? (
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--accent-green)' }} />
            ) : call.status === 'error' || call.status === 'denied' ? (
              <XCircle className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--accent-red)' }} />
            ) : (
              <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" style={{ color: 'var(--accent-blue)' }} />
            );

          return (
            <div key={call.call_id} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 font-mono" style={{ color: 'var(--text)' }}>
                  {icon}
                  {call.tool_name}
                  {call.error_code && <span style={{ color: 'var(--accent-red)' }}>· {call.error_code}</span>}
                </span>
                <span style={{ color: 'var(--text-faint)' }}>{call.duration_ms ?? '—'}ms</span>
              </div>
              <div className="relative h-1.5 w-full rounded-full" style={{ backgroundColor: 'var(--border)' }}>
                <div
                  className="absolute h-full rounded-full"
                  style={{ left: `${offsetPct}%`, width: `${widthPct}%`, backgroundColor: 'var(--accent-blue)' }}
                />
              </div>
            </div>
          );
        })}
        {run.tool_calls.length === 0 && (
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
            Chưa có tool call nào.
          </p>
        )}
      </div>

      <p className="border-t pt-2 text-xs" style={{ borderColor: 'var(--border)', color: 'var(--text-faint)' }}>
        {RUN_STATUS_LABEL_VI[run.status]} · deadline {new Date(run.deadline_at).toLocaleString('vi-VN')}
      </p>
    </div>
  );
}
