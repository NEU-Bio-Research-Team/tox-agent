import { Check, Loader2, RotateCcw } from 'lucide-react';
import { Link } from 'react-router';
import { getDeveloperModeEnabled } from '../../lib/preferences';
import type { ActivityLive } from '../../lib/api/types';
import type { ToolCallLive } from '../../hooks/useSessionEvents';

const LABELS: Record<string, string> = {
  'activity.searching_literature': 'Đang tìm các nghiên cứu liên quan…',
  'activity.reading_sources': 'Đang đọc những nguồn phù hợp nhất…',
  'activity.running_predictor': 'Đang chạy mô hình dự đoán độc tính…',
  'activity.recognizing_structure': 'Đang nhận diện cấu trúc…',
  'activity.reviewing_results': 'Đang kiểm tra kết quả…',
  'activity.synthesizing': 'Đang tổng hợp bằng chứng…',
  'activity.processing': 'Đang xử lý yêu cầu…',
};

function fallback(tools: ToolCallLive[]): string {
  const name = tools.at(-1)?.tool_name ?? '';
  if (name.includes('search')) return LABELS['activity.searching_literature'];
  if (name.includes('evidence')) return LABELS['activity.reading_sources'];
  if (name.includes('predict')) return LABELS['activity.running_predictor'];
  return LABELS['activity.processing'];
}

/** A product activity line. Raw trace remains available in the run inspector. */
export function ActivityPresence({ activities, tools, status, sessionId, runId }: {
  activities: ActivityLive[];
  tools: ToolCallLive[];
  status: string;
  sessionId?: string;
  runId?: string;
}) {
  const active = [...activities].reverse().find((item) => item.status === 'started' || item.status === 'progress');
  const completed = [...activities].reverse().find((item) => item.status === 'completed');
  const label = active ? LABELS[active.label_key] ?? LABELS['activity.processing'] : tools.length ? fallback(tools) : null;
  const details = getDeveloperModeEnabled() && sessionId && runId ? <Link className="ml-2 text-xs underline" to={`/s/${sessionId}/runs/${runId}`}>Run details</Link> : null;
  if (status === 'failed') return <p className="my-2 text-sm" style={{ color: 'var(--accent-red)' }}>Không thể tiếp tục phản hồi. Hãy thử lại.{details}</p>;
  if (status === 'cancelled') return <p className="my-2 text-sm" style={{ color: 'var(--text-muted)' }}>Đã dừng theo yêu cầu.</p>;
  if (label && (status === 'queued' || status === 'running' || status === 'validating')) {
    return <div className="my-2 flex items-center gap-2 text-sm motion-reduce:transition-none" role="status" aria-live="polite" style={{ color: 'var(--text-muted)' }}>
      <Loader2 className="h-4 w-4 animate-spin motion-reduce:animate-none" style={{ color: 'var(--purple-600)' }} />
      <span>{label}</span>{details}
    </div>;
  }
  if (completed && status === 'completed') return <div className="my-2 flex items-center gap-2 text-xs" style={{ color: 'var(--text-faint)' }}><Check className="h-3.5 w-3.5" />Đã hoàn tất</div>;
  if (status === 'queued') return <div className="my-2 flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}><RotateCcw className="h-4 w-4 animate-spin motion-reduce:animate-none" />Đang phân tích yêu cầu…</div>;
  return null;
}
