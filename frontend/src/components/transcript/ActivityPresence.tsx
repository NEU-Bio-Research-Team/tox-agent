import { Check, Loader2, RotateCcw } from 'lucide-react';
import { Link } from 'react-router';
import { getDeveloperModeEnabled } from '../../lib/preferences';
import type { ActivityLive } from '../../lib/api/types';
import type { ToolCallLive } from '../../hooks/useSessionEvents';
import { Popover, PopoverContent, PopoverTrigger } from '../ui/popover';
import { Button } from '../ui/button';

const LABELS: Record<string, string> = {
  'activity.searching_literature': 'Đang tìm các nghiên cứu liên quan…',
  'activity.reading_sources': 'Đang đọc những nguồn phù hợp nhất…',
  'activity.running_predictor': 'Đang chạy mô hình dự đoán độc tính…',
  'activity.recognizing_structure': 'Đang nhận diện cấu trúc…',
  'activity.reviewing_results': 'Đang kiểm tra kết quả…',
  'activity.inspecting_factors': 'Đang xem xét các yếu tố ảnh hưởng…',
  'activity.synthesizing': 'Đang tổng hợp bằng chứng…',
  'activity.processing': 'Đang xử lý yêu cầu…',
};

function fallback(tools: ToolCallLive[]): string {
  const name = tools.at(-1)?.tool_name ?? '';
  const repeated = tools.filter((tool) => tool.tool_name === name).length;
  if (name.includes('search')) return repeated > 1 ? `Đang tìm kiếm trên nhiều nguồn… (${repeated})` : LABELS['activity.searching_literature'];
  if (name.includes('evidence')) return repeated > 1 ? `Đang đọc các nghiên cứu phù hợp… (${repeated})` : LABELS['activity.reading_sources'];
  if (name.includes('predict')) return LABELS['activity.running_predictor'];
  return LABELS['activity.processing'];
}

function ActivityHistory({ activities }: { activities: ActivityLive[] }) {
  const grouped = new Map<string, { label: string; count: number; failed: boolean }>();
  for (const activity of activities) {
    const label = LABELS[activity.label_key] ?? LABELS['activity.processing'];
    const current = grouped.get(label) ?? { label, count: 0, failed: false };
    current.count += 1;
    current.failed ||= activity.status === 'failed';
    grouped.set(label, current);
  }
  if (grouped.size === 0) return null;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="h-auto px-1.5 py-0 text-xs" style={{ color: 'var(--text-faint)' }}>
          {activities.length} hoạt động
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72 space-y-2">
        <p className="text-sm font-medium">Tiến trình đã thực hiện</p>
        <ul className="space-y-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
          {[...grouped.values()].map((item) => (
            <li key={item.label} className="flex items-start gap-2">
              <Check className="mt-0.5 h-3.5 w-3.5 shrink-0" style={{ color: item.failed ? 'var(--accent-red)' : 'var(--accent-green)' }} />
              <span>{item.label}{item.count > 1 ? ` × ${item.count}` : ''}</span>
            </li>
          ))}
        </ul>
      </PopoverContent>
    </Popover>
  );
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
      <span>{label}</span><ActivityHistory activities={activities} />{details}
    </div>;
  }
  if (completed && status === 'completed') return <div className="my-2 flex items-center gap-2 text-xs" style={{ color: 'var(--text-faint)' }}><Check className="h-3.5 w-3.5" />Đã hoàn tất<ActivityHistory activities={activities} /></div>;
  if (status === 'queued') return <div className="my-2 flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}><RotateCcw className="h-4 w-4 animate-spin motion-reduce:animate-none" />Đang phân tích yêu cầu…</div>;
  return null;
}
