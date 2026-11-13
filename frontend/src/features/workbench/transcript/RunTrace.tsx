import { useEffect, useMemo, useRef, useState, type ComponentType } from 'react';
import { Link } from 'react-router';
import {
  BookOpen,
  Brain,
  Check,
  ChevronRight,
  Cpu,
  FileSearch,
  FileText,
  FlaskConical,
  Loader2,
  Search,
  Send,
  ShieldAlert,
  ShieldCheck,
  Wrench,
  X,
  type LucideProps,
} from 'lucide-react';
import type { ActivityLive, RunProjection } from '../../../shared/api/types';
import { getDeveloperModeEnabled } from '../../../shared/lib/preferences';
import type { TraceEntry } from '../store/sessionEventsReducer';

// The run's live log, the way an agent harness shows it: what the turn runs on,
// which skills it was given, the model's reasoning as it streams, every tool
// call with its arguments and timing, and the validator's verdicts. Open while
// the run works, folded into a one-line summary once it is done.
//
// Reasoning is the model's own unvalidated text. It is shown as thinking, never
// as an answer — the answer still appears only after validation.

type Icon = ComponentType<LucideProps>;

const RUNNING = new Set(['queued', 'running', 'validating']);

const PHASES: Record<string, string> = {
  investigator: 'điều tra',
  workflow_acquisition: 'thu thập dữ liệu',
  workflow_synthesis: 'tổng hợp',
  hard_correction: 'sửa lỗi',
  review_initial: 'rà soát',
  semantic_repair: 'sửa nội dung',
  review_final: 'rà soát cuối',
};

function toolIcon(name: string): Icon {
  if (name.includes('skill') || name.includes('procedure')) return BookOpen;
  if (name.startsWith('search')) return Search;
  if (name.includes('evidence') || name.includes('read_')) return FileSearch;
  if (name.includes('report') || name.includes('draft')) return FileText;
  if (name.includes('analysis') || name.includes('explanation') || name.includes('attribution') || name.includes('screening') || name.includes('activity')) return FlaskConical;
  if (name.startsWith('submit')) return Send;
  return Wrench;
}

function toolVerb(name: string): string {
  if (name.includes('skill')) return 'Đang đọc skill';
  if (name.startsWith('search')) return 'Đang tìm bằng chứng';
  if (name.includes('evidence') || name.startsWith('read_')) return 'Đang đọc nguồn';
  if (name.startsWith('submit')) return 'Đang nộp câu trả lời';
  if (name.includes('report') || name.includes('draft')) return 'Đang soạn báo cáo';
  if (name.includes('procedure')) return 'Đang chạy quy trình';
  return 'Đang gọi tool';
}

/** The arguments worth a glance: a skill's id, a query, else the JSON. */
function argsSummary(args: string | null): string | null {
  if (!args) return null;
  try {
    const value = JSON.parse(args.endsWith('…') ? '' : args) as Record<string, unknown>;
    for (const key of ['skill_id', 'procedure_id', 'query', 'evidence_id', 'smiles', 'analysis_id']) {
      if (typeof value[key] === 'string') return String(value[key]);
    }
  } catch { /* a truncated preview is shown as is */ }
  return args;
}

function seconds(ms: number): string {
  if (ms < 1000) return `${Math.max(0, Math.round(ms))}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`;
  const m = Math.floor(ms / 60_000);
  return `${m}m ${Math.round((ms % 60_000) / 1000)}s`;
}

function between(from: string, to: string | undefined, now: number): number {
  return (to ? Date.parse(to) : now) - Date.parse(from);
}

function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

function Dot({ icon: IconComponent, tone = 'muted', spinning = false }: { icon: Icon; tone?: 'muted' | 'purple' | 'green' | 'red' | 'amber'; spinning?: boolean }) {
  const color = { muted: 'var(--ink-tertiary)', purple: 'var(--purple-600)', green: 'var(--success)', red: 'var(--danger)', amber: 'var(--warning)' }[tone];
  return (
    <span className="relative z-10 mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border" style={{ background: 'var(--surface-solid)', borderColor: 'var(--line)', color }}>
      {spinning ? <Loader2 className="h-3 w-3 animate-spin motion-reduce:animate-none" /> : <IconComponent className="h-3 w-3" />}
    </span>
  );
}

function Chip({ children, mono = false, title }: { children: React.ReactNode; mono?: boolean; title?: string }) {
  return (
    <span title={title} className={`inline-flex max-w-full items-center truncate rounded-md px-1.5 py-0.5 text-2xs ${mono ? 'font-mono' : ''}`} style={{ background: 'var(--surface-muted)', color: 'var(--ink-secondary)' }}>
      {children}
    </span>
  );
}

function ReasoningRow({ entry, live, now }: { entry: Extract<TraceEntry, { kind: 'reasoning' }>; live: boolean; now: number }) {
  const streaming = live && !entry.done;
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (streaming && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [entry.text, streaming]);
  const took = between(entry.at, streaming ? undefined : entry.updatedAt, now);
  const firstLine = entry.text.trim().split('\n').find((line) => line.trim()) ?? '';

  return (
    <div className="min-w-0 flex-1">
      <button type="button" onClick={() => setOpen((v) => !v)} disabled={streaming} className="flex w-full items-center gap-1.5 text-left text-13" aria-expanded={streaming || open}>
        <span className={streaming ? 'trace-shimmer font-medium' : 'font-medium'} style={streaming ? undefined : { color: 'var(--ink-secondary)' }}>
          {streaming ? 'Đang suy nghĩ' : `Đã suy nghĩ ${seconds(took)}`}
        </span>
        {!streaming && <ChevronRight className={`h-3 w-3 transition-transform ${open ? 'rotate-90' : ''}`} style={{ color: 'var(--ink-tertiary)' }} />}
        <span className="ml-auto text-2xs uppercase tracking-wide text-ink-tertiary" title="Suy nghĩ của model chưa qua kiểm chứng; câu trả lời cuối mới được kiểm chứng.">chưa kiểm chứng</span>
      </button>
      {streaming ? (
        <div ref={box} className="trace-fade-top mt-1 max-h-28 overflow-hidden whitespace-pre-wrap text-xs italic leading-relaxed text-ink-tertiary">
          {entry.text}
        </div>
      ) : open ? (
        <div className="mt-1 max-h-80 overflow-y-auto whitespace-pre-wrap border-l-2 pl-3 text-xs italic leading-relaxed animate-in fade-in duration-200 text-ink-secondary">
          {entry.truncated && <span className="not-italic text-ink-tertiary">… (phần đầu đã lược) </span>}
          {entry.text}
        </div>
      ) : (
        firstLine && <p className="mt-0.5 truncate text-xs italic text-ink-tertiary">{firstLine}</p>
      )}
    </div>
  );
}

function EntryRow({ entry, live, now }: { entry: TraceEntry; live: boolean; now: number }) {
  switch (entry.kind) {
    case 'turn':
      return (
        <>
          <Dot icon={Cpu} />
          <div className="min-w-0 flex-1 text-13">
            <p style={{ color: 'var(--ink-secondary)' }}>
              Bắt đầu lượt {entry.phase ? (PHASES[entry.phase] ?? entry.phase) : 'agent'}
            </p>
            <div className="mt-1 flex flex-wrap gap-1">
              <Chip mono>{entry.modelId}</Chip>
              <Chip>{entry.toolCount} tool</Chip>
              {entry.promptTokens ? <Chip>~{(entry.promptTokens / 1000).toFixed(1)}k token ngữ cảnh</Chip> : null}
            </div>
          </div>
        </>
      );
    case 'skills':
      return (
        <>
          <Dot icon={BookOpen} tone="purple" />
          <div className="min-w-0 flex-1 text-13">
            <p style={{ color: 'var(--ink-secondary)' }}>
              Nạp {entry.skills.length} skill {entry.mode === 'static' ? 'vào prompt' : '(mục lục — đọc khi cần)'}
            </p>
            <div className="mt-1 flex flex-wrap gap-1">
              {entry.skills.map((skill) => (
                <Chip key={skill.skill_id} mono title={skill.description}>{skill.skill_id}{skill.version ? `@${skill.version}` : ''}</Chip>
              ))}
            </div>
          </div>
        </>
      );
    case 'reasoning':
      return (
        <>
          <Dot icon={Brain} tone="purple" spinning={live && !entry.done} />
          <ReasoningRow entry={entry} live={live} now={now} />
        </>
      );
    case 'tool': {
      const running = entry.state === 'running' && live;
      const summary = argsSummary(entry.args);
      const took = entry.durationMs ?? (entry.endedAt ? between(entry.at, entry.endedAt, now) : running ? between(entry.at, undefined, now) : null);
      return (
        <>
          <Dot icon={toolIcon(entry.toolName)} tone={entry.state === 'failed' ? 'red' : running ? 'purple' : 'muted'} spinning={running} />
          <div className="flex min-w-0 flex-1 items-center gap-2 text-13">
            <span className={`shrink-0 font-mono text-xs ${running ? 'trace-shimmer' : ''}`} style={running ? undefined : { color: 'var(--ink)' }}>{entry.toolName}</span>
            {summary && <span className="min-w-0 truncate font-mono text-2xs text-ink-tertiary" title={entry.args ?? undefined}>{summary}</span>}
            <span className="ml-auto flex shrink-0 items-center gap-1 text-2xs tabular-nums text-ink-tertiary">
              {entry.state === 'failed' && <span style={{ color: 'var(--danger)' }}>{entry.errorCode ?? 'lỗi'}</span>}
              {took !== null && seconds(took)}
              {entry.state === 'completed' && <Check className="h-3 w-3 text-success" />}
              {entry.state === 'failed' && <X className="h-3 w-3 text-danger" />}
            </span>
          </div>
        </>
      );
    }
    case 'rejected':
      return (
        <>
          <Dot icon={ShieldAlert} tone="amber" />
          <p className="flex-1 text-13 text-ink-secondary">
            Trình kiểm chứng từ chối câu trả lời ({entry.violations} vi phạm){live ? ' — model đang sửa' : ''}
          </p>
        </>
      );
    case 'validating':
      return (
        <>
          <Dot icon={ShieldCheck} tone="green" />
          <p className="flex-1 text-13 text-ink-secondary">Kiểm chứng câu trả lời với bằng chứng</p>
        </>
      );
  }
}

const ACTIVITY_LABELS: Record<string, string> = {
  'activity.searching_literature': 'Đang tìm các nghiên cứu liên quan',
  'activity.reading_sources': 'Đang đọc những nguồn phù hợp nhất',
  'activity.running_predictor': 'Đang chạy mô hình dự đoán độc tính',
  'activity.recognizing_structure': 'Đang nhận diện cấu trúc',
  'activity.reviewing_results': 'Đang kiểm tra kết quả',
  'activity.inspecting_factors': 'Đang xem xét các yếu tố ảnh hưởng',
  'activity.synthesizing': 'Đang tổng hợp bằng chứng',
};

function liveHeadline(entries: TraceEntry[], activities: ActivityLive[], status: string): string {
  if (status === 'queued') return 'Đang xếp hàng';
  if (status === 'validating') return 'Đang kiểm chứng câu trả lời';
  const last = entries.at(-1);
  if (last?.kind === 'reasoning' && !last.done) return 'Đang suy nghĩ';
  if (last?.kind === 'tool' && last.state === 'running') return toolVerb(last.toolName);
  if (last?.kind === 'rejected') return 'Đang sửa câu trả lời';
  const active = [...activities].reverse().find((item) => item.status === 'started' || item.status === 'progress');
  if (active && ACTIVITY_LABELS[active.label_key]) return ACTIVITY_LABELS[active.label_key];
  return entries.length ? 'Đang xử lý' : 'Đang khởi động agent';
}

export function RunTrace({
  run,
  entries,
  activities,
  sessionId,
}: {
  run: RunProjection;
  entries: TraceEntry[];
  activities: ActivityLive[];
  sessionId: string;
}) {
  const live = RUNNING.has(run.status);
  const now = useNow(live);
  const [expanded, setExpanded] = useState(live);
  const touched = useRef(false);

  // Open while working, folded once done — unless the user chose otherwise.
  useEffect(() => {
    if (!touched.current) setExpanded(live);
  }, [live]);

  const stats = useMemo(() => ({
    tools: entries.filter((item) => item.kind === 'tool').length,
    thoughts: entries.filter((item) => item.kind === 'reasoning').length,
    model: [...entries].reverse().find((item): item is Extract<TraceEntry, { kind: 'turn' }> => item.kind === 'turn')?.modelId ?? null,
  }), [entries]);

  const start = run.started_at ?? run.created_at;
  const elapsed = between(start, live ? undefined : (run.ended_at ?? undefined), now);
  const failed = run.status === 'failed';
  const cancelled = run.status === 'cancelled';

  const headline = live
    ? liveHeadline(entries, activities, run.status)
    : failed
      ? 'Không thể hoàn tất phản hồi'
      : cancelled
        ? 'Đã dừng theo yêu cầu'
        : `Đã xử lý trong ${seconds(elapsed)}`;

  const details = getDeveloperModeEnabled()
    ? <Link className="text-2xs underline text-ink-tertiary" to={`/s/${sessionId}/runs/${run.run_id}`}>Run details</Link>
    : null;

  return (
    <div className="my-2 max-w-[720px]" role={live ? 'status' : undefined} aria-live={live ? 'polite' : undefined}>
      <button
        type="button"
        onClick={() => { touched.current = true; setExpanded((v) => !v); }}
        aria-expanded={expanded}
        disabled={entries.length === 0}
        className="group flex w-full items-center gap-2 rounded-lg px-1 py-1 text-left text-sm transition-colors hover:bg-[var(--surface-hover)] disabled:hover:bg-transparent"
      >
        {live ? (
          <Loader2 className="h-4 w-4 shrink-0 animate-spin motion-reduce:animate-none" style={{ color: 'var(--purple-600)' }} />
        ) : failed ? (
          <X className="h-4 w-4 shrink-0 text-danger" />
        ) : (
          <Check className="h-4 w-4 shrink-0" style={{ color: cancelled ? 'var(--ink-tertiary)' : 'var(--success)' }} />
        )}
        <span className={live ? 'trace-shimmer font-medium' : 'font-medium'} style={live ? undefined : { color: failed ? 'var(--danger)' : 'var(--ink-secondary)' }}>
          {headline}{live ? '…' : ''}
        </span>
        {!live && entries.length > 0 && (
          <span className="truncate text-xs text-ink-tertiary">
            {[stats.tools ? `${stats.tools} tool` : null, stats.thoughts ? `${stats.thoughts} lần suy nghĩ` : null].filter(Boolean).join(' · ')}
          </span>
        )}
        <span className="ml-auto flex shrink-0 items-center gap-2 text-xs tabular-nums text-ink-tertiary">
          {stats.model && <span className="hidden max-w-[160px] truncate font-mono sm:inline">{stats.model}</span>}
          {live && seconds(elapsed)}
          {entries.length > 0 && <ChevronRight className={`h-3.5 w-3.5 transition-transform duration-200 ${expanded ? 'rotate-90' : ''}`} />}
        </span>
      </button>

      <div className={`grid transition-[grid-template-rows] duration-300 ease-out motion-reduce:transition-none ${expanded && entries.length ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}`}>
        <div className="overflow-hidden">
          <ol className="relative ml-[9px] mt-1 space-y-2.5 border-l pb-1 pl-4">
            {entries.map((entry) => (
              <li key={`${entry.kind}:${entry.id}`} className="-ml-[27px] flex items-start gap-2.5 animate-in fade-in slide-in-from-bottom-1 duration-300 motion-reduce:animate-none">
                <EntryRow entry={entry} live={live} now={now} />
              </li>
            ))}
          </ol>
          {(failed || details) && (
            <div className="ml-6 mt-1 flex items-center gap-2 text-xs">
              {failed && <span style={{ color: 'var(--danger)' }}>{run.failure_code ? `Lỗi: ${run.failure_code}. ` : ''}Hãy thử lại.</span>}
              {details}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
