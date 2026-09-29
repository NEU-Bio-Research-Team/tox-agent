import type { ToxAgentEvent } from '../../../shared/api/types';

/** One report build's progress, projected from `report.stage_changed` (WS09).
 *
 * The server sends semantic stage codes and counts, never prose a model wrote,
 * so everything a person reads here is a label this file owns. Two rules the
 * projection enforces rather than trusts the stream for:
 *
 * - **Monotonic.** A reconnect can replay an older event, and two stage events
 *   written in separate transactions can arrive out of order. A stage that has
 *   settled never goes back to running unless its attempt number went up —
 *   that is a retry, and a retry is shown as one.
 * - **Skipped is not pending.** A stage the build switched off is reached and
 *   recorded; showing it as "waiting" would tell the reader work is still due.
 */

export type ReportStageStatus = 'pending' | 'running' | 'completed' | 'skipped' | 'failed';

export interface ReportStageView {
  stage: string;
  status: ReportStageStatus;
  attempt: number;
  willRun: boolean;
}

export type ReportOutcome = 'completed' | 'completed_with_gaps' | 'failed' | 'cancelled';

export interface ReportProgress {
  buildId: string;
  total: number;
  completed: number;
  currentStage: string | null;
  stages: ReportStageView[];
  lastSequence: number;
  /** True once any stage has run more than once. */
  retried: boolean;
  outcome: ReportOutcome | null;
}

const SETTLED: ReadonlySet<ReportStageStatus> = new Set(['completed', 'skipped', 'failed']);

const TERMINAL: Record<string, ReportOutcome> = {
  'report.completed': 'completed',
  'report.completed_with_gaps': 'completed_with_gaps',
  'report.failed': 'failed',
  'report.cancelled': 'cancelled',
};

function asStatus(value: unknown): ReportStageStatus {
  return value === 'running' || value === 'completed' || value === 'skipped' || value === 'failed'
    ? value
    : 'pending';
}

function readStages(payload: Record<string, unknown>): ReportStageView[] | null {
  if (!Array.isArray(payload.stages)) return null;
  return payload.stages
    .filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
    .map((item) => ({
      stage: String(item.stage ?? ''),
      status: asStatus(item.status),
      attempt: Number(item.attempt ?? 0),
      willRun: item.will_run !== false,
    }))
    .filter((item) => item.stage);
}

function mergeStage(prior: ReportStageView | undefined, incoming: ReportStageView): ReportStageView {
  if (!prior) return incoming;
  if (incoming.attempt > prior.attempt) return incoming;
  if (incoming.attempt < prior.attempt) return prior;
  // Same attempt: a settled stage does not un-settle, and running does not
  // regress to pending because an older snapshot arrived late.
  if (SETTLED.has(prior.status)) return prior;
  if (incoming.status === 'pending') return prior;
  return incoming;
}

export function reduceReportProgress(
  prior: ReportProgress | undefined,
  event: ToxAgentEvent,
): ReportProgress | undefined {
  const outcome = TERMINAL[event.type];
  if (outcome) {
    if (!prior) return prior;
    return { ...prior, outcome, currentStage: null, lastSequence: Math.max(prior.lastSequence, event.sequence) };
  }
  if (event.type !== 'report.stage_changed') return prior;
  if (prior && event.sequence <= prior.lastSequence) return prior;

  const payload = event.payload ?? {};
  const buildId = prior?.buildId ?? event.entity_id;
  const incoming = readStages(payload);

  if (!incoming) {
    // The model-driven path emits `{stage}` alone. Keep what is known without
    // inventing a pipeline it did not report.
    const stage = String(payload.stage ?? '');
    return {
      buildId,
      total: prior?.total ?? 0,
      completed: prior?.completed ?? 0,
      currentStage: stage || (prior?.currentStage ?? null),
      stages: prior?.stages ?? [],
      lastSequence: event.sequence,
      retried: prior?.retried ?? false,
      outcome: prior?.outcome ?? null,
    };
  }

  const byStage = new Map((prior?.stages ?? []).map((item) => [item.stage, item]));
  const stages = incoming.map((item) => mergeStage(byStage.get(item.stage), item));
  const completed = stages.filter((item) => SETTLED.has(item.status) && item.status !== 'failed').length;
  const running = stages.find((item) => item.status === 'running');
  return {
    buildId,
    total: Math.max(Number(payload.total ?? stages.length), stages.length),
    completed: Math.max(completed, prior?.completed ?? 0),
    currentStage: running?.stage ?? null,
    stages,
    lastSequence: event.sequence,
    retried: (prior?.retried ?? false) || stages.some((item) => item.attempt > 1),
    outcome: prior?.outcome ?? null,
  };
}

/** Vietnamese labels owned by the product, one per semantic stage code. */
export const REPORT_STAGE_LABELS: Record<string, string> = {
  queued: 'Đang xếp hàng',
  preparing_analysis: 'Chuẩn bị phân tích',
  assembling_substance: 'Xác định danh tính hợp chất',
  assembling_predictions: 'Tổng hợp kết quả dự đoán',
  generating_explanations: 'Tạo giải thích mô hình',
  researching_evidence: 'Tìm bằng chứng tài liệu',
  synthesizing: 'Viết phần diễn giải',
  validating: 'Kiểm tra tính nhất quán',
  rendering: 'Xuất báo cáo',
};

export function reportStageLabel(stage: string | null | undefined): string {
  if (!stage) return 'Đang xử lý';
  return REPORT_STAGE_LABELS[stage] ?? stage;
}
