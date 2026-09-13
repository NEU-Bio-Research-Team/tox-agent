// @vitest-environment node
import { describe, expect, it } from 'vitest';
import type { EventType, ToxAgentEvent } from '../api/types';
import { reduceReportProgress, type ReportProgress } from './reportProgress';

const PIPELINE = [
  'preparing_analysis',
  'assembling_substance',
  'assembling_predictions',
  'generating_explanations',
  'researching_evidence',
  'synthesizing',
  'validating',
  'rendering',
];

function stageEvent(
  sequence: number,
  statuses: Record<string, [string, number?]>,
  overrides: Partial<ToxAgentEvent> = {},
): ToxAgentEvent {
  const stages = PIPELINE.map((stage) => ({
    stage,
    status: statuses[stage]?.[0] ?? 'pending',
    attempt: statuses[stage]?.[1] ?? (statuses[stage] ? 1 : 0),
    will_run: !['generating_explanations', 'researching_evidence'].includes(stage),
  }));
  return {
    event_id: `evt-${sequence}`,
    session_id: 'ses_test',
    sequence,
    type: 'report.stage_changed' as EventType,
    entity_type: 'report_build',
    entity_id: 'rpb_test',
    entity_version: 1,
    run_id: 'run_test',
    occurred_at: '2026-09-13T00:00:00Z',
    payload: { stages, total: stages.length },
    ...overrides,
  };
}

function fold(events: ToxAgentEvent[]): ReportProgress | undefined {
  return events.reduce<ReportProgress | undefined>(reduceReportProgress, undefined);
}

describe('reduceReportProgress', () => {
  it('follows the stages the server reports, with skipped distinct from pending', () => {
    const progress = fold([
      stageEvent(1, { preparing_analysis: ['running'] }),
      stageEvent(2, { preparing_analysis: ['completed'] }),
      stageEvent(3, { preparing_analysis: ['completed'], generating_explanations: ['skipped'] }),
      stageEvent(4, {
        preparing_analysis: ['completed'],
        generating_explanations: ['skipped'],
        synthesizing: ['running'],
      }),
    ]);
    expect(progress?.currentStage).toBe('synthesizing');
    expect(progress?.completed).toBe(2);
    expect(progress?.stages.find((s) => s.stage === 'generating_explanations')?.status).toBe('skipped');
    expect(progress?.stages.find((s) => s.stage === 'validating')?.status).toBe('pending');
  });

  it('ignores a replayed or late event instead of moving backwards', () => {
    const newer = stageEvent(5, { preparing_analysis: ['completed'], synthesizing: ['completed'] });
    const older = stageEvent(3, { preparing_analysis: ['running'] });
    const state = fold([newer]);
    expect(reduceReportProgress(state, older)).toBe(state);
    // Same attempt, later sequence, but a stale view of one stage.
    const stale = stageEvent(6, { preparing_analysis: ['running'], synthesizing: ['completed'] });
    const next = reduceReportProgress(state, stale);
    expect(next?.stages.find((s) => s.stage === 'preparing_analysis')?.status).toBe('completed');
  });

  it('shows a retry as a retry, not as a regression', () => {
    const progress = fold([
      stageEvent(1, { synthesizing: ['failed', 1] }),
      stageEvent(2, { synthesizing: ['running', 2] }),
    ]);
    expect(progress?.stages.find((s) => s.stage === 'synthesizing')).toMatchObject({
      status: 'running',
      attempt: 2,
    });
    expect(progress?.retried).toBe(true);
  });

  it('records the terminal outcome and clears the current stage', () => {
    const progress = fold([
      stageEvent(1, { rendering: ['running'] }),
      { ...stageEvent(2, {}), type: 'report.completed_with_gaps' as EventType, payload: {} },
    ]);
    expect(progress?.outcome).toBe('completed_with_gaps');
    expect(progress?.currentStage).toBeNull();
  });

  it('keeps the legacy single-stage event readable without inventing a pipeline', () => {
    const progress = fold([{ ...stageEvent(1, {}), payload: { stage: 'synthesizing' } }]);
    expect(progress?.currentStage).toBe('synthesizing');
    expect(progress?.stages).toEqual([]);
  });
});
