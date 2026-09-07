import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it } from 'vitest';
import { RunBlock } from './RunBlock';
import type { RunProjection } from '../../lib/api/types';

afterEach(cleanup);

function makeRun(overrides: Partial<RunProjection> = {}): RunProjection {
  return {
    run_id: 'run_1', status: 'validating', lane: 'agentic', intent: 'report_qa',
    trigger_message_id: 'msg_1', runtime_binding_id: null, recovery_of_run_id: null,
    failure_code: null, potentially_billed: false, deadline_at: '2026-09-06T01:00:00Z',
    created_at: '2026-09-06T00:00:00Z', started_at: '2026-09-06T00:00:01Z', ended_at: null,
    ...overrides,
  };
}

function renderRun(run: RunProjection) {
  return render(
    <MemoryRouter>
      <RunBlock sessionId="ses_1" run={run} liveToolCalls={[]} />
    </MemoryRouter>,
  );
}

describe('RunBlock', () => {
  it('announces the durable run state to a screen reader', () => {
    renderRun(makeRun());

    const status = screen.getByRole('status');
    expect(status).toHaveAttribute('aria-live', 'polite');
    expect(status).toHaveAttribute('aria-atomic', 'true');
    expect(status).toHaveAccessibleName('Run hỏi báo cáo: đang kiểm định');
  });

  it('renders a failure_code as its fixed Vietnamese sentence plus the raw code', () => {
    renderRun(makeRun({ status: 'failed', failure_code: 'deadline_exceeded', ended_at: '2026-09-06T00:05:00Z' }));

    expect(screen.getByText(/Run vượt quá thời hạn cho phép\./)).toBeInTheDocument();
    expect(screen.getByText('(deadline_exceeded)')).toBeInTheDocument();
  });

  it('treats a cancelled run as a terminal state, not a red failure', () => {
    renderRun(makeRun({ status: 'cancelled', failure_code: null }));

    expect(screen.getByText('Run đã huỷ theo yêu cầu.')).toBeInTheDocument();
    // A cancelled run carries no failure paragraph and no raw code suffix.
    expect(screen.queryByText('Run thất bại.')).not.toBeInTheDocument();
  });

  it('links a recovery run back to the run it recovers, even when it completed', () => {
    renderRun(makeRun({ status: 'completed', recovery_of_run_id: 'run_origin', ended_at: '2026-09-06T00:05:00Z' }));

    const link = screen.getByRole('link', { name: 'run trước đó' });
    expect(link).toHaveAttribute('href', '/s/ses_1/runs/run_origin');
  });
});
