import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it } from 'vitest';
import type { RunProjection } from '../../../shared/api/types';
import type { TraceEntry } from '../store/sessionEventsReducer';
import { RunTrace } from './RunTrace';

afterEach(cleanup);

function run(status: RunProjection['status']): RunProjection {
  return {
    run_id: 'run_1', status, lane: 'agentic', intent: 'decision_support', trigger_message_id: 'msg_1',
    runtime_binding_id: null, recovery_of_run_id: null, failure_code: null, potentially_billed: false,
    deadline_at: '2026-11-07T00:10:00Z', created_at: '2026-11-07T00:00:00Z', started_at: '2026-11-07T00:00:00Z',
    ended_at: status === 'running' ? null : '2026-11-07T00:00:34Z',
  } as RunProjection;
}

const ENTRIES: TraceEntry[] = [
  { kind: 'turn', id: 't', at: '2026-11-07T00:00:01Z', phase: 'investigator', providerId: 'openai', modelId: 'gpt-5.1', toolCount: 14, promptTokens: 8200 },
  { kind: 'skills', id: 's', at: '2026-11-07T00:00:01Z', mode: 'dynamic', skills: [{ skill_id: 'assess-conflicting-evidence', version: '1' }] },
  { kind: 'reasoning', id: 'r', at: '2026-11-07T00:00:02Z', updatedAt: '2026-11-07T00:00:05Z', text: 'Check hERG first.\nThen Tox21.', done: true, truncated: false },
  { kind: 'tool', id: 'c', at: '2026-11-07T00:00:06Z', toolName: 'search_toxicology_evidence', args: '{"query": "hERG aspirin"}', state: 'completed', durationMs: 1200 },
];

const renderTrace = (status: RunProjection['status'], entries = ENTRIES) =>
  render(<MemoryRouter><RunTrace run={run(status)} entries={entries} activities={[]} sessionId="ses_1" /></MemoryRouter>);

describe('RunTrace', () => {
  it('shows every step while the run works', () => {
    renderTrace('running', [...ENTRIES, { kind: 'tool', id: 'd', at: '2026-11-07T00:00:08Z', toolName: 'read_evidence_text', args: null, state: 'running' }]);
    expect(screen.getByText(/Đang đọc nguồn/)).toBeTruthy();
    expect(screen.getAllByText('gpt-5.1').length).toBeGreaterThan(0);
    expect(screen.getByText('assess-conflicting-evidence@1')).toBeTruthy();
    expect(screen.getByText('search_toxicology_evidence')).toBeTruthy();
    expect(screen.getByText('hERG aspirin')).toBeTruthy();
    expect(screen.getByText('1.2s')).toBeTruthy();
  });

  it('labels reasoning as unverified and opens it on click', () => {
    renderTrace('running');
    expect(screen.getByText('chưa kiểm chứng')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /Đã suy nghĩ 3.0s/ }));
    expect(screen.getByText(/Then Tox21/)).toBeTruthy();
  });

  it('folds into a summary once the run is done', () => {
    renderTrace('completed');
    const header = screen.getByRole('button', { name: /Đã xử lý trong 34s/ });
    expect(header.getAttribute('aria-expanded')).toBe('false');
    expect(screen.getByText('1 tool · 1 lần suy nghĩ')).toBeTruthy();
    fireEvent.click(header);
    expect(header.getAttribute('aria-expanded')).toBe('true');
  });
});
