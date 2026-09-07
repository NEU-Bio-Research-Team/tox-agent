import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { RunDetail } from '../../lib/api/types';
import { RuntimeManifestTab } from './RuntimeManifestTab';

function run(overrides: Partial<RunDetail> = {}): RunDetail {
  return {
    run_id: 'run_1',
    status: 'completed',
    lane: 'agentic',
    intent: 'report_qa',
    trigger_message_id: 'msg_1',
    runtime_binding_id: 'bind_1',
    recovery_of_run_id: null,
    failure_code: null,
    potentially_billed: false,
    deadline_at: '2026-09-06T00:00:00Z',
    created_at: '2026-09-06T00:00:00Z',
    started_at: '2026-09-06T00:00:00Z',
    ended_at: '2026-09-06T00:00:01Z',
    runtime: {
      runtime_binding_id: 'bind_1',
      runtime_kind: 'opencode',
      runtime_version: '1.17.11',
      provider_id: 'provider_1',
      model_id: 'model_1',
      profile_hash: 'profile',
      tool_schema_hash: 'tools',
      system_prompt_hash: 'prompt',
    },
    usage: { status: 'unknown', events: [] },
    tool_calls: [],
    ...overrides,
  };
}

describe('RuntimeManifestTab', () => {
  it('shows unknown usage as unknown rather than a zero cost', () => {
    render(<RuntimeManifestTab run={run()} />);

    expect(screen.getByText(/không biết.*không phải chi phí bằng 0/i)).toBeInTheDocument();
    expect(screen.queryByText(/hiện luôn false/i)).toBeNull();
  });

  it('keeps reported zero distinct from missing usage and warns about uncertain billing', () => {
    render(<RuntimeManifestTab run={run({
      potentially_billed: true,
      usage: {
        status: 'reported',
        events: [{
          usage_event_id: 'usage_1',
          runtime_binding_id: 'bind_1',
          provider_id: 'provider_1',
          model_id: 'model_1',
          reported_at: '2026-09-06T00:00:00Z',
          tokens: { input: 0, output: 0, reasoning: null, cache_read: null, cache_write: null, total: 0 },
          cost: { amount: '0', currency: 'USD' },
        }],
      },
    })} />);

    expect(screen.getByText(/có thể đã nhận lượt này/i)).toBeInTheDocument();
    expect(screen.getByText('0 USD')).toBeInTheDocument();
    expect(screen.getAllByText('0').length).toBeGreaterThanOrEqual(3);
  });
});
