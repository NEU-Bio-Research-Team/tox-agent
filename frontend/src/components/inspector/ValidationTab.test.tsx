import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

const { listAllEventsForRun } = vi.hoisted(() => ({ listAllEventsForRun: vi.fn() }));
vi.mock('../../lib/api/endpoints', () => ({ listAllEventsForRun }));

import { ValidationTab } from './ValidationTab';

function renderTab() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ValidationTab sessionId="ses_1" runId="run_1" />
    </QueryClientProvider>,
  );
}

describe('ValidationTab', () => {
  afterEach(() => listAllEventsForRun.mockReset());

  it('keeps rejection violations behind an explicit details control', async () => {
    listAllEventsForRun.mockResolvedValue([
      {
        event_id: 'evt_1', session_id: 'ses_1', sequence: 1, type: 'answer.rejected',
        entity_type: 'answer', entity_id: 'ans_1', entity_version: 1, run_id: 'run_1',
        occurred_at: '2026-09-06T00:00:00Z',
        payload: {
          candidate_generation: 1,
          violations: [{ code: 'numeric_mismatch', message: 'Giá trị không khớp nguồn.' }],
        },
      },
    ]);
    renderTab();

    const details = await screen.findByText(/Chi tiết 1 vi phạm/i);
    expect(details.closest('details')).not.toHaveAttribute('open');
    fireEvent.click(details);
    expect(details.closest('details')).toHaveAttribute('open');
    expect(screen.getByText('numeric_mismatch')).toBeInTheDocument();
  });
});
