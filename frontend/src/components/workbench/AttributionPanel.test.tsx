import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';

const { listAttributions } = vi.hoisted(() => ({ listAttributions: vi.fn() }));
vi.mock('../../lib/api/endpoints', () => ({ listAttributions }));

import { AttributionPanel } from './AttributionPanel';

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <AttributionPanel sessionId="ses_1" analysisId="ana_1" />
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

describe('AttributionPanel', () => {
  afterEach(() => listAttributions.mockReset());

  it('keeps a partial endpoint-specific attribution neutral and exposes its observation', async () => {
    listAttributions.mockResolvedValue({
      attributions: [{
        observation_id: 'obs_1',
        run_id: 'run_1',
        created_at: '2026-09-06T00:00:00Z',
        content_sha256: 'abc',
        required_limitations: ['attribution_not_causality'],
        analysis_id: 'ana_1',
        endpoint: 'tox21',
        task: 'SR-p53',
        status: 'partial',
        method: 'integrated_gradients',
        model_id: 'tox21-v1',
        top_tokens: [{ token: 'Cl', score: -0.1234 }],
      }],
    });
    renderPanel();

    expect(await screen.findByText('Attribution theo endpoint')).toBeInTheDocument();
    expect(screen.getByText(/không chứng minh quan hệ nhân quả/i)).toBeInTheDocument();
    expect(screen.getByText(/partial.*không coi.*hoàn chỉnh/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Mở observation attribution/i })).toHaveAttribute(
      'href', '/s/ses_1/observations/obs_1',
    );
  });

  it('shows a deploy-compatible error instead of silently removing the panel', async () => {
    listAttributions.mockRejectedValue(new Error('HTTP 404'));
    renderPanel();

    expect(await screen.findByText(/Không tải được attribution/i)).toBeInTheDocument();
  });
});
