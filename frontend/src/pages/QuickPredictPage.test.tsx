import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(cleanup);
import type { AnalysisProjection } from '../lib/api/types';

const {
  quickPredict,
  quickPredictBatch,
  quickPredictCapabilities,
  recognizeStructure,
  explainPrediction,
} = vi.hoisted(() => ({
  quickPredict: vi.fn(),
  quickPredictBatch: vi.fn(),
  quickPredictCapabilities: vi.fn(),
  recognizeStructure: vi.fn(),
  explainPrediction: vi.fn(),
}));
vi.mock('../lib/api/endpoints', () => ({
  quickPredict,
  quickPredictBatch,
  quickPredictCapabilities,
  recognizeStructure,
  explainPrediction,
}));

const { getEndpointSelection, getExpertModeEnabled, setEndpointSelection } = vi.hoisted(() => ({ getEndpointSelection: vi.fn(), getExpertModeEnabled: vi.fn(), setEndpointSelection: vi.fn() }));
vi.mock('../lib/preferences', () => ({ getEndpointSelection, getExpertModeEnabled, setEndpointSelection }));

import { QuickPredictPage } from './QuickPredictPage';

const FIXTURE: AnalysisProjection = {
  analysis_id: 'ana_x',
  input_smiles: 'CCO',
  canonical_smiles: 'CCO',
  requested_endpoints: ['herg', 'tox21'],
  served_endpoints: ['herg', 'tox21'],
  unavailable_endpoints: [],
  sections: {
    herg: {
      measurement: 'hERG channel blockade liability',
      probability_blocker: 0.73,
      label: 'blocker',
      threshold: 0.5,
      threshold_source: 'model_default',
      model_id: 'm',
    },
  },
  applicability: { status: 'ok', method: 'element_rules_v1', reasons: [] },
  provenance: { content_sha256: 'abc' },
  policy_snapshot: {},
  required_limitations: ['uncalibrated_probability'],
  created_at: '2026-09-06T00:00:00Z',
};

const CAPS = {
  capability_version: 'predict-capabilities-v2' as const,
  default_endpoints: ['herg', 'tox21'] as const,
  served_endpoints: ['herg', 'tox21'] as const,
  endpoints: [
    { id: 'herg' as const, display_name: 'hERG blockade', enabled: true, model_id: 'm', supports_explanation: true, explanation_target_required: false, tasks: [], blocked_reason: null },
    { id: 'tox21' as const, display_name: 'Tox21 assays', enabled: true, model_id: 'm', supports_explanation: true, explanation_target_required: true, tasks: ['SR-p53'], blocked_reason: null },
    { id: 'clintox' as const, display_name: 'Clinical toxicity', enabled: false, model_id: null, supports_explanation: false, explanation_target_required: false, tasks: [], blocked_reason: 'Unavailable' },
  ],
  models: [], predictor_id: 'toxpred-local', ocr_available: false,
};

function renderPage() {
  return render(
    <MemoryRouter>
      <QuickPredictPage />
    </MemoryRouter>,
  );
}

describe('QuickPredictPage', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders the analysis panel from the quickPredict result', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue(CAPS);
    quickPredict.mockResolvedValue({ ...FIXTURE, analysis_id: null, persisted: false });

    renderPage();
    fireEvent.change(screen.getByLabelText('SMILES'), { target: { value: 'CCO' } });
    fireEvent.click(screen.getByRole('button', { name: 'Phân tích' }));

    await waitFor(() => expect(quickPredict).toHaveBeenCalledOnce());
    expect(quickPredict.mock.calls[0][0]).toMatchObject({ smiles: 'CCO', endpoints: ['herg', 'tox21'] });
    expect(await screen.findByText('hERG')).toBeInTheDocument();
  });

  it('disables an unavailable endpoint from the capability inventory', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue(CAPS);

    renderPage();
    await waitFor(() => expect(quickPredictCapabilities).toHaveBeenCalled());
    const clintox = screen.getByRole('button', { name: /clinical toxicity/i });
    await waitFor(() => expect(clintox).toBeDisabled());
  });

  it('hides the threshold override field for a non-expert', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue(CAPS);

    renderPage();
    expect(screen.queryByLabelText(/threshold override/i)).not.toBeInTheDocument();
  });

  it('runs a batch and renders one panel per result plus the error list', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue(CAPS);
    quickPredictBatch.mockResolvedValue({
      count: 3,
      results: [
        { ...FIXTURE, analysis_id: null, persisted: false, canonical_smiles: 'CCO' },
        { ...FIXTURE, analysis_id: null, persisted: false, canonical_smiles: 'CCN' },
      ],
      errors: [{ index: 1, input_smiles: 'nope', error: 'invalid_smiles', detail: '' }],
    });

    renderPage();
    const batchToggle = screen.getByRole('button', { name: 'Hàng loạt' });
    fireEvent.click(batchToggle);
    await waitFor(() => expect(batchToggle).toHaveAttribute('aria-pressed', 'true'));
    fireEvent.change(await screen.findByLabelText('Danh sách SMILES'), {
      target: { value: 'CCO\nnope\nCCN' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Phân tích' }));

    await waitFor(() => expect(quickPredictBatch).toHaveBeenCalledOnce());
    expect(quickPredictBatch.mock.calls[0][0].smiles).toEqual(['CCO', 'nope', 'CCN']);
    expect(await screen.findByText(/1 phân tử lỗi/)).toBeInTheDocument();
    expect(screen.getAllByText('hERG')).toHaveLength(2);
  });

  it('shows the threshold override field for an expert', async () => {
    getExpertModeEnabled.mockReturnValue(true);
    quickPredictCapabilities.mockResolvedValue(CAPS);

    renderPage();
    expect(screen.getByLabelText(/threshold override/i)).toBeInTheDocument();
  });
});
