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

const { getExpertModeEnabled } = vi.hoisted(() => ({ getExpertModeEnabled: vi.fn() }));
vi.mock('../lib/preferences', () => ({ getExpertModeEnabled }));

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
    quickPredictCapabilities.mockResolvedValue({
      served_endpoints: ['herg', 'tox21'],
      models: [],
      predictor_id: 'toxpred-local',
      ocr_available: false,
    });
    quickPredict.mockResolvedValue({ ...FIXTURE, analysis_id: null, persisted: false });

    renderPage();
    fireEvent.change(screen.getByLabelText('SMILES'), { target: { value: 'CCO' } });
    fireEvent.click(screen.getByRole('button', { name: 'Phân tích' }));

    await waitFor(() => expect(quickPredict).toHaveBeenCalledOnce());
    expect(quickPredict.mock.calls[0][0]).toMatchObject({ smiles: 'CCO', endpoints: ['herg', 'tox21'] });
    expect(await screen.findByText('hERG')).toBeInTheDocument();
  });

  it('disables the clintox checkbox when the predictor does not serve it', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue({
      served_endpoints: ['herg', 'tox21'],
      models: [],
      predictor_id: 'toxpred-local',
      ocr_available: false,
    });

    renderPage();
    await waitFor(() => expect(quickPredictCapabilities).toHaveBeenCalled());
    const clintox = screen.getByRole('checkbox', { name: /clintox/i });
    await waitFor(() => expect(clintox).toBeDisabled());
  });

  it('hides the threshold override field for a non-expert', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue({
      served_endpoints: ['herg', 'tox21'],
      models: [],
      predictor_id: 'toxpred-local',
      ocr_available: false,
    });

    renderPage();
    expect(screen.queryByLabelText(/threshold override/i)).not.toBeInTheDocument();
  });

  it('runs a batch and renders one panel per result plus the error list', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue({
      served_endpoints: ['herg', 'tox21'],
      models: [],
      predictor_id: 'toxpred-local',
      ocr_available: false,
    });
    quickPredictBatch.mockResolvedValue({
      count: 3,
      results: [
        { ...FIXTURE, analysis_id: null, persisted: false, canonical_smiles: 'CCO' },
        { ...FIXTURE, analysis_id: null, persisted: false, canonical_smiles: 'CCN' },
      ],
      errors: [{ index: 1, input_smiles: 'nope', error: 'invalid_smiles', detail: '' }],
    });

    renderPage();
    fireEvent.click(screen.getByRole('checkbox', { name: /Nhiều phân tử/ }));
    fireEvent.change(screen.getByLabelText('Danh sách SMILES'), {
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
    quickPredictCapabilities.mockResolvedValue({
      served_endpoints: ['herg', 'tox21'],
      models: [],
      predictor_id: 'toxpred-local',
      ocr_available: false,
    });

    renderPage();
    expect(screen.getByLabelText(/threshold override/i)).toBeInTheDocument();
  });
});
