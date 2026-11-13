import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(cleanup);
import type { AnalysisProjection } from '../../shared/api/types';

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
vi.mock('../../shared/api/endpoints', () => ({
  quickPredict,
  quickPredictBatch,
  quickPredictCapabilities,
  recognizeStructure,
  explainPrediction,
}));

const { getEndpointSelection, getExpertModeEnabled, setEndpointSelection } = vi.hoisted(() => ({ getEndpointSelection: vi.fn(), getExpertModeEnabled: vi.fn(), setEndpointSelection: vi.fn() }));
vi.mock('../../shared/lib/preferences', () => ({ getEndpointSelection, getExpertModeEnabled, setEndpointSelection }));

import { QuickPredictContent } from './QuickPredictPage';

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
      <QuickPredictContent />
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
    fireEvent.change(screen.getByLabelText('Chuỗi SMILES'), { target: { value: 'CCO' } });
    fireEvent.click(screen.getByRole('button', { name: 'Phân tích' }));

    await waitFor(() => expect(quickPredict).toHaveBeenCalledOnce());
    expect(quickPredict.mock.calls[0][0]).toMatchObject({ smiles: 'CCO', endpoints: ['herg', 'tox21'] });
    expect(await screen.findByRole('heading', { name: 'hERG' })).toBeInTheDocument();
  });

  it('disables an unavailable endpoint from the capability inventory', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue(CAPS);

    renderPage();
    await waitFor(() => expect(quickPredictCapabilities).toHaveBeenCalled());
    const clintox = screen.getByRole('checkbox', { name: /ClinTox/ });
    await waitFor(() => expect(clintox).toBeDisabled());
    expect(clintox).not.toBeChecked();
  });

  it('hides the threshold override field for a non-expert', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue(CAPS);

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Tuỳ chọn nâng cao/ }));
    expect(screen.queryByLabelText(/ghi đè ngưỡng/i)).not.toBeInTheDocument();
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
    const batchToggle = screen.getByRole('button', { name: 'Danh sách' });
    fireEvent.click(batchToggle);
    await waitFor(() => expect(batchToggle).toHaveAttribute('aria-pressed', 'true'));
    fireEvent.change(await screen.findByLabelText('Danh sách SMILES'), {
      target: { value: 'CCO\nnope\nCCN' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Phân tích' }));

    await waitFor(() => expect(quickPredictBatch).toHaveBeenCalledOnce());
    expect(quickPredictBatch.mock.calls[0][0].smiles).toEqual(['CCO', 'nope', 'CCN']);
    expect(await screen.findByText(/1 phân tử lỗi/)).toBeInTheDocument();
    expect(screen.getAllByRole('heading', { name: 'hERG' })).toHaveLength(2);
  });

  it('keeps Analyze disabled until there is something to send', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue(CAPS);

    renderPage();
    const analyze = screen.getByRole('button', { name: 'Phân tích' });
    expect(analyze).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: /Aspirin/ }));
    expect(screen.getByLabelText('Chuỗi SMILES')).toHaveValue('CC(=O)Oc1ccccc1C(=O)O');
    expect(analyze).toBeEnabled();
  });

  it('says so when image recognition is not enabled on the deployment', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    quickPredictCapabilities.mockResolvedValue(CAPS);

    renderPage();
    const tab = screen.getByRole('tab', { name: 'Ảnh' });
    fireEvent.mouseDown(tab);
    fireEvent.click(tab);
    expect(await screen.findByText(/chưa bật nhận diện cấu trúc từ ảnh/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Chọn ảnh/ })).toBeDisabled();
  });

  it('starts on the declared default model, not the first one listed, and names each release', async () => {
    getExpertModeEnabled.mockReturnValue(false);
    const model = (id: string, extra: Record<string, unknown>) => ({
      model_id: id, capabilities: ['herg', 'tox21'], loaded: true, required: false, detail: '', blocked_reason: null, ...extra,
    });
    // Listed by identifier, so the quick release comes first.
    const models = [
      model('herg-tox21-chemberta-quick-v1', { display_name: 'ChemBERTa quick', notes: 'Short training run.', default_for: [] }),
      model('herg-tox21-chemberta-v1', { display_name: 'ChemBERTa full', notes: 'The standard release.', default_for: ['herg', 'tox21'] }),
    ];
    quickPredictCapabilities.mockResolvedValue({
      ...CAPS,
      endpoints: CAPS.endpoints.map((endpoint) => (endpoint.enabled ? { ...endpoint, models } : endpoint)),
    });
    quickPredict.mockResolvedValue({ ...FIXTURE, analysis_id: null, persisted: false });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Tuỳ chọn nâng cao/ }));
    const select = (await screen.findByLabelText('Mô hình cho hERG')) as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe('herg-tox21-chemberta-v1'));
    expect(Array.from(select.options).map((option) => option.textContent)).toEqual([
      'ChemBERTa quick',
      'ChemBERTa full · mặc định',
    ]);
    expect(screen.getAllByText('The standard release.').length).toBeGreaterThan(0);

    fireEvent.change(select, { target: { value: 'herg-tox21-chemberta-quick-v1' } });
    expect(screen.getAllByText('Short training run.').length).toBeGreaterThan(0);

    fireEvent.change(screen.getByLabelText('Chuỗi SMILES'), { target: { value: 'CCO' } });
    fireEvent.click(screen.getByRole('button', { name: 'Phân tích' }));
    await waitFor(() => expect(quickPredict).toHaveBeenCalledOnce());
    expect(quickPredict.mock.calls[0][0].model_selection).toMatchObject({
      herg: 'herg-tox21-chemberta-quick-v1',
      tox21: 'herg-tox21-chemberta-v1',
    });
  });

  it('shows the threshold override field for an expert', async () => {
    getExpertModeEnabled.mockReturnValue(true);
    quickPredictCapabilities.mockResolvedValue(CAPS);

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Tuỳ chọn nâng cao/ }));
    expect(screen.getByLabelText(/ghi đè ngưỡng/i)).toBeInTheDocument();
  });
});
