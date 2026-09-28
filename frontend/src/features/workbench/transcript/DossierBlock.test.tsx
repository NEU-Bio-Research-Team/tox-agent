import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

const { getRunDossier } = vi.hoisted(() => ({ getRunDossier: vi.fn() }));
vi.mock('../../../shared/api/endpoints', () => ({ getRunDossier }));

import { DossierBlock } from './DossierBlock';
import type { DecisionDossier } from '../../../shared/api/types';

const DOSSIER: DecisionDossier = {
  schema_version: 'decision-dossier-v1',
  case_id: 'scase_1',
  case_revision: 7,
  run_id: 'run_1',
  answer_id: 'ans_1',
  question: 'Should hERG stop us?',
  data_scope: { external_search: false, reason: 'unpublished', run_id: null },
  hypotheses: [{
    id: 'h1', statement: 'A blocks hERG at exposure', kind: 'mechanism', refutation_condition: 'x',
    status: 'weakened', status_reason: '', actor: 'model', run_id: 'run_1',
    evidence_for: [], evidence_against: [],
  }],
  open_uncertainties: [{ id: 'u1', kind: 'missing_exposure', description: 'no Cmax', severity: 'blocking',
    hypothesis_ids: ['h1'], status: 'open', resolution: '', resolving_refs: [], actor: 'model', run_id: 'run_1' }],
  next_tests: [],
  conclusion: { can_say: [{ text: 'In-house block is weak', evidence_ids: ['e2'], sources: ['context:c1'] }],
    cannot_say: ['whether A blocks at exposure'], what_would_change: ['a free Cmax'], approver: '', run_id: 'run_1' },
  coverage: { hypotheses: 1, with_any_source: 1, with_independent_direct_evidence: 1,
    with_counterevidence_considered: 0, open_uncertainties: 1, blocking_uncertainties: 1 },
  stop_reason: 'sufficient',
};

function renderBlock() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}><DossierBlock sessionId="ses_1" runId="run_1" /></QueryClientProvider>,
  );
}

describe('DossierBlock', () => {
  afterEach(() => getRunDossier.mockReset());

  it('shows the run dossier under the answer', async () => {
    getRunDossier.mockResolvedValue(DOSSIER);
    renderBlock();
    expect(await screen.findByRole('region', { name: 'Hồ sơ quyết định' })).toBeInTheDocument();
    expect(getRunDossier).toHaveBeenCalledWith('ses_1', 'run_1');
    expect(screen.getByText(/bị yếu đi; ủng hộ 0, phản bác 0/)).toBeInTheDocument();
    expect(screen.getByText('whether A blocks at exposure')).toBeInTheDocument();
    expect(screen.getByText('a free Cmax')).toBeInTheDocument();
    expect(screen.getByText(/1 chặn kết luận/)).toBeInTheDocument();
    expect(screen.getByText(/chỉ dữ liệu nội bộ/)).toBeInTheDocument();
  });

  it('renders nothing for a dossier with nothing to show', async () => {
    getRunDossier.mockResolvedValue({ ...DOSSIER, hypotheses: [],
      conclusion: { ...DOSSIER.conclusion, can_say: [], cannot_say: [] } });
    const { container } = renderBlock();
    await vi.waitFor(() => expect(getRunDossier).toHaveBeenCalled());
    expect(container.querySelector('section')).toBeNull();
  });
});
