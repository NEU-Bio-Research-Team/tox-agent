import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

const { listScientificCases, getScientificCase, addScientificCaseContext } = vi.hoisted(() => ({
  listScientificCases: vi.fn(),
  getScientificCase: vi.fn(),
  addScientificCaseContext: vi.fn(),
}));
vi.mock('../../lib/api/endpoints', () => ({ listScientificCases, getScientificCase, addScientificCaseContext }));

import { InvestigationBoard } from './InvestigationBoard';
import type { ScientificCase } from '../../lib/api/types';

const CASE: ScientificCase = {
  schema_version: 'scientific-case-v1',
  case_id: 'scase_1',
  session_id: 'ses_1',
  subject_key: 'analysis:ana_1',
  question: 'Should hERG stop us developing compound A?',
  decision_context: '',
  subject_refs: ['analysis:ana_1'],
  status: 'open',
  context: [{ id: 'c1', key: 'patch_clamp_ic50', value: '30 µM', note: '', actor: 'user', run_id: null }],
  hypotheses: [
    { id: 'h1', statement: 'A blocks hERG at relevant exposure', kind: 'mechanism',
      refutation_condition: 'IC50 far above free Cmax', status: 'weakened',
      status_reason: 'in-house data shows weak block', actor: 'model', run_id: 'run_1' },
  ],
  evidence: [
    { id: 'e1', claim: 'The model scores A as a likely blocker', source_class: 'predictor_fact',
      source_ref: 'observation:obs_1', stance: 'supports', directness: 'direct', hypothesis_ids: ['h1'],
      locator: null, scope: { endpoint: 'herg' }, actor: 'model', run_id: 'run_1' },
    { id: 'e2', claim: 'In-house IC50 is 30 µM', source_class: 'user_supplied', source_ref: 'context:c1',
      stance: 'contradicts', directness: 'direct', hypothesis_ids: ['h1'], locator: null, scope: {},
      actor: 'model', run_id: 'run_2' },
  ],
  uncertainties: [
    { id: 'u1', kind: 'missing_exposure', description: 'No free Cmax known', severity: 'blocking',
      hypothesis_ids: ['h1'], status: 'open', resolution: '', resolving_refs: [], actor: 'model', run_id: 'run_1' },
  ],
  actions: [],
  next_tests: [
    { id: 't1', test: 'Measure free Cmax at the intended dose', rationale: 'Sets the margin',
      discriminates: ['h1'], expected_readouts: ['a margin above 30-fold weakens h1'], actor: 'model', run_id: 'run_2' },
  ],
  conclusion: { can_say: [{ text: 'In-house block is weak', evidence_ids: ['e2'] }],
    cannot_say: ['whether A is safe'], what_would_change: ['a free Cmax near 30 µM'], approver: '', run_id: 'run_2' },
  runs: [{ run_id: 'run_1', goal: 'q', stop_reason: 'sufficient', answer_id: 'ans_1' }],
  coverage: { hypotheses: 1, with_any_source: 1, with_independent_direct_evidence: 1,
    with_counterevidence_considered: 1, open_uncertainties: 1, blocking_uncertainties: 1 },
  revision: 9,
  created_at: '2026-09-25T00:00:00Z',
  updated_at: '2026-09-25T00:00:00Z',
};

function renderBoard() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <InvestigationBoard sessionId="ses_1" />
    </QueryClientProvider>,
  );
}

describe('InvestigationBoard', () => {
  afterEach(() => {
    listScientificCases.mockReset();
    getScientificCase.mockReset();
    addScientificCaseContext.mockReset();
  });

  it('shows the case: question, evidence for and against, unknowns and what would change it', async () => {
    listScientificCases.mockResolvedValue({ cases: [{ ...CASE, hypotheses: 1, evidence: 2, open_uncertainties: 1, runs: 1 }] });
    getScientificCase.mockResolvedValue(CASE);
    renderBoard();

    expect(await screen.findByText('Should hERG stop us developing compound A?')).toBeInTheDocument();
    expect(screen.getByText('bị yếu đi')).toBeInTheDocument();
    expect(screen.getByText('Ủng hộ (1)')).toBeInTheDocument();
    expect(screen.getByText('Phản bác (1)')).toBeInTheDocument();
    // A model score is labelled as a model signal, never as independent evidence.
    expect(screen.getByText(/Tín hiệu về mô hình, không phải bằng chứng độc lập/)).toBeInTheDocument();
    expect(screen.getByText(/missing_exposure · chặn kết luận/)).toBeInTheDocument();
    expect(screen.getByText('a free Cmax near 30 µM')).toBeInTheDocument();
    expect(screen.getByText('Measure free Cmax at the intended dose')).toBeInTheDocument();
    expect(screen.getByText(/không gộp thành một điểm/)).toBeInTheDocument();
  });

  it('files researcher context against the open case', async () => {
    listScientificCases.mockResolvedValue({ cases: [{ ...CASE, hypotheses: 1, evidence: 2, open_uncertainties: 1, runs: 1 }] });
    getScientificCase.mockResolvedValue(CASE);
    addScientificCaseContext.mockResolvedValue(CASE);
    renderBoard();

    await screen.findByText('Should hERG stop us developing compound A?');
    fireEvent.change(screen.getByLabelText('Loại dữ liệu'), { target: { value: 'free_cmax' } });
    fireEvent.change(screen.getByLabelText('Giá trị'), { target: { value: '0.05 µM' } });
    fireEvent.click(screen.getByRole('button', { name: 'Thêm vào hồ sơ' }));
    await waitFor(() => expect(addScientificCaseContext).toHaveBeenCalledWith(
      'ses_1', 'scase_1', { key: 'free_cmax', value: '0.05 µM' },
    ));
  });

  it('says plainly when the session has no case', async () => {
    listScientificCases.mockResolvedValue({ cases: [] });
    renderBoard();
    expect(await screen.findByText(/chưa có hồ sơ điều tra/i)).toBeInTheDocument();
    expect(getScientificCase).not.toHaveBeenCalled();
  });
});
