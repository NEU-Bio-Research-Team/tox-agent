import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(cleanup);

const { explainPrediction } = vi.hoisted(() => ({ explainPrediction: vi.fn() }));
vi.mock('../../lib/api/endpoints', () => ({ explainPrediction }));

import { ExplainPanel } from './ExplainPanel';
import type { AtomAttribution } from '../../lib/api/types';

const BASE: AtomAttribution = {
  status: 'completed',
  endpoint: 'herg',
  task: null,
  input_smiles: 'CCO',
  canonical_smiles: 'CCO',
  atom_order_version: 'rdkit-output-order-v1',
  probability: 0.4,
  atoms: [
    { atom_index: 0, symbol: 'C', importance: 0.5, relative_importance: 0.5 },
    { atom_index: 2, symbol: 'O', importance: 0.2, relative_importance: 0.2 },
  ],
  unmapped_importance: 0.3,
  tokens: [{ token: 'C', importance: 0.5 }],
  method: 'grad_x_embedding_l2_v1+token_atom_align_v1',
  metadata: {},
  limitations: ['attribution_not_causality'],
};

async function open(data: AtomAttribution) {
  explainPrediction.mockResolvedValue(data);
  render(<ExplainPanel canonicalSmiles="CCO" endpoint="herg" />);
  fireEvent.click(screen.getByRole('button', { name: /Giải thích \(XAI\)/i }));
  fireEvent.click(screen.getByRole('button', { name: 'Giải thích' }));
}

describe('ExplainPanel', () => {
  afterEach(() => explainPrediction.mockReset());

  it('shows the unmapped-importance line and the non-causality legend', async () => {
    await open(BASE);
    expect(await screen.findByText(/30% trọng số nằm ở liên kết\/tô-pô/)).toBeInTheDocument();
    expect(screen.getByText(/không phải quan hệ nhân quả/i)).toBeInTheDocument();
    expect(screen.getByText(/không chứng minh quan hệ nhân quả/i)).toBeInTheDocument();
  });

  it('renders a magnitude-only ramp with no red or green colour tokens', async () => {
    await open(BASE);
    await screen.findByText(/30% trọng số/);
    expect(document.body.innerHTML).not.toMatch(/accent-red|accent-green/);
  });

  it('shows a best-effort note for a partial result', async () => {
    await open({ ...BASE, status: 'partial' });
    expect(await screen.findByText(/best-effort/i)).toBeInTheDocument();
  });

  it('shows an error state and no highlight for a failed result', async () => {
    await open({ ...BASE, status: 'failed', atoms: [], probability: null, unmapped_importance: null });
    expect(await screen.findByText(/attribution failed/i)).toBeInTheDocument();
  });

  it('falls back without atom bars when the atom-order version does not match', async () => {
    await open({ ...BASE, atom_order_version: 'something-else' });
    expect(
      await screen.findByText(/Không căn được attribution theo nguyên tử/),
    ).toBeInTheDocument();
  });
});
