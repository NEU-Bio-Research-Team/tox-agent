import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ScreeningSummary } from '../../../shared/api/endpoints';

const { listScreenings } = vi.hoisted(() => ({ listScreenings: vi.fn() }));
vi.mock('../../../shared/api/endpoints', () => ({ listScreenings }));

import { SubjectSelector, subjectOptions } from './SubjectSelector';

function screening(overrides: Partial<ScreeningSummary> = {}): ScreeningSummary {
  return {
    screening_id: 'scr_1',
    status: 'completed_with_toxicity_not_run',
    canonical_smiles: 'CCO',
    target: { target_id: 'EGFR', uniprot_id: 'P00533' },
    assay_context: 'IC50|wild_type',
    toxicity_run: false,
    has_activity_estimate: true,
    has_prioritisation_decision: true,
    has_toxicity_snapshot: false,
    intended_target_role: 'intended_target',
    ligand_names: [],
    model_admitted: false,
    record_sha256: 'abc',
    source: 'replay',
    attached_at: '2026-10-21T00:00:00Z',
    ...overrides,
  } as ScreeningSummary;
}

function renderSelector(props: Record<string, unknown> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <SubjectSelector sessionId="ses_1" onSelect={() => {}} {...props} />
    </QueryClientProvider>,
  );
}

describe('SubjectSelector', () => {
  afterEach(() => {
    cleanup();
    listScreenings.mockReset();
  });

  it('names the ligand, the target and the assay context of every subject', () => {
    const options = subjectOptions(
      { analysis_id: 'ana_1', canonical_smiles: 'CCO' },
      [screening(), screening({ screening_id: 'scr_2', target: { target_id: 'KCNH2' },
                                toxicity_run: true })],
    );
    expect(options.map((option) => option.key)).toEqual([
      'analysis:ana_1', 'screening:scr_1', 'screening:scr_2',
    ]);
    expect(options[1].label).toBe('EGFR');
    expect(options[1].detail).toContain('IC50|wild_type');
    expect(options[1].detail).toContain('độc tính chưa chạy');
    expect(options[2].detail).toContain('độc tính đã chạy');
    // One ligand on two targets is two subjects, never one.
    expect(new Set(options.map((option) => option.key)).size).toBe(3);
  });

  it('shows a variant as part of the target, so a variant is not merged in', () => {
    const [option] = subjectOptions(null, [
      screening({ target: { target_id: 'EGFR', variant: 'T790M' } }),
    ]);
    expect(option.label).toBe('EGFR (T790M)');
  });

  it('asks the researcher to choose and never selects for them', async () => {
    listScreenings.mockResolvedValue({
      screenings: [screening(), screening({ screening_id: 'scr_2',
                                            target: { target_id: 'KCNH2' } })],
      mode: 'replay',
    });
    const onSelect = vi.fn();
    renderSelector({ onSelect });

    expect(await screen.findByText(/Hãy chọn một trước khi hỏi/)).toBeInTheDocument();
    expect(onSelect).not.toHaveBeenCalled();
    await userEvent.click(screen.getByText('KCNH2'));
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ key: 'screening:scr_2' }),
    );
  });

  it('says a queued run keeps the subject it was sent with', async () => {
    listScreenings.mockResolvedValue({ screenings: [screening()], mode: 'replay' });
    renderSelector({ runPending: true, selected: 'screening:scr_1' });
    expect(await screen.findByText(/Run đang chạy vẫn trả lời về chủ thể đã chọn lúc gửi/))
      .toBeInTheDocument();
  });

  it('renders nothing on a deployment with no subjects', async () => {
    listScreenings.mockResolvedValue({ screenings: [], mode: 'replay' });
    const { container } = renderSelector();
    expect(container).toBeEmptyDOMElement();
  });
});
