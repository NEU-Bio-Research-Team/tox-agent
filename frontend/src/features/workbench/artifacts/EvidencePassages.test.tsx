import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EvidencePassagesView } from '../../../shared/api/endpoints';

const { getEvidencePassages } = vi.hoisted(() => ({ getEvidencePassages: vi.fn() }));
vi.mock('../../../shared/api/endpoints', () => ({ getEvidencePassages }));

import { EvidencePassages } from './EvidencePassages';

const READ = {
  state: 'full_text_retrieved' as const, reason: null, full_text_id: 'PMC1111111', passage_count: 5,
  omitted: { table_row: 2 }, truncated: false,
  provenance: { replay: true, served_by: 'snapshot', captured_from: 'europepmc', captured_at: '2026-10-21T00:00:00+00:00',
    parser_version: 'jats-passages-v1' },
};

function view(overrides: Partial<EvidencePassagesView>): EvidencePassagesView {
  return {
    schema_version: 'evidence-passages-v1', evidence_id: 'evd_1', access: READ, measurement: null,
    focus: null, passages: [], untrusted_external_content: true, ...overrides,
  };
}

const TABLE = {
  locator: 'table-wrap:1', label: 'Table 1', caption: 'Table 1 hERG potency by patch clamp',
  header_locator: 'table-wrap:1/row:1', header: ['Compound', 'hERG IC50 (uM)'],
  row_locator: 'table-wrap:1/row:3', row_kind: 'table_row', cells: ['compound 7', '> 30'], cited_cell: 2,
  columns: [{ header: 'Compound', value: 'compound 7', cited: false }, { header: 'hERG IC50 (uM)', value: '> 30', cited: true }],
};

function renderPassages(locator?: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <EvidencePassages sessionId="ses_1" evidenceId="evd_1" locator={locator} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('EvidencePassages', () => {
  afterEach(() => {
    cleanup();
    getEvidencePassages.mockReset();
  });

  it('shows a cited cell under its header, with the comparator as written', async () => {
    getEvidencePassages.mockResolvedValue(view({
      focus: { locator: 'table-wrap:1/row:3/cell:2', found: true,
        passage: { kind: 'table_cell', text: '> 30', section: 'Table 1' }, table: TABLE },
    }));
    renderPassages('table-wrap:1/row:3/cell:2');

    const table = await screen.findByRole('table', { name: 'Table 1' });
    expect(within(table).getByRole('columnheader', { name: 'hERG IC50 (uM)' })).toBeInTheDocument();
    const cited = within(table).getByText('> 30').closest('td');
    expect(cited).toHaveAttribute('aria-current', 'true');
    expect(screen.getByText(/Ô được trích: cột 2 — hERG IC50 \(uM\): > 30/)).toBeInTheDocument();
    expect(screen.getByText(/Phát lại từ bản chụp đóng băng của europepmc/)).toBeInTheDocument();
    expect(screen.getByText(/Bị giới hạn bỏ qua: 2 table_row/)).toBeInTheDocument();
    expect(screen.getByText(/không phải hướng dẫn cho hệ thống/)).toBeInTheDocument();
    expect(getEvidencePassages).toHaveBeenCalledWith('ses_1', 'evd_1', 'table-wrap:1/row:3/cell:2');
  });

  it('does not re-align a row that does not line up with its header', async () => {
    getEvidencePassages.mockResolvedValue(view({
      focus: { locator: 'table-wrap:1/row:4', found: true, passage: { kind: 'table_row', text: 'x' },
        table: { ...TABLE, cells: ['c9', '0.4', 'n=3'], cited_cell: null, columns: [] } },
    }));
    renderPassages('table-wrap:1/row:4');
    expect(await screen.findByText(/không cùng số cột với header/)).toBeInTheDocument();
  });

  it('tells a provider refusal apart from a text never read', async () => {
    getEvidencePassages.mockResolvedValueOnce(view({
      access: { ...READ, state: 'access_failed', reason: 'the provider holds no full text for PMC1', passage_count: 0, provenance: {} },
      focus: { locator: 'table-wrap:1/row:3', found: false, passage: null, table: null },
    }));
    renderPassages('table-wrap:1/row:3');
    expect(await screen.findByText('Không đọc được toàn văn')).toBeInTheDocument();
    expect(screen.getByText(/the provider holds no full text for PMC1/)).toBeInTheDocument();
    expect(screen.getByText(/vì toàn văn chưa được đọc/)).toBeInTheDocument();
    cleanup();

    getEvidencePassages.mockResolvedValueOnce(view({
      access: { ...READ, state: 'not_read', full_text_id: null, passage_count: 0, provenance: {} },
    }));
    renderPassages();
    expect(await screen.findByText(/Chưa đọc toàn văn/)).toBeInTheDocument();
    expect(screen.queryByText('Không đọc được toàn văn')).toBeNull();
  });

  it('says a locator the text does not hold is not there, and offers the outline', async () => {
    getEvidencePassages.mockResolvedValue(view({
      focus: { locator: 'table-wrap:1/row:9', found: false, passage: null, table: null },
    }));
    renderPassages('table-wrap:1/row:9');
    expect(await screen.findByText(/trong văn bản đã đọc/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /xem mọi đoạn đã đọc/ })).toHaveAttribute('href', '/s/ses_1/evidence/evd_1');
  });

  it('lists every passage read, each a deep link to its locator', async () => {
    getEvidencePassages.mockResolvedValue(view({
      passages: [
        { locator: 'sec:1/p:1', section: 'Results', kind: 'paragraph', preview: 'Potencies are in Table 1.', truncated: false },
        { locator: 'table-wrap:1/row:3', section: 'Table 1', kind: 'table_row', preview: 'compound 7 | 0.38', truncated: false },
      ],
      measurement: { measurement: { state: 'measured', relation: '<', value: 10, unit: 'uM' } },
    }));
    renderPassages();
    expect(await screen.findByText(/< 10 uM/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Xem các đoạn đã đọc (2)' }));
    const row = screen.getByRole('link', { name: /table-wrap:1\/row:3/ });
    expect(row).toHaveAttribute('href', '/s/ses_1/evidence/evd_1?locator=table-wrap%3A1%2Frow%3A3');
  });
});
