import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Claim, EvidenceRecordView } from '../../lib/api/types';

const { listAllEvidence } = vi.hoisted(() => ({ listAllEvidence: vi.fn() }));
vi.mock('../../lib/api/endpoints', () => ({ listAllEvidence }));

import { AnswerSources } from './AnswerSources';

function claim(overrides: Partial<Claim>): Claim {
  return {
    claim_id: 'clm_1',
    kind: 'scientific',
    text: 'hERG blockade is associated with QT prolongation.',
    transform: 'identity',
    citation_ids: [],
    ...overrides,
  };
}

function record(id: string, title: string): EvidenceRecordView {
  return {
    evidence_id: id,
    title,
    authors: ['A. Author'],
    published_at: '2021-04-01',
    source_type: 'literature',
    source_quality_tier: 'authoritative_secondary',
    identifier: { doi: '10.1/x' },
    canonical_url: 'https://europepmc.org/article/MED/1',
    abstract_or_excerpt: null,
    normalized_facts: {},
    status: 'accepted',
    rejection_reason: null,
    provider: 'europepmc',
    retrieved_at: '2026-09-06T00:00:00Z',
    content_sha256: 'x',
    untrusted_external_content: true,
  };
}

function renderSources(claims: Claim[]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <AnswerSources claims={claims} sessionId="ses_1" />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  listAllEvidence.mockReset();
});

describe('AnswerSources', () => {
  it('shows a source for a claim that has no rendered value', async () => {
    // The gap this closes. `scientific` claims are not field-backed, so they
    // have no rendered_value, so linkifyClaims never anchors them and they
    // never became chips — and their citation appeared nowhere at all. The
    // one kind of claim a reader most needs a source for showed none.
    listAllEvidence.mockResolvedValue([record('evd_1', 'hERG and QT prolongation')]);
    renderSources([claim({ citation_ids: ['evd_1'], rendered_value: undefined })]);

    expect(await screen.findByText('hERG and QT prolongation')).toBeTruthy();
    expect(screen.getByText(/Nguồn được trích dẫn \(1\)/)).toBeTruthy();
  });

  it('renders nothing at all when no claim cites anything', () => {
    const { container } = renderSources([claim({ citation_ids: [] })]);
    expect(container.textContent).toBe('');
    expect(listAllEvidence).not.toHaveBeenCalled();
  });

  it('lists one entry per source, in reading order, without repeats', async () => {
    listAllEvidence.mockResolvedValue([record('evd_1', 'First'), record('evd_2', 'Second')]);
    renderSources([
      claim({ claim_id: 'clm_1', citation_ids: ['evd_2', 'evd_1'] }),
      claim({ claim_id: 'clm_2', citation_ids: ['evd_1'] }),
    ]);

    await screen.findByText('Second');
    const items = screen.getAllByRole('listitem');
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toContain('Second');
    expect(items[1].textContent).toContain('First');
  });

  it('shows the id when the record cannot be read, rather than dropping it', async () => {
    // A citation that disappears quietly is worse than one that admits it
    // could not be loaded: the reader would never know a claim was sourced.
    listAllEvidence.mockResolvedValue([]);
    renderSources([claim({ citation_ids: ['evd_missing'] })]);

    await waitFor(() => {
      expect(screen.getByText(/evidence evd_missing/)).toBeTruthy();
      expect(screen.getByText(/không đọc được bản ghi/)).toBeTruthy();
    });
  });

  it('shows the record title as the provider gave it, not one composed here', async () => {
    listAllEvidence.mockResolvedValue([record('evd_1', 'An awkward <title> with punctuation')]);
    renderSources([claim({ citation_ids: ['evd_1'] })]);
    expect(await screen.findByText('An awkward <title> with punctuation')).toBeTruthy();
  });

  it('links each source to its evidence record', async () => {
    listAllEvidence.mockResolvedValue([record('evd_1', 'First')]);
    renderSources([claim({ citation_ids: ['evd_1'] })]);
    const link = await screen.findByRole('link', { name: 'First' });
    expect(link.getAttribute('href')).toBe('/s/ses_1/evidence/evd_1');
  });

  it('has a heading its list is labelled by', async () => {
    listAllEvidence.mockResolvedValue([record('evd_1', 'First')]);
    renderSources([claim({ citation_ids: ['evd_1'] })]);
    await screen.findByText('First');
    expect(screen.getByRole('region', { name: /Nguồn được trích dẫn/ })).toBeTruthy();
  });
});
