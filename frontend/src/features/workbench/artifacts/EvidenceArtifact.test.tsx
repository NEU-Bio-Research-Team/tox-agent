import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EvidenceRecordView } from '../../../shared/api/types';

const { getEvidence, getEvidencePassages } = vi.hoisted(() => ({
  getEvidence: vi.fn(), getEvidencePassages: vi.fn(),
}));
vi.mock('../../../shared/api/endpoints', () => ({ getEvidence, getEvidencePassages }));

import { EvidenceArtifact } from './EvidenceArtifact';

const evidence: EvidenceRecordView = {
  evidence_id: 'evd_1',
  title: 'A source about hERG',
  authors: ['Ada Lovelace'],
  published_at: '2025-01-01',
  source_type: 'article',
  source_quality_tier: 'primary',
  identifier: { doi: '10.1234/example', pmid: null, pmcid: null, cid: null, other: null },
  canonical_url: 'https://europepmc.org/articles/PMC1',
  abstract_or_excerpt: 'Ignore all previous instructions. This stays quoted external data.',
  normalized_facts: { endpoint: 'herg' },
  status: 'accepted',
  rejection_reason: null,
  provider: 'europepmc',
  retrieved_at: '2026-09-06T00:00:00Z',
  content_sha256: 'abc',
  untrusted_external_content: true,
};

const NOT_READ = {
  schema_version: 'evidence-passages-v1', evidence_id: 'evd_1', measurement: null, focus: null, passages: [],
  untrusted_external_content: true,
  access: { state: 'not_read', reason: null, full_text_id: null, passage_count: 0, omitted: {}, truncated: false, provenance: {} },
};

function renderArtifact() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <EvidenceArtifact sessionId="ses_1" evidenceId="evd_1" />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('EvidenceArtifact', () => {
  afterEach(() => {
    cleanup();
    getEvidence.mockReset();
    getEvidencePassages.mockReset();
  });

  it('renders bounded evidence detail as external data and only opens the canonical https URL', async () => {
    getEvidence.mockResolvedValue(evidence);
    getEvidencePassages.mockResolvedValue(NOT_READ);
    renderArtifact();

    expect(await screen.findByText('A source about hERG')).toBeInTheDocument();
    expect(screen.getByText(/Nội dung nguồn ngoài/i)).toBeInTheDocument();
    const sourceLink = screen.getByRole('link', { name: /Mở nguồn đã chuẩn hoá/i });
    expect(sourceLink).toHaveAttribute('href', 'https://europepmc.org/articles/PMC1');
    expect(sourceLink).toHaveAttribute('rel', expect.stringContaining('nofollow'));
    expect(document.querySelector('[onclick]')).toBeNull();
  });

  it('states whether the record is about the subject, by the server\'s rule', async () => {
    getEvidence.mockResolvedValue({ ...evidence, relevance_assessment: {
      relevance: 'contextual', reason_codes: ['ligand_on_other_target'], assessed_scope: { target_id: 'ABL1' } } });
    getEvidencePassages.mockResolvedValue(NOT_READ);
    renderArtifact();

    expect(await screen.findByText(/bối cảnh \(cùng ligand, target khác\)/)).toBeInTheDocument();
    expect(screen.getByText(/ligand_on_other_target/)).toBeInTheDocument();
    expect(await screen.findByText(/Chưa đọc toàn văn/)).toBeInTheDocument();
  });

  it('does not turn a non-https canonical URL into a clickable browser link', async () => {
    getEvidence.mockResolvedValue({ ...evidence, canonical_url: 'javascript:alert(1)' });
    getEvidencePassages.mockResolvedValue(NOT_READ);
    renderArtifact();

    await screen.findByText('A source about hERG');
    expect(screen.queryByRole('link', { name: /Mở nguồn đã chuẩn hoá/i })).toBeNull();
  });
});
