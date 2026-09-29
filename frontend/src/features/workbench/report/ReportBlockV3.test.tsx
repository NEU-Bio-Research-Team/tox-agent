import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

const { getReport, getReportFigure, getReportRendering } = vi.hoisted(() => ({
  getReport: vi.fn(),
  getReportFigure: vi.fn(),
  getReportRendering: vi.fn(),
}));
vi.mock('../../../shared/api/endpoints', () => ({
  getReport,
  getReportFigure,
  getReportRendering,
}));

import { ReportBlock } from './ReportBlock';
import type { ReportArtifact } from '../../../shared/api/types';

const LIMITATION = 'This is a screening report. It is not a safety assessment and does not support a decision about human exposure.';
const FACT = 'fct_' + 'a'.repeat(32);

/** A server-compiled v3 artifact: no claims or tables, fact ids as basis, and a
 * limitations section whose body already is the compiled sentences. */
function v3(): ReportArtifact {
  const section = (section_id: string, body_markdown: string) => ({
    section_id,
    heading: section_id.replace(/_/g, ' '),
    body_markdown,
    claim_ids: [],
    table_ids: [],
    figure_ids: [],
    gap_ids: section_id === 'external_evidence' ? ['gap_1'] : [],
    source_classes: [],
  });
  return {
    schema_version: 'toxagent-report-v3',
    report_id: 'rpt_v3',
    report_build_id: 'rpb_v3',
    analysis_id: 'ana_v3',
    title: 'hERG screening report',
    status: 'completed_with_gaps',
    version: 1,
    subject: {
      canonical_smiles: 'CCO',
      structure_figure_id: null,
      preferred_name: null,
      synonyms: [],
      identifiers: {},
      properties: [],
      source_refs: {},
    },
    sections: [
      section('executive_summary', 'The hERG blocker probability is 0.027.'),
      section('substance_profile', 'Narrative.'),
      section('predictor_results', '| Endpoint | Probability |\n|---|---|\n| herg | 0.027 |'),
      section('explanation_and_visuals', 'Narrative.'),
      section('external_evidence', 'No external literature was consulted for this build.'),
      section('integrated_interpretation', 'Narrative.'),
      section('conclusions', 'Narrative.'),
      section('recommendations', 'Narrative.'),
      section('limitations', `- ${LIMITATION}`),
      section('references', 'No external sources are cited in this report.'),
      section('provenance_appendix', '- compiler: report-compiler-v3'),
    ],
    tables: [],
    figures: [],
    claims: [],
    explanations: [],
    evidence_synthesis: [],
    conclusions: [
      {
        conclusion_id: 'conclusion_c1',
        text: 'This model predicts low hERG blocker liability.',
        basis_claim_ids: [FACT],
        endpoint: 'herg',
        task: null,
        is_integrated: false,
      },
    ],
    recommendations: [],
    references: [],
    gaps: [
      {
        gap_id: 'gap_1',
        reason: 'external_evidence_not_requested',
        detail: 'this build did not ask for a literature search',
        section_id: 'external_evidence',
      },
    ],
    limitations: [{ code: 'screening_not_safety_assessment', text: LIMITATION }],
    provenance: { compiler_version: 'report-compiler-v3' },
    renderings: [{ format: 'markdown', size_bytes: 10 }],
    content_sha256: 'sha256:v3',
    created_at: '2026-09-13T00:00:00Z',
  } as unknown as ReportArtifact;
}

function renderBlock() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ReportBlock sessionId="ses_1" reportId="rpt_v3" />
    </QueryClientProvider>,
  );
}

afterEach(() => vi.clearAllMocks());

describe('ReportBlock with a v3 artifact', () => {
  it('renders the compiled report, its gap and its conclusion without claim rows', async () => {
    getReport.mockResolvedValue(v3());
    renderBlock();

    await waitFor(() => expect(screen.getByText('hERG screening report')).toBeInTheDocument());
    expect(screen.getByText(/The hERG blocker probability is 0\.027\./)).toBeInTheDocument();
    expect(screen.getAllByText(/external_evidence_not_requested/).length).toBeGreaterThan(0);
    expect(screen.getByText(/This model predicts low hERG blocker liability\./)).toBeInTheDocument();
  });

  it('prints each compiled limitation once, not once from the body and again from the list', async () => {
    getReport.mockResolvedValue(v3());
    renderBlock();

    await waitFor(() => expect(screen.getByText('hERG screening report')).toBeInTheDocument());
    expect(screen.getAllByText(new RegExp(LIMITATION.slice(0, 40)))).toHaveLength(1);
  });
});
