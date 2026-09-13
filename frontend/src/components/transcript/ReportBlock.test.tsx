import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

const { getReport, getReportFigure, getReportRendering } = vi.hoisted(() => ({
  getReport: vi.fn(),
  getReportFigure: vi.fn(),
  getReportRendering: vi.fn(),
}));
vi.mock('../../lib/api/endpoints', () => ({
  getReport,
  getReportFigure,
  getReportRendering,
}));

import { ReportBlock } from './ReportBlock';
import type { ReportArtifact } from '../../lib/api/types';

// jsdom implements neither, and `ReportFigure` needs both to turn fetched bytes
// into something an <img> can point at.
URL.createObjectURL = vi.fn(() => 'blob:figure');
URL.revokeObjectURL = vi.fn();

const EVIDENCE = 'evd_' + 'a'.repeat(32);

const SECTION_IDS = [
  'executive_summary',
  'substance_profile',
  'predictor_results',
  'explanation_and_visuals',
  'external_evidence',
  'integrated_interpretation',
  'conclusions',
  'recommendations',
  'limitations',
  'references',
  'provenance_appendix',
] as const;

function reference(overrides: Record<string, unknown> = {}) {
  return {
    evidence_id: EVIDENCE,
    number: 1,
    title: 'hERG blockade of the scaffold',
    provider: 'europepmc',
    canonical_url: 'https://europepmc.org/article/MED/12345',
    link_url: 'https://europepmc.org/article/MED/12345',
    authors: ['Nguyen, T.', 'Tran, H.'],
    published_at: '2024-05-01',
    identifier: { doi: '10.1000/example' },
    source_type: 'article',
    source_quality_tier: 'primary',
    retrieved_at: '2026-09-09T00:00:00+00:00',
    unresolved_reason: null,
    short_form: 'Nguyen, T. et al. (2024)',
    ...overrides,
  };
}

function artifact(overrides: Partial<ReportArtifact> = {}): ReportArtifact {
  const bodies: Record<string, string> = {
    executive_summary: `Screening summary with a citation [@${EVIDENCE}].`,
  };
  return {
    schema_version: 'toxagent-report-v2',
    report_id: 'rpt_1',
    report_build_id: 'rpb_1',
    analysis_id: 'ana_1',
    title: 'Toxicity Screening Report',
    status: 'completed',
    version: 1,
    subject: {
      canonical_smiles: 'CCO',
      structure_figure_id: 'fig_structure',
      preferred_name: 'Ethanol',
      synonyms: ['ethyl alcohol'],
      identifiers: { cid: '702' },
      properties: [],
      source_refs: {},
    },
    sections: SECTION_IDS.map((section_id) => ({
      section_id,
      heading: section_id.replace(/_/g, ' '),
      body_markdown: bodies[section_id] ?? 'Recorded content.',
      claim_ids: section_id === 'predictor_results' ? ['clm_1'] : [],
      figure_ids: [],
      table_ids: section_id === 'predictor_results' ? ['t1'] : [],
      gap_ids: [],
      source_classes: section_id === 'predictor_results'
        ? ['predictor_fact']
        : ['agent_synthesis'],
    })),
    tables: [{
      table_id: 't1',
      title: 'Predictor results',
      columns: ['Endpoint', 'Probability'],
      rows: [['herg', '0.73']],
      source_class: 'predictor_fact',
    }],
    figures: [{
      figure_id: 'fig_structure',
      attachment_id: 'att_1',
      media_type: 'image/svg+xml',
      caption: '2D structure of the analysed compound (CCO).',
      alt_text: 'Two-dimensional chemical structure diagram of Ethanol.',
      content_sha256: 'a'.repeat(64),
      renderer_version: 'toxagent-figure-sanitizer-v1',
      endpoint: null,
      task: null,
      observation_id: null,
    }],
    claims: [{
      claim_id: 'clm_1',
      kind: 'numeric',
      text: 'hERG blockade probability.',
      observation_id: 'obs_abcdef123456',
      field_path: 'predictions.herg.probability_blocker',
      rendered_value: '0.73',
      citation_ids: [],
    }],
    explanations: [],
    evidence_synthesis: [{
      synthesis_id: 'syn_1',
      proposition: 'The scaffold is associated with hERG blockade.',
      relation: 'supports',
      evidence_ids: [EVIDENCE],
      endpoint: 'herg',
      assay: 'patch clamp',
      organism: null,
      dose_context: null,
      quality_notes: [],
      conflict_id: null,
    }],
    conclusions: [{
      conclusion_id: 'cnl_1',
      text: 'Screening flags hERG for confirmation.',
      basis_claim_ids: ['clm_1'],
      endpoint: 'herg',
      task: null,
      is_integrated: false,
    }],
    recommendations: [{
      recommendation_id: 'rec_1',
      text: 'Run a patch-clamp confirmation assay.',
      basis_claim_ids: ['clm_1'],
      action_category: 'confirmatory_assay',
      priority: 'high',
      rationale: 'The screening probability is above threshold.',
      conditions: '',
    }],
    references: [reference()],
    gaps: [],
    limitations: [{ code: 'screening_not_safety_assessment', text: 'Screening only.' }],
    provenance: { compiler_version: 'toxagent-report-compiler-v2', artifact_hashes: ['h'] },
    renderings: [
      { format: 'markdown', size_bytes: 10 },
      { format: 'markdown_bundle', size_bytes: 20 },
      { format: 'html', size_bytes: 30 },
    ],
    content_sha256: 'f'.repeat(64),
    created_at: '2026-09-09T00:00:00+00:00',
    ...overrides,
  } as ReportArtifact;
}

function renderBlock() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ReportBlock sessionId="ses_1" reportId="rpt_1" />
    </QueryClientProvider>,
  );
}

describe('ReportBlock', () => {
  afterEach(() => {
    getReport.mockReset();
    getReportFigure.mockReset();
    getReportRendering.mockReset();
  });

  it('renders predictor tables, not just section prose', async () => {
    // The failure this replaces: ReportBlock rendered body_markdown and nothing
    // else, so table_ids, figure_ids, claims and references were data the
    // artifact carried and the reader never saw.
    getReport.mockResolvedValue(artifact());
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    expect(await screen.findByText('Predictor results')).toBeInTheDocument();
    // Twice: once in the table, once in the claim-traceability disclosure.
    expect(screen.getAllByText('0.73').length).toBeGreaterThan(0);
    expect(screen.getByText('Endpoint')).toBeInTheDocument();
  });

  it('turns an inline citation token into a link to the reference', async () => {
    getReport.mockResolvedValue(artifact());
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    // The marker appears inline in the sentence and again in the section's
    // "Nguồn của mục này" line; both must point at the same reference.
    const markers = await screen.findAllByTitle('hERG blockade of the scaffold');
    expect(markers.length).toBeGreaterThan(0);
    for (const marker of markers) {
      expect(marker).toHaveTextContent('[1]');
      expect(marker).toHaveAttribute('href', `#report-rpt_1-reference-${EVIDENCE}`);
    }
    // The raw token must never reach the reader.
    expect(screen.queryByText(new RegExp(EVIDENCE))).not.toBeInTheDocument();
  });

  it('opens a source in a new tab with a hardened rel', async () => {
    getReport.mockResolvedValue(artifact());
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    const link = await screen.findByRole('link', { name: /Mở nguồn/i });
    expect(link).toHaveAttribute('href', 'https://europepmc.org/article/MED/12345');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer nofollow');
  });

  it('shows citation metadata but no link when the URL is not safe', async () => {
    getReport.mockResolvedValue(artifact({
      references: [reference({ link_url: null, canonical_url: 'javascript:alert(1)' })],
    }));
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    expect(await screen.findByText(/Không có link HTTPS an toàn/i)).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /Mở nguồn/i })).not.toBeInTheDocument();
    // Refusing the link is not refusing the citation.
    expect(screen.getByText('hERG blockade of the scaffold')).toBeInTheDocument();
  });

  it('warns visibly when a cited source has no snapshot behind it', async () => {
    getReport.mockResolvedValue(artifact({ references: [] }));
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    const markers = await screen.findAllByTitle(new RegExp(`Nguồn ${EVIDENCE}`));
    expect(markers.length).toBeGreaterThan(0);
    expect(markers[0]).toHaveTextContent('[?]');
  });

  it('flags a reference the compiler could not resolve', async () => {
    getReport.mockResolvedValue(artifact({
      references: [reference({ unresolved_reason: 'the record is rejected, not accepted' })],
    }));
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    expect(await screen.findByText(/Chưa giải được/i)).toBeInTheDocument();
  });

  it('fetches a figure with auth and shows it', async () => {
    getReport.mockResolvedValue(artifact());
    getReportFigure.mockResolvedValue(new Blob(['<svg/>'], { type: 'image/svg+xml' }));
    renderBlock();

    const image = await screen.findByAltText(/structure diagram of Ethanol/i);
    await waitFor(() => expect(image).toHaveAttribute('src'));
    expect(getReportFigure).toHaveBeenCalledWith('ses_1', 'rpt_1', 'fig_structure');
  });

  it('falls back to the alt text rather than a broken image', async () => {
    getReport.mockResolvedValue(artifact());
    getReportFigure.mockRejectedValue(new Error('404'));
    renderBlock();

    // The alt text is written to carry the picture's content, so a reader who
    // cannot see it still gets the substance instead of an apology.
    expect(
      await screen.findByText(/Không tải được hình/i),
    ).toHaveTextContent(/structure diagram of Ethanol/i);
  });

  it('names each source class so a model number is never read as a published one', async () => {
    getReport.mockResolvedValue(artifact());
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    // Once on the section heading, once on the table it contains.
    expect((await screen.findAllByText('Model')).length).toBeGreaterThan(0);
    expect(screen.getAllByText('Tổng hợp của agent').length).toBeGreaterThan(0);
  });

  it('links each gap to the section it affects', async () => {
    getReport.mockResolvedValue(artifact({
      status: 'completed_with_gaps',
      gaps: [{
        gap_id: 'gap_1',
        reason: 'no_relevant_evidence',
        detail: 'The provider returned nothing relevant.',
        section_id: 'external_evidence',
      }],
      evidence_synthesis: [],
    }));
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    const link = await screen.findByRole('link', { name: 'no_relevant_evidence' });
    expect(link).toHaveAttribute('href', '#report-rpt_1-external_evidence');
    // And the gap itself sits in the affected section, not only in the header.
    expect(screen.getByText(/The provider returned nothing relevant/)).toBeInTheDocument();
  });

  it('offers the markdown bundle as a distinct download', async () => {
    getReport.mockResolvedValue(artifact());
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    // The plain .md links its figures at relative paths; the bundle is what
    // makes those paths resolve, so it has to be visibly a different thing.
    expect(await screen.findByText(/MD \+ hình/i)).toBeInTheDocument();
  });

  it('shows an explanation gap with its reason code, not a bare failure', async () => {
    getReport.mockResolvedValue(artifact({
      explanations: [{
        explanation_id: 'xpl_1',
        observation_id: 'obs_1',
        endpoint: 'herg',
        task: null,
        method: null,
        status: 'failed',
        figure: null,
        extracted_highlights: {
          positive_contributors: [],
          negative_contributors: [],
          unmapped_importance: null,
        },
        failure_reason: 'attribution_unsupported',
      }],
    }));
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    expect(await screen.findByText(/attribution_unsupported/)).toBeInTheDocument();
  });

  it('shows signed contributors in both directions with the target named', async () => {
    getReport.mockResolvedValue(artifact({
      explanations: [{
        explanation_id: 'xpl_1',
        observation_id: 'obs_1',
        endpoint: 'herg',
        task: null,
        method: 'grad_x_input',
        status: 'completed',
        figure: null,
        extracted_highlights: {
          positive_contributors: [{ atom_index: 0, symbol: 'C', signed_contribution: 0.5 }],
          negative_contributors: [{ atom_index: 1, symbol: 'N', signed_contribution: -0.3 }],
          unmapped_importance: 0.35,
        },
        failure_reason: null,
      }],
    }));
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    // Direction is in the text as well as the colour: red and green are the
    // classic indistinguishable pair (XAI-02).
    expect(await screen.findByText('+0.5000')).toBeInTheDocument();
    expect(screen.getByText('-0.3000')).toBeInTheDocument();
    expect(screen.getAllByText(/tăng herg/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/giảm herg/i).length).toBeGreaterThan(0);
    // Unmapped mass is stated, never omitted.
    expect(screen.getByText(/35\.0%/)).toBeInTheDocument();
  });

  it('renders a v1 artifact that has no reference snapshot', async () => {
    getReport.mockResolvedValue(artifact({
      schema_version: 'toxagent-report-v1',
      references: undefined,
      sections: SECTION_IDS.map((section_id) => ({
        section_id,
        heading: section_id.replace(/_/g, ' '),
        body_markdown: 'Recorded content.',
        claim_ids: [],
        figure_ids: [],
        table_ids: [],
        gap_ids: [],
      })),
      evidence_synthesis: [],
    }));
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    expect(await screen.findByText('Toxicity Screening Report')).toBeInTheDocument();
    expect(
      screen.getByText(/không trích dẫn nguồn ngoài hệ thống/i),
    ).toBeInTheDocument();
  });

  it('warns when a section references a figure the report does not carry', async () => {
    getReport.mockResolvedValue(artifact({
      sections: SECTION_IDS.map((section_id) => ({
        section_id,
        heading: section_id.replace(/_/g, ' '),
        body_markdown: 'Recorded content.',
        claim_ids: [],
        figure_ids: section_id === 'explanation_and_visuals' ? ['fig_missing'] : [],
        table_ids: [],
        gap_ids: [],
      })),
    }));
    getReportFigure.mockRejectedValue(new Error('no bytes in this test'));
    renderBlock();

    expect(await screen.findByText(/tham chiếu hình/i)).toBeInTheDocument();
  });
});
