import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { toast } from 'sonner';
import { Download, FileWarning } from 'lucide-react';
import { getReport, getReportRendering } from '../../lib/api/endpoints';
import type {
  ReportArtifact,
  ReportClaim,
  ReportExplanation,
  ReportFormat,
  ReportReference,
  ReportSection,
} from '../../lib/api/types';
import { ExplanationLegend } from './report/ExplanationLegend';
import { ReportFigure } from './report/ReportFigure';
import { citedInProse, ReportProse } from './report/ReportProse';
import { ReportReferences, SectionSources } from './report/ReportReferences';
import { ReportTable, SourceClassBadge } from './report/ReportTable';

/** The filename a download gets. `markdown_bundle` is a zip of `report.md` plus
 * a `figures/` directory — the plain `.md` links its images at relative paths,
 * and the bundle is what makes those paths resolve. */
const DOWNLOAD_SUFFIX: Record<string, string> = {
  markdown: 'md',
  markdown_bundle: 'zip',
  html: 'html',
  pdf: 'pdf',
};

const FORMAT_LABEL: Record<string, string> = {
  markdown: 'MD',
  markdown_bundle: 'MD + hình (.zip)',
  html: 'HTML',
  pdf: 'PDF',
};

export function ReportBlock({ sessionId, reportId }: { sessionId: string; reportId: string }) {
  const query = useQuery({
    queryKey: ['report', sessionId, reportId],
    queryFn: () => getReport(sessionId, reportId),
  });

  if (query.isLoading) return <p className="text-sm text-muted-foreground">Đang tải báo cáo…</p>;
  if (query.isError || !query.data) {
    return <p className="text-sm text-red-600">Không tải được báo cáo {reportId}.</p>;
  }
  return <ReportBody sessionId={sessionId} reportId={reportId} report={query.data} />;
}

function ReportBody({
  sessionId,
  reportId,
  report,
}: {
  sessionId: string;
  reportId: string;
  report: ReportArtifact;
}) {
  const [technical, setTechnical] = useState(true);

  // Every lookup the sections need, built once. A v1 artifact simply has no
  // references, which is a readable older report rather than a broken one.
  const references = useMemo(
    () => new Map((report.references ?? []).map((r) => [r.evidence_id, r])),
    [report.references],
  );
  const claims = useMemo(
    () => new Map((report.claims ?? []).map((c) => [c.claim_id, c])),
    [report.claims],
  );
  const figures = useMemo(
    () => new Map((report.figures ?? []).map((f) => [f.figure_id, f])),
    [report.figures],
  );
  const tables = useMemo(
    () => new Map((report.tables ?? []).map((t) => [t.table_id, t])),
    [report.tables],
  );
  async function download(format: ReportFormat) {
    try {
      const blob = await getReportRendering(sessionId, reportId, format);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `${reportId}.${DOWNLOAD_SUFFIX[format] ?? format}`;
      anchor.click();
      // Revoked after the click has been dispatched; leaving it live pins the
      // whole file in memory for the lifetime of the tab.
      URL.revokeObjectURL(url);
    } catch {
      toast.error(`Không tải được bản ${format}.`);
    }
  }

  return (
    <article className="space-y-5 rounded-xl border p-5" aria-label={report.title}>
      <header className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">{report.title}</h2>
            <p className="text-xs text-muted-foreground">
              Phiên bản {report.version} · {report.content_sha256.slice(0, 12)}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setTechnical((current) => !current)}
              className="rounded-md border px-2 py-1 text-xs"
              aria-pressed={technical}
            >
              {technical ? 'Xem bản tóm tắt' : 'Xem bản đầy đủ'}
            </button>
            {report.renderings.map((item) => (
              <button
                key={item.format}
                type="button"
                onClick={() => void download(item.format)}
                className="inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs"
              >
                <Download size={13} /> {FORMAT_LABEL[item.format] ?? item.format.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {report.status === 'completed_with_gaps' && (
          <div className="space-y-1 rounded-md bg-amber-50 p-2 text-sm text-amber-900">
            <p className="flex items-center gap-2">
              <FileWarning size={16} /> Báo cáo hoàn tất với {report.gaps.length} khoảng
              trống dữ liệu được ghi rõ.
            </p>
            {/* Each gap links to the section it affects, so "completed with
                gaps" is a claim a reader can check rather than a badge. */}
            <ul className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
              {report.gaps.map((gap) => (
                <li key={gap.gap_id}>
                  <a
                    href={`#report-${reportId}-${gap.section_id}`}
                    className="underline"
                  >
                    {gap.reason}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}

        <nav className="flex flex-wrap gap-2" aria-label="Mục báo cáo">
          {report.sections.map((section) => (
            <a
              key={section.section_id}
              href={`#report-${reportId}-${section.section_id}`}
              className="text-xs underline"
            >
              {section.heading}
            </a>
          ))}
        </nav>
      </header>

      {report.sections.map((section) => (
        <SectionView
          key={section.section_id}
          sessionId={sessionId}
          reportId={reportId}
          report={report}
          section={section}
          technical={technical}
          references={references}
          claims={claims}
          figures={figures}
          tables={tables}
        />
      ))}
    </article>
  );
}

/** Which sources a section draws on: its inline markers first, then its claims'
 * citations, then — in the external-evidence section — the synthesis. Derived,
 * never asserted, so the list cannot name a paper the section did not cite. */
function sectionSourceIds(report: ReportArtifact, section: ReportSection): string[] {
  const order = citedInProse(section.body_markdown);
  const add = (evidenceId: string) => {
    if (!order.includes(evidenceId)) order.push(evidenceId);
  };
  const byId = new Map((report.claims ?? []).map((c) => [c.claim_id, c]));
  for (const claimId of section.claim_ids ?? []) {
    for (const evidenceId of byId.get(claimId)?.citation_ids ?? []) add(evidenceId);
  }
  if (section.section_id === 'external_evidence') {
    for (const item of report.evidence_synthesis ?? []) {
      for (const evidenceId of item.evidence_ids) add(evidenceId);
    }
  }
  return order;
}

function SectionView({
  sessionId,
  reportId,
  report,
  section,
  technical,
  references,
  claims,
  figures,
  tables,
}: {
  sessionId: string;
  reportId: string;
  report: ReportArtifact;
  section: ReportSection;
  technical: boolean;
  references: Map<string, ReportReference>;
  claims: Map<string, ReportClaim>;
  figures: Map<string, ReportArtifact['figures'][number]>;
  tables: Map<string, ReportArtifact['tables'][number]>;
}) {
  const structureFigureId = report.subject?.structure_figure_id ?? null;

  return (
    <section
      id={`report-${reportId}-${section.section_id}`}
      className="space-y-2 scroll-mt-20"
    >
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="font-semibold">{section.heading}</h3>
        {(section.source_classes ?? []).map((sourceClass) => (
          <SourceClassBadge key={sourceClass} sourceClass={sourceClass} />
        ))}
      </div>

      {section.body_markdown?.trim() && (
        <ReportProse
          reportId={reportId}
          body={section.body_markdown}
          references={references}
        />
      )}

      {/* The neutral structure drawing, in the section that is about identity.
          Deliberately not the explanation heat map: that is a statement about
          one model on one endpoint, and showing it here would invite a reader
          to take the colours as a property of the compound (REP-02). */}
      {section.section_id === 'substance_profile' && (
        <>
          {structureFigureId && figures.get(structureFigureId) && (
            <ReportFigure
              sessionId={sessionId}
              reportId={reportId}
              figure={figures.get(structureFigureId)!}
            />
          )}
          <SubstanceFacts report={report} />
        </>
      )}

      {(section.table_ids ?? []).map((tableId) => {
        const table = tables.get(tableId);
        return table ? <ReportTable key={tableId} table={table} /> : (
          <MissingRef key={tableId} kind="bảng" id={tableId} />
        );
      })}

      {(section.figure_ids ?? []).map((figureId) => {
        const figure = figures.get(figureId);
        return figure ? (
          <ReportFigure
            key={figureId}
            sessionId={sessionId}
            reportId={reportId}
            figure={figure}
          />
        ) : (
          <MissingRef key={figureId} kind="hình" id={figureId} />
        );
      })}

      {section.section_id === 'explanation_and_visuals' &&
        (report.explanations ?? []).map((explanation) => (
          <ExplanationView
            key={explanation.explanation_id}
            sessionId={sessionId}
            reportId={reportId}
            explanation={explanation}
            technical={technical}
          />
        ))}

      {section.section_id === 'external_evidence' && (
        <EvidenceSynthesisView report={report} reportId={reportId} references={references} />
      )}

      {section.section_id === 'conclusions' && <ConclusionsView report={report} />}
      {section.section_id === 'recommendations' && <RecommendationsView report={report} />}
      {/* A v3 limitations section is compiled: its body already is these
          sentences, so listing them again would print every limitation twice. */}
      {section.section_id === 'limitations' && report.schema_version !== 'toxagent-report-v3' && (
        <LimitationsView report={report} />
      )}
      {section.section_id === 'references' && (
        <ReportReferences reportId={reportId} references={report.references ?? []} />
      )}
      {section.section_id === 'provenance_appendix' && <ProvenanceView report={report} />}

      {/* Claims with their observation field path, so a number in the report can
          be traced back to the predictor response it came from. Only in the
          technical view: a reader of the summary does not need the path, but the
          summary must not be the only view that has the number. */}
      {technical && (section.claim_ids ?? []).length > 0 && (
        <ClaimsView claimIds={section.claim_ids} claims={claims} />
      )}

      {/* Gaps sit in the section they affect, not in a list at the end. A failed
          provider or a failed explainer must not be able to make a section
          quietly complete.

          Matched two ways — the section's own `gap_ids` and any gap naming this
          section — because those are two independent records of the same fact
          and a gap that appears in only one of them would otherwise be a piece
          of missing data that is itself missing. */}
      {sectionGaps(report, section).map((gap) => (
        <p key={gap.gap_id} className="rounded-md bg-amber-50 p-2 text-sm text-amber-900">
          <strong>{gap.reason}:</strong> {gap.detail}
        </p>
      ))}

      <SectionSources
        reportId={reportId}
        references={references}
        evidenceIds={sectionSourceIds(report, section)}
      />
    </section>
  );
}

function sectionGaps(report: ReportArtifact, section: ReportSection) {
  const ids = new Set(section.gap_ids ?? []);
  return (report.gaps ?? []).filter(
    (gap) => ids.has(gap.gap_id) || gap.section_id === section.section_id,
  );
}

function MissingRef({ kind, id }: { kind: string; id: string }) {
  return (
    <p className="rounded-md bg-amber-50 p-2 text-xs text-amber-900">
      Mục này tham chiếu {kind} <code>{id}</code> nhưng báo cáo không chứa nó.
    </p>
  );
}

function SubstanceFacts({ report }: { report: ReportArtifact }) {
  const subject = report.subject;
  if (!subject) return null;
  const identifiers = Object.entries(subject.identifiers ?? {}).filter(([, v]) => v);
  return (
    <dl className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 text-sm">
      {subject.preferred_name && (
        <>
          <dt className="text-muted-foreground">Tên</dt>
          <dd>{subject.preferred_name}</dd>
        </>
      )}
      <dt className="text-muted-foreground">SMILES chuẩn hoá</dt>
      <dd className="break-all font-mono text-xs">{subject.canonical_smiles}</dd>
      {identifiers.length > 0 && (
        <>
          <dt className="text-muted-foreground">Định danh</dt>
          <dd>
            {identifiers.map(([key, value]) => `${key}: ${String(value)}`).join(' · ')}
          </dd>
        </>
      )}
      {(subject.synonyms ?? []).length > 0 && (
        <>
          <dt className="text-muted-foreground">Tên khác</dt>
          <dd>{subject.synonyms.slice(0, 8).join(', ')}</dd>
        </>
      )}
    </dl>
  );
}

function ExplanationView({
  sessionId,
  reportId,
  explanation,
  technical,
}: {
  sessionId: string;
  reportId: string;
  explanation: ReportExplanation;
  technical: boolean;
}) {
  const target = explanation.task
    ? `${explanation.endpoint} / ${explanation.task}`
    : explanation.endpoint;
  return (
    <div className="space-y-2 rounded-md border p-3">
      <p className="text-sm font-medium">{target}</p>
      {explanation.status === 'failed' ? (
        // The reason code, not "explanation failed". Five different causes lead
        // to five different actions (XAI-01).
        <p className="rounded-md bg-amber-50 p-2 text-sm text-amber-900">
          Không tạo được explanation cho {target}
          {explanation.failure_reason ? `: ${explanation.failure_reason}` : '.'}
        </p>
      ) : (
        <>
          {explanation.figure ? (
            <ReportFigure
              sessionId={sessionId}
              reportId={reportId}
              figure={explanation.figure}
            />
          ) : (
            <p className="rounded-md bg-amber-50 p-2 text-sm text-amber-900">
              Có số liệu attribution nhưng không có hình
              {explanation.failure_reason ? `: ${explanation.failure_reason}` : '.'}
            </p>
          )}
          {technical && <ExplanationLegend explanation={explanation} />}
        </>
      )}
      {explanation.status === 'partial' && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Explanation này là một phần, không phải kết quả hoàn chỉnh.
        </p>
      )}
    </div>
  );
}

const RELATION_LABEL: Record<string, string> = {
  supports: 'Ủng hộ',
  contradicts: 'Mâu thuẫn',
  contextualizes: 'Bổ sung ngữ cảnh',
  insufficient: 'Không đủ dữ liệu',
};

function EvidenceSynthesisView({
  report,
  reportId,
  references,
}: {
  report: ReportArtifact;
  reportId: string;
  references: Map<string, ReportReference>;
}) {
  const synthesis = report.evidence_synthesis ?? [];
  if (synthesis.length === 0) return null;
  return (
    <ul className="space-y-2 text-sm">
      {synthesis.map((item) => (
        <li key={item.synthesis_id} className="space-y-1">
          <p>
            <span className="font-medium">
              {RELATION_LABEL[item.relation] ?? item.relation}:
            </span>{' '}
            {item.proposition}
          </p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {[
              item.endpoint ? `endpoint: ${item.endpoint}` : null,
              item.assay ? `assay: ${item.assay}` : null,
              item.organism ? `sinh vật: ${item.organism}` : null,
              item.dose_context ? `liều: ${item.dose_context}` : null,
            ]
              .filter(Boolean)
              .join(' · ')}
          </p>
          <SectionSources
            reportId={reportId}
            references={references}
            evidenceIds={item.evidence_ids}
          />
          {(item.quality_notes ?? []).length > 0 && (
            <ul className="ml-4 list-disc text-xs" style={{ color: 'var(--text-muted)' }}>
              {item.quality_notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </li>
      ))}
    </ul>
  );
}

function ConclusionsView({ report }: { report: ReportArtifact }) {
  return (
    <ul className="space-y-2 text-sm">
      {(report.conclusions ?? []).map((item) => (
        <li key={item.conclusion_id}>
          <span className="font-medium">
            {item.is_integrated
              ? 'Diễn giải tích hợp'
              : [item.endpoint, item.task].filter(Boolean).join(' / ')}
            :
          </span>{' '}
          {item.text}
        </li>
      ))}
    </ul>
  );
}

function RecommendationsView({ report }: { report: ReportArtifact }) {
  return (
    <ul className="space-y-2 text-sm">
      {(report.recommendations ?? []).map((item) => (
        <li key={item.recommendation_id} className="space-y-0.5">
          <p>
            <span className="font-medium">
              {item.priority} · {item.action_category}:
            </span>{' '}
            {item.text}
          </p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Cơ sở: {item.rationale}
            {item.conditions ? ` · Điều kiện: ${item.conditions}` : ''}
          </p>
        </li>
      ))}
    </ul>
  );
}

function LimitationsView({ report }: { report: ReportArtifact }) {
  return (
    <ul className="space-y-1 text-sm">
      {(report.limitations ?? []).map((item) => (
        <li key={item.code}>
          <code className="text-xs font-semibold">{item.code}</code> — {item.text}
        </li>
      ))}
    </ul>
  );
}

function ClaimsView({
  claimIds,
  claims,
}: {
  claimIds: string[];
  claims: Map<string, ReportClaim>;
}) {
  const resolved = claimIds.map((id) => claims.get(id)).filter(Boolean) as ReportClaim[];
  if (resolved.length === 0) return null;
  return (
    <details className="text-xs">
      <summary className="cursor-pointer" style={{ color: 'var(--text-muted)' }}>
        Truy nguồn {resolved.length} phát biểu
      </summary>
      <ul className="mt-1 space-y-1">
        {resolved.map((claim) => (
          <li key={claim.claim_id}>
            <span className="font-medium">{claim.rendered_value ?? claim.kind}</span> —{' '}
            {claim.text}
            {claim.field_path && (
              <span style={{ color: 'var(--text-muted)' }}>
                {' '}
                (<code>{claim.field_path}</code>
                {claim.observation_id ? ` @ ${claim.observation_id.slice(0, 12)}…` : ''})
              </span>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}

/** Human-readable first, raw hashes behind a disclosure. A provenance appendix
 * that opens with JSON is one nobody reads. */
function ProvenanceView({ report }: { report: ReportArtifact }) {
  const provenance = report.provenance ?? {};
  const headline: Array<[string, string]> = [
    ['Model artifact hashes', String((provenance.artifact_hashes as string[])?.length ?? 0)],
    ['Predictor version', String(provenance.predictor_service_version ?? '—')],
    ['Compiler', String(provenance.compiler_version ?? '—')],
    ['Report hash', report.content_sha256.slice(0, 16)],
    ['Tạo lúc', report.created_at],
  ];
  return (
    <div className="space-y-2 text-sm">
      <dl className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1">
        {headline.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="break-all font-mono text-xs">{value}</dd>
          </div>
        ))}
      </dl>
      <details>
        <summary className="cursor-pointer text-xs" style={{ color: 'var(--text-muted)' }}>
          Provenance đầy đủ
        </summary>
        <pre className="mt-1 overflow-x-auto rounded-md border p-2 text-[11px]">
          {JSON.stringify(provenance, null, 2)}
        </pre>
      </details>
    </div>
  );
}
