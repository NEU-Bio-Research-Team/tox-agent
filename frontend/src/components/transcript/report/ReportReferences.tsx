import { AlertTriangle, Copy, ExternalLink } from 'lucide-react';
import { toast } from 'sonner';
import type { ReportReference } from '../../../lib/api/types';

/** The citation as a line of text somebody can paste into a manuscript. Built
 * from the artifact's own snapshot, so it says what the report was written
 * against rather than what the source says today. */
function citationText(reference: ReportReference): string {
  const parts = [
    reference.authors.length > 0 ? reference.authors.join(', ') : reference.provider,
    reference.published_at ? `(${reference.published_at.slice(0, 4)})` : null,
    reference.title,
    reference.identifier.doi ? `doi:${reference.identifier.doi}` : null,
    reference.identifier.pmid ? `PMID:${reference.identifier.pmid}` : null,
    reference.canonical_url,
  ];
  return parts.filter(Boolean).join('. ');
}

/** REP-01: `[n]`, as an in-page link to the full citation.
 *
 * Jumps within the report rather than straight out to the source: a reader
 * lands on the title, authors and provider and decides for themselves whether
 * to leave. A number with no snapshot behind it renders as a visible warning —
 * a citation that vanished would leave the sentence reading as the report's own
 * unsupported assertion. */
export function CitationMarker({
  reportId,
  reference,
  evidenceId,
}: {
  reportId: string;
  reference: ReportReference | undefined;
  evidenceId: string;
}) {
  if (!reference) {
    return (
      <span
        className="font-semibold"
        style={{ color: 'var(--accent-red)' }}
        title={`Nguồn ${evidenceId} không giải được trong báo cáo này`}
      >
        [?]
      </span>
    );
  }
  return (
    <a
      href={`#report-${reportId}-reference-${reference.evidence_id}`}
      className="font-semibold no-underline hover:underline"
      style={{ color: 'var(--accent-blue)' }}
      title={reference.title}
    >
      [{reference.number}
      {reference.unresolved_reason ? '!' : ''}]
    </a>
  );
}

/** The sources one section draws on, derived from that section's own claims and
 * inline markers — never asserted, so it cannot list a paper the section did not
 * cite. */
export function SectionSources({
  reportId,
  references,
  evidenceIds,
}: {
  reportId: string;
  references: Map<string, ReportReference>;
  evidenceIds: string[];
}) {
  if (evidenceIds.length === 0) return null;
  return (
    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
      Nguồn của mục này:{' '}
      {evidenceIds.map((evidenceId, index) => (
        <span key={evidenceId}>
          {index > 0 ? '; ' : ''}
          <CitationMarker
            reportId={reportId}
            reference={references.get(evidenceId)}
            evidenceId={evidenceId}
          />{' '}
          {references.get(evidenceId)?.short_form ?? evidenceId}
        </span>
      ))}
    </p>
  );
}

export function ReportReferences({
  reportId,
  references,
}: {
  reportId: string;
  references: ReportReference[];
}) {
  if (references.length === 0) {
    return (
      <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
        Báo cáo này không trích dẫn nguồn ngoài hệ thống.
      </p>
    );
  }
  return (
    <ol className="space-y-3">
      {[...references]
        .sort((a, b) => a.number - b.number)
        .map((reference) => (
          <li
            key={reference.evidence_id}
            id={`report-${reportId}-reference-${reference.evidence_id}`}
            className="scroll-mt-20 text-sm"
          >
            <div className="flex items-start gap-2">
              <span className="font-semibold tabular-nums">[{reference.number}]</span>
              <div className="min-w-0 space-y-1">
                <p className="font-medium">{reference.title}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {[
                    reference.authors.join(', ') || null,
                    reference.published_at?.slice(0, 4) || null,
                    reference.provider,
                    reference.source_quality_tier,
                    ...Object.entries(reference.identifier).map(
                      ([key, value]) => `${key.toUpperCase()}: ${value}`,
                    ),
                    reference.retrieved_at
                      ? `truy xuất ${reference.retrieved_at.slice(0, 10)}`
                      : null,
                  ]
                    .filter(Boolean)
                    .join(' · ')}
                </p>
                <div className="flex flex-wrap items-center gap-3 text-xs">
                  {reference.link_url ? (
                    <a
                      href={reference.link_url}
                      target="_blank"
                      // `noopener noreferrer` so the opened page cannot reach
                      // back into this one, `nofollow` because a cited source
                      // is data the report quotes, not a link it endorses.
                      rel="noopener noreferrer nofollow"
                      className="inline-flex items-center gap-1 underline"
                      style={{ color: 'var(--accent-blue)' }}
                    >
                      <ExternalLink size={12} aria-hidden /> Mở nguồn
                    </a>
                  ) : (
                    <span
                      className="inline-flex items-center gap-1"
                      style={{ color: 'var(--text-muted)' }}
                      // The URL is recorded but was refused as a link. Saying so
                      // beats a link a reader clicks that goes nowhere safe.
                      title={reference.canonical_url ?? 'không có URL'}
                    >
                      <AlertTriangle size={12} aria-hidden /> Không có link HTTPS an toàn
                    </span>
                  )}
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 underline"
                    style={{ color: 'var(--text-muted)' }}
                    onClick={() => {
                      void navigator.clipboard
                        ?.writeText(citationText(reference))
                        .then(() => toast.success('Đã copy citation'))
                        .catch(() => toast.error('Không copy được citation'));
                    }}
                  >
                    <Copy size={12} aria-hidden /> Copy citation
                  </button>
                </div>
                {reference.unresolved_reason && (
                  <p className="text-xs" style={{ color: 'var(--accent-red)' }}>
                    Chưa giải được: {reference.unresolved_reason}
                  </p>
                )}
              </div>
            </div>
          </li>
        ))}
    </ol>
  );
}
