import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router';
import { listAllEvidence } from '../../lib/api/endpoints';
import type { Claim, EvidenceRecordView } from '../../lib/api/types';

/**
 * The literature an answer cites, listed under the answer (K11 S3 / K08).
 *
 * Citations were reachable only by hovering a claim chip, and a chip only
 * exists where `linkifyClaims` could anchor a `rendered_value` in the prose.
 * `scientific`, `limitation` and `recommendation` claims are not field-backed,
 * so they have no rendered value and never became chips — which meant a claim
 * whose entire basis was a cited paper displayed its citation nowhere. The one
 * kind of claim a reader most needs a source for was the one that showed none.
 *
 * Built from `claim.citation_ids`, never from the prose: a source appears here
 * because the answer record says a claim cites it, not because something
 * matched text. An id whose record cannot be loaded is shown as an unresolved
 * id rather than dropped — a citation that silently disappears is worse than
 * one that admits it could not be read.
 */
export function AnswerSources({
  claims,
  sessionId,
}: {
  claims: Claim[];
  sessionId: string;
}) {
  // Ordered by first appearance, so the numbering matches reading order.
  const citedIds: string[] = [];
  for (const claim of claims) {
    for (const id of claim.citation_ids) {
      if (!citedIds.includes(id)) citedIds.push(id);
    }
  }

  const query = useQuery({
    queryKey: ['evidence', sessionId, 'accepted'],
    queryFn: () => listAllEvidence(sessionId, { status: 'accepted' }),
    enabled: citedIds.length > 0,
  });

  if (citedIds.length === 0) return null;

  const byId = new Map<string, EvidenceRecordView>(
    (query.data ?? []).map((record) => [record.evidence_id, record]),
  );

  return (
    <section
      className="space-y-1.5 border-t pt-3"
      aria-labelledby="answer-sources-heading"
      style={{ borderColor: 'var(--border)' }}
    >
      <h3
        id="answer-sources-heading"
        className="text-xs font-semibold"
        style={{ color: 'var(--text)' }}
      >
        Nguồn được trích dẫn ({citedIds.length})
      </h3>
      {query.isLoading && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Đang tải nguồn…
        </p>
      )}
      <ol className="space-y-1">
        {citedIds.map((evidenceId, index) => {
          const record = byId.get(evidenceId);
          return (
            <li key={evidenceId} className="text-xs" style={{ color: 'var(--text-muted)' }}>
              <span style={{ color: 'var(--text-faint)' }}>[{index + 1}]</span>{' '}
              <Link
                to={`/s/${sessionId}/evidence/${evidenceId}`}
                className="font-medium underline"
                style={{ color: 'var(--accent-blue)' }}
              >
                {/* The record's own title, never a title composed here. When
                    it cannot be loaded the id is shown: a citation that
                    disappears quietly is worse than one that says so. */}
                {record ? record.title : `evidence ${evidenceId}`}
              </Link>
              {record && (
                <span>
                  {record.published_at ? ` · ${record.published_at.slice(0, 4)}` : ''}
                  {` · ${record.provider}`}
                </span>
              )}
              {!record && !query.isLoading && (
                <span className="ml-1 italic" style={{ color: 'var(--text-faint)' }}>
                  (không đọc được bản ghi)
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
