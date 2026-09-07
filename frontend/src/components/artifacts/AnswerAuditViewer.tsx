import { Link } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { getAnswer } from '../../lib/api/endpoints';
import { LimitationBlock } from '../answer/LimitationBlock';
import { FallbackBadge } from '../answer/FallbackBadge';
import { scrollToRun } from '../../lib/scrollToRun';
import { artifactPath } from '../../hooks/useArtifactSelection';
import { ArtifactUnavailable } from './ArtifactUnavailable';

/**
 * "Kiểm chứng answer" viewer (plan section 8.2.1): claims, observation
 * drill-down, hash and fallback/validation — the provenance chain, not a
 * second copy of `answer_markdown` (that stays the chat's job, section 5.3).
 * Supersedes the standalone `AuditPage`: this same content now opens inside
 * the workspace via `/s/:sessionId/answers/:answerId`, so a shared link and
 * an in-chat "kiểm chứng" click land on the identical view.
 */
export function AnswerAuditViewer({ sessionId, answerId }: { sessionId: string; answerId: string }) {
  const query = useQuery({
    queryKey: ['answer', sessionId, answerId],
    queryFn: () => getAnswer(sessionId, answerId),
  });

  if (query.isLoading) {
    return <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Đang tải…</p>;
  }
  if (query.isError || !query.data) {
    return <ArtifactUnavailable artifact="đáp án" error={query.error} />;
  }

  const answer = query.data;

  return (
    <div className="space-y-4 text-sm">
      <div className="space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <p className="font-mono text-sm" style={{ color: 'var(--text)' }}>
            {answer.answer_id}
          </p>
          {answer.is_fallback && <FallbackBadge />}
        </div>
        <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
          candidate {answer.candidate_generation}/2 · sha256 {answer.content_sha256.slice(0, 16)}… ·{' '}
          {new Date(answer.created_at).toLocaleString('vi-VN')}
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <Link
            to={`/s/${sessionId}/runs/${answer.run_id}`}
            className="font-mono text-xs underline"
            style={{ color: 'var(--accent-blue)' }}
          >
            run {answer.run_id} →
          </Link>
          <button
            type="button"
            onClick={() => scrollToRun(answer.run_id)}
            className="text-xs font-medium underline"
            style={{ color: 'var(--accent-blue)' }}
          >
            về lượt chat tạo kết quả
          </button>
        </div>
      </div>

      {answer.is_fallback && (
        <div className="flex flex-col items-start gap-2 rounded-lg border p-3" style={{ borderColor: 'var(--accent-yellow)' }}>
          <FallbackBadge />
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Model thất bại validator hai lần; đây là đáp án dự phòng deterministic. Xem lý do candidate trước bị bác:
          </p>
          <Link
            to={`/s/${sessionId}/runs/${answer.run_id}?tab=validation`}
            className="text-xs font-medium underline"
            style={{ color: 'var(--accent-blue)' }}
          >
            kiểm định của run này →
          </Link>
        </div>
      )}

      <div>
        <p className="mb-2 text-xs font-semibold" style={{ color: 'var(--text)' }}>
          {answer.claims.length} claim
        </p>
        <div className="space-y-2">
          {answer.claims.map((claim, index) => (
            <div key={claim.claim_id} className="rounded-lg p-2.5 text-xs" style={{ backgroundColor: 'var(--surface-alt)' }}>
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="font-mono font-semibold" style={{ color: 'var(--accent-blue)' }}>
                  #{index + 1} · {claim.kind}
                </span>
                <span style={{ color: 'var(--text-faint)' }}>{claim.transform}</span>
              </div>
              <p className="mb-1.5" style={{ color: 'var(--text-muted)' }}>
                {claim.text}
              </p>
              <dl className="space-y-0.5 font-mono" style={{ color: 'var(--text-faint)' }}>
                {claim.rendered_value !== undefined && (
                  <div className="flex justify-between gap-2">
                    <dt>hiển thị</dt>
                    <dd className="truncate">{claim.rendered_value}</dd>
                  </div>
                )}
                {claim.source_value !== undefined && (
                  <div className="flex justify-between gap-2">
                    <dt>nguồn</dt>
                    <dd className="truncate">{String(claim.source_value)}</dd>
                  </div>
                )}
                {claim.field_path && (
                  <div className="flex justify-between gap-2">
                    <dt>field_path</dt>
                    <dd className="truncate">{claim.field_path}</dd>
                  </div>
                )}
              </dl>
              {claim.observation_id && (
                <Link
                  to={artifactPath(sessionId, {
                    kind: 'observation',
                    entityId: claim.observation_id,
                    fieldPath: claim.field_path,
                  })}
                  className="mt-1.5 inline-block font-medium underline"
                  style={{ color: 'var(--accent-blue)' }}
                >
                  mở observation {claim.observation_id.slice(0, 12)}… →
                </Link>
              )}
              {claim.citation_ids.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-x-2 gap-y-1">
                  {claim.citation_ids.map((evidenceId) => (
                    <Link
                      key={evidenceId}
                      to={`/s/${sessionId}/evidence/${evidenceId}`}
                      className="font-medium underline"
                      style={{ color: 'var(--accent-blue)' }}
                    >
                      evidence {evidenceId.slice(0, 12)}… →
                    </Link>
                  ))}
                </div>
              )}
            </div>
          ))}
          {answer.claims.length === 0 && (
            <p className="text-xs italic" style={{ color: 'var(--text-faint)' }}>
              Đáp án này không có claim nào (không có con số cần trích dẫn).
            </p>
          )}
        </div>
      </div>

      <LimitationBlock limitations={answer.limitations} />

      {answer.recommended_next_steps.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-semibold" style={{ color: 'var(--text)' }}>
            Bước tiếp theo đề xuất
          </p>
          <ul className="space-y-1">
            {answer.recommended_next_steps.map((step, index) => (
              <li key={index} className="text-xs" style={{ color: 'var(--text-muted)' }}>
                • {step.text}
                {step.basis_claim_ids.length === 0 && (
                  <span className="ml-1 italic" style={{ color: 'var(--text-faint)' }}>
                    (không neo claim)
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
