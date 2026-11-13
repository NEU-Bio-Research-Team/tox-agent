import { Link } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { Activity, Check, FileText, Quote } from 'lucide-react';
import { getAnswer, getRunResult } from '../../../shared/api/endpoints';
import type { Claim } from '../../../shared/api/types';
import { getDeveloperModeEnabled } from '../../../shared/lib/preferences';
import { Callout } from '../../../shared/ui/callout';
import { LimitationBlock } from '../answer/LimitationBlock';
import { ReleaseBanner } from '../answer/ReleaseBanner';
import { scrollToRun } from '../lib/scrollToRun';
import { describeLocator } from '../lib/describeLocator';
import { artifactPath } from '../hooks/useArtifactSelection';
import { ArtifactUnavailable } from './ArtifactUnavailable';
import { StageLineage } from './StageLineage';

const SOURCE_LINK =
  'inline-flex items-center gap-1.5 rounded-full border border-line bg-surface-solid px-2.5 py-1 text-xs font-medium text-link transition-colors hover:bg-surface-muted';

/** One point the answer makes, with where it came from: the model result it was
 * read from, the papers it cites, and the passages it quotes. */
function ClaimCard({ claim, sessionId }: { claim: Claim; sessionId: string }) {
  return (
    <li className="rounded-xl border border-line bg-card p-3.5">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-success text-canvas" aria-hidden="true">
          <Check className="size-3" strokeWidth={3} />
        </span>
        <div className="min-w-0 flex-1 space-y-2">
          <p className="text-13 leading-snug text-ink">{claim.text}</p>
          {claim.rendered_value !== undefined && (
            <p className="text-xs text-ink-tertiary">
              Con số trong câu trả lời: <span className="font-mono font-medium text-ink">{claim.rendered_value}</span>
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            {claim.observation_id && (
              <Link
                to={artifactPath(sessionId, { kind: 'observation', entityId: claim.observation_id, fieldPath: claim.field_path })}
                className={SOURCE_LINK}
              >
                <Activity className="size-3.5" aria-hidden="true" />
                Kết quả gốc của mô hình
              </Link>
            )}
            {claim.citation_ids.map((evidenceId, index) => (
              <Link key={evidenceId} to={`/s/${sessionId}/evidence/${evidenceId}`} className={SOURCE_LINK}>
                <FileText className="size-3.5" aria-hidden="true" />
                Nguồn tham khảo {claim.citation_ids.length > 1 ? index + 1 : ''}
              </Link>
            ))}
            {(claim.evidence_spans ?? []).map((span) => (
              <Link
                key={`${span.evidence_id}:${span.locator}`}
                to={artifactPath(sessionId, { kind: 'evidence', entityId: span.evidence_id, locator: span.locator })}
                className={SOURCE_LINK}
              >
                <Quote className="size-3.5" aria-hidden="true" />
                Xem đoạn trích · {describeLocator(span.locator)}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </li>
  );
}

/**
 * "Cách câu trả lời này được kiểm tra" (plan section 8.2.1): how the answer was
 * checked, the points it makes and where each comes from, and the path it took
 * before it reached the reader — not a second copy of `answer_markdown` (that
 * stays the chat's job, section 5.3). Codes, hashes and ids sit in "Chi tiết kỹ
 * thuật", closed unless opened.
 */
export function AnswerAuditViewer({ sessionId, answerId }: { sessionId: string; answerId: string }) {
  const query = useQuery({
    queryKey: ['answer', sessionId, answerId],
    queryFn: () => getAnswer(sessionId, answerId),
  });
  const runId = query.data?.run_id;
  const result = useQuery({
    queryKey: ['run-result', sessionId, runId],
    queryFn: () => getRunResult(sessionId, runId as string),
    enabled: Boolean(runId),
    retry: false,
  });

  if (query.isLoading) {
    return <p className="text-sm text-ink-secondary">Đang tải…</p>;
  }
  if (query.isError || !query.data) {
    return <ArtifactUnavailable artifact="câu trả lời" error={query.error} />;
  }

  const answer = query.data;
  const technical = getDeveloperModeEnabled();

  return (
    <div className="space-y-5 text-sm">
      <header className="space-y-1">
        <h2 className="text-base font-semibold text-ink">Cách câu trả lời này được kiểm tra</h2>
        <p className="text-13 leading-snug text-ink-secondary">
          Tạo lúc {new Date(answer.created_at).toLocaleString('vi-VN')}. Mỗi ý bên dưới được nối với kết quả mô hình hoặc nguồn mà nó dựa vào.
        </p>
        <button type="button" onClick={() => scrollToRun(answer.run_id)} className="text-xs font-medium text-link underline">
          Về tin nhắn tạo ra câu trả lời
        </button>
      </header>

      <ReleaseBanner result={result.data} />

      {answer.is_fallback && (
        <Callout tone="warning" title="Bản tóm tắt tự động" className="text-13">
          Mô hình chưa viết được câu trả lời đạt yêu cầu sau hai lần thử, nên hệ thống dựng bản này từ những số liệu đã được xác nhận.
          {technical ? (
            <>
              {' '}
              <Link to={`/s/${sessionId}/runs/${answer.run_id}?tab=validation`} className="font-medium underline">
                Xem lý do bản trước bị bác
              </Link>
            </>
          ) : null}
        </Callout>
      )}

      <section aria-labelledby="audit-claims" className="space-y-2">
        <h3 id="audit-claims" className="text-xs font-semibold uppercase tracking-wider text-ink-tertiary">
          Các ý chính · {answer.claims.length}
        </h3>
        <ul className="space-y-2">
          {answer.claims.map((claim) => (
            <ClaimCard key={claim.claim_id} claim={claim} sessionId={sessionId} />
          ))}
        </ul>
        {answer.claims.length === 0 && (
          <p className="text-xs italic text-ink-tertiary">Câu trả lời này không có ý nào cần dẫn nguồn (không có con số hay nhận định cần trích dẫn).</p>
        )}
      </section>

      <LimitationBlock limitations={answer.limitations} />

      <section className="space-y-2" aria-labelledby="audit-journey">
        <h3 id="audit-journey" className="text-xs font-semibold uppercase tracking-wider text-ink-tertiary">
          Hành trình của câu trả lời
        </h3>
        <StageLineage sessionId={sessionId} runId={answer.run_id} />
      </section>

      {answer.recommended_next_steps.length > 0 && (
        <section className="space-y-1.5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-tertiary">Gợi ý bước tiếp theo</h3>
          <ul className="space-y-1">
            {answer.recommended_next_steps.map((step, index) => (
              <li key={index} className="text-xs text-ink-secondary">
                • {step.text}
                {step.basis_claim_ids.length === 0 && <span className="ml-1 italic text-ink-tertiary">(chưa gắn với ý nào)</span>}
              </li>
            ))}
          </ul>
        </section>
      )}

      <details className="rounded-lg border border-line bg-canvas-subtle px-3 py-2 text-xs">
        <summary className="cursor-pointer font-medium text-ink-secondary">Chi tiết kỹ thuật</summary>
        <dl className="mt-2 space-y-1.5 font-mono text-ink-secondary">
          <div><dt className="text-ink-tertiary">mã câu trả lời</dt><dd className="break-all">{answer.answer_id}</dd></div>
          <div><dt className="text-ink-tertiary">bản nháp</dt><dd>{answer.candidate_generation}/2 · sha256 {answer.content_sha256.slice(0, 16)}…</dd></div>
          <div>
            <dt className="text-ink-tertiary">lượt chạy</dt>
            <dd className="break-all">
              <Link to={`/s/${sessionId}/runs/${answer.run_id}`} className="underline">{answer.run_id}</Link>
            </dd>
          </div>
          {answer.claims.map((claim, index) => (
            <div key={claim.claim_id}>
              <dt className="text-ink-tertiary">ý {index + 1} · {claim.kind} · {claim.transform}</dt>
              <dd className="break-all">{claim.field_path ?? '—'}{claim.source_value !== undefined ? ` = ${String(claim.source_value)}` : ''}</dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  );
}
