import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getRunLineage, type LineageClaim, type LineageReview, type LineageStage, type RunStageLineage } from '../../../shared/api/endpoints';
import { ERROR_CODE_LABEL_VI, LIMITATION_LABEL_VI } from '../../../shared/lib/labels';
import { getDeveloperModeEnabled } from '../../../shared/lib/preferences';
import { Callout, type Tone } from '../../../shared/ui/callout';
import { StatusPill } from '../../../shared/ui/status-pill';
import { cn } from '../../../shared/ui/utils';

/**
 * Hành trình của một câu trả lời (AG-UI audit, §10.2, §14.2): bản nháp đầu, các
 * lần sửa, những lần đối chiếu với nguồn, và điều gì đã đến tay người đọc. Hiển
 * thị theo từng ý, vì hai điều người đọc cần đánh giá không thấy được ở câu trả
 * lời cuối: bản sửa đã bỏ hay thu hẹp ý nào, và còn thiếu đầu vào hay lưu ý nào.
 *
 * Mọi thứ đọc từ những gì lượt chạy đã ghi; không phán xét lại. Văn bản của một
 * bản nháp là tư liệu tham khảo, không phải câu trả lời — nhãn nói rõ điều đó.
 * Tên chính sách, mã lỗi và khoá nội bộ chỉ hiện trong "Chi tiết kỹ thuật".
 */

const PHASE_LABEL: Record<string, string> = {
  initial: 'Bản nháp đầu tiên',
  hard_correction: 'Sau khi sửa lỗi định dạng',
  semantic_repair: 'Sau khi sửa theo phần đối chiếu nguồn',
};

const VERDICT_LABEL: Record<string, string> = {
  supported: 'khớp nguồn',
  partially_supported: 'khớp một phần',
  not_supported: 'chưa có nguồn hỗ trợ',
  contradicted: 'bị nguồn phản bác',
  cannot_judge: 'chưa đủ căn cứ',
  no_verdict: 'chưa đánh giá',
};

const CHANGE: Record<LineageClaim['change'], { label: string; tone: Tone }> = {
  kept: { label: 'Giữ nguyên', tone: 'neutral' },
  reworded: { label: 'Đã viết lại', tone: 'info' },
  removed: { label: 'Đã bỏ', tone: 'warning' },
  added: { label: 'Mới thêm', tone: 'success' },
};

export function StageLineage({ sessionId, runId }: { sessionId: string; runId: string }) {
  const query = useQuery({
    queryKey: ['run-lineage', sessionId, runId],
    queryFn: () => getRunLineage(sessionId, runId),
  });
  if (query.isLoading) {
    return <p className="text-xs text-ink-tertiary">Đang tải hành trình…</p>;
  }
  if (query.isError || !query.data) {
    return <p className="text-xs text-danger">Không tải được hành trình của câu trả lời.</p>;
  }
  return <LineageBody lineage={query.data} />;
}

export function LineageBody({ lineage }: { lineage: RunStageLineage }) {
  const technical = getDeveloperModeEnabled();
  if (!lineage.recorded) {
    return (
      <p className="text-xs text-ink-secondary">
        Câu trả lời này được tạo trước khi hệ thống ghi nhận việc kiểm tra, nên chưa rõ nó đã được đối chiếu với nguồn hay chưa.
      </p>
    );
  }
  return (
    <section className="space-y-4 text-xs" aria-label="Hành trình của câu trả lời">
      <ol className="space-y-2" aria-label="Bản nháp theo thứ tự">
        {lineage.stages.map((stage) => <StageRow key={stage.index} stage={stage} technical={technical} />)}
        {lineage.stages.length === 0 && <li className="text-ink-tertiary">Chưa có bản nháp nào được ghi lại.</li>}
      </ol>

      {lineage.reviews.map((review) => <ReviewRound key={review.round} review={review} technical={technical} />)}

      {lineage.claims.length > 0 && (
        <div className="space-y-1.5">
          <p className="font-semibold text-ink">Từng ý, từ bản nháp đến bản bạn đọc</p>
          <ul className="space-y-2" aria-label="Hành trình từng ý">
            {lineage.claims.map((claim) => <ClaimRow key={claim.key} claim={claim} technical={technical} />)}
          </ul>
          {lineage.repair?.removal_only === true && (
            <p className="text-warning">
              Bản sửa chỉ bỏ bớt ý, không viết lại: độ chính xác có thể tăng nhưng thông tin bắt buộc có thể đã mất.
            </p>
          )}
        </div>
      )}

      <Missing lineage={lineage} />

      <details className="rounded-lg border border-line bg-canvas-subtle px-3 py-2">
        <summary className="cursor-pointer font-medium text-ink-secondary">Chi tiết kỹ thuật</summary>
        <dl className="mt-2 space-y-1 font-mono text-ink-secondary">
          <div className="flex justify-between gap-3"><dt>chính sách phát hành</dt><dd className="text-right">{lineage.release_policy}</dd></div>
          <div className="flex justify-between gap-3"><dt>kết quả</dt><dd className="text-right">{lineage.disposition ?? 'chưa ghi'}</dd></div>
          <div className="flex justify-between gap-3"><dt>đối chiếu</dt><dd className="text-right">{lineage.review_status ?? 'chưa ghi'}</dd></div>
          {lineage.execution_outcome && lineage.execution_outcome !== 'completed' ? (
            <div className="flex justify-between gap-3"><dt>thực thi</dt><dd className="text-right">{lineage.execution_outcome}</dd></div>
          ) : null}
        </dl>
      </details>
    </section>
  );
}

function StageRow({ stage, technical }: { stage: LineageStage; technical: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-xl border border-line bg-card p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-ink">
          <span className="font-semibold">{PHASE_LABEL[stage.phase] ?? 'Bản nháp'}</span>
          <span className="text-ink-secondary">
            {' · '}{stage.hard_valid ? 'định dạng hợp lệ' : 'định dạng chưa hợp lệ'}
            {' · '}{stage.claim_count} ý
          </span>
        </p>
        {stage.is_public ? (
          <StatusPill tone="success">đang là câu trả lời bạn thấy</StatusPill>
        ) : (
          <StatusPill tone="neutral">chỉ để tham khảo</StatusPill>
        )}
      </div>
      {technical && stage.violation_codes.length > 0 && (
        <p className="mt-1 font-mono text-warning">{stage.violation_codes.join(', ')}</p>
      )}
      {stage.answer_markdown && (
        <>
          <button type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open} className="mt-1.5 text-link underline">
            {open ? 'Ẩn nội dung' : 'Xem nội dung bản này'}
          </button>
          {open && (
            <div className="mt-1.5 space-y-1">
              {!stage.is_public && (
                <p className="italic text-ink-tertiary">Bản nháp, chỉ để tham khảo — không phải câu trả lời đã công bố.</p>
              )}
              <p className="whitespace-pre-wrap text-ink-secondary">{stage.answer_markdown}</p>
            </div>
          )}
        </>
      )}
    </li>
  );
}

function ReviewRound({ review, technical }: { review: LineageReview; technical: boolean }) {
  const counts = Object.entries(review.counts).map(([verdict, n]) => `${n} ${VERDICT_LABEL[verdict] ?? verdict}`);
  return (
    <div className="space-y-1.5">
      <p className="font-semibold text-ink">
        Đối chiếu nguồn lần {review.round} ({review.phase === 'review_final' ? 'sau khi sửa' : 'lần đầu'})
        <span className="font-normal text-ink-secondary">
          {' · '}{review.status === 'completed' ? counts.join(', ') || 'chưa có đánh giá' : 'chưa hoàn tất'}
        </span>
      </p>
      {review.verdicts.length > 0 && (
        <ul className="space-y-1">
          {review.verdicts.map((verdict, index) => (
            <li key={`${verdict.claim_id}-${index}`} className="text-ink-secondary">
              <span className={verdict.verdict === 'supported' ? 'font-medium text-success' : 'font-medium text-warning'}>
                {VERDICT_LABEL[verdict.verdict ?? ''] ?? verdict.verdict}
              </span>
              {technical ? <span className="font-mono text-ink-tertiary"> · {verdict.key ?? verdict.claim_id}</span> : null}
              {verdict.reason ? `: ${verdict.reason}` : ''}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ClaimRow({ claim, technical }: { claim: LineageClaim; technical: boolean }) {
  const change = CHANGE[claim.change];
  return (
    <li className="rounded-xl border border-line bg-card p-3">
      <div className="mb-1.5 flex flex-wrap items-center gap-2">
        <StatusPill tone={change.tone}>{change.label}</StatusPill>
        {claim.verdicts.length > 0 && (
          <span className="text-ink-tertiary">
            {claim.verdicts.map((v) => `lần ${String(v.round)}: ${VERDICT_LABEL[String(v.verdict)] ?? String(v.verdict)}`).join('; ')}
          </span>
        )}
        {technical ? <span className="font-mono text-ink-tertiary">{claim.key}</span> : null}
      </div>
      {claim.change === 'reworded' ? (
        <>
          <p className="text-ink-tertiary line-through">{claim.candidate_text}</p>
          <p className="text-ink">{claim.final_text}</p>
        </>
      ) : (
        <p className={cn(claim.change === 'removed' ? 'text-ink-tertiary line-through' : 'text-ink')}>
          {claim.final_text ?? claim.candidate_text}
        </p>
      )}
    </li>
  );
}

function Missing({ lineage }: { lineage: RunStageLineage }) {
  const missing = lineage.missing;
  const lost = missing.lost_limitation_codes.map((code) => (LIMITATION_LABEL_VI as Record<string, string>)[code] ?? code);
  const lines = [
    missing.needed_inputs.length > 0 && `Cần thêm: ${missing.needed_inputs.join('; ')}`,
    missing.unresolved_claim_ids.length > 0 && `${missing.unresolved_claim_ids.length} ý vẫn chưa có nguồn hỗ trợ`,
    lost.length > 0 && `Bản sửa làm mất lưu ý: ${lost.join(' ')}`,
    missing.failure_code && `Dừng vì: ${ERROR_CODE_LABEL_VI[missing.failure_code] ?? 'một lỗi hệ thống'}`,
  ].filter(Boolean) as string[];
  if (lines.length === 0 && !missing.reason) return null;
  return (
    <Callout tone="warning" title="Bạn không nhận được gì, và vì sao" className="text-xs">
      {missing.reason && <p className="text-ink-secondary">{missing.reason}</p>}
      {lines.map((line) => <p key={line} className="text-ink-secondary">{line}</p>)}
    </Callout>
  );
}
