import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getRunLineage, type LineageClaim, type LineageReview, type LineageStage, type RunStageLineage } from '../../../shared/api/endpoints';

/**
 * Câu trả lời đã đi qua những giai đoạn nào (AG-UI audit, §10.2, §14.2): bản
 * nháp đầu, ứng viên qua validator cứng, phán quyết của reviewer từng vòng, bản
 * sửa, và chính sách phát hành làm gì với tất cả. Hiển thị theo từng claim vì
 * hai điều người đọc cần đánh giá không thấy được ở câu trả lời cuối: bản sửa đã
 * xóa hay thu hẹp claim nào, và run thiếu đầu vào hay limitation nào.
 *
 * Mọi thứ đọc từ những gì run đã ghi; không phán xét lại. Văn bản của ứng viên
 * là tư liệu audit, không phải câu trả lời — nhãn nói rõ điều đó.
 */

const PHASE_LABEL: Record<string, string> = {
  initial: 'Bản nháp đầu',
  hard_correction: 'Sau sửa cấu trúc (validator cứng)',
  semantic_repair: 'Sau sửa theo review',
};

const VERDICT_LABEL: Record<string, string> = {
  supported: 'được nguồn hỗ trợ',
  partially_supported: 'hỗ trợ một phần',
  not_supported: 'không được hỗ trợ',
  contradicted: 'bị nguồn phản bác',
  cannot_judge: 'không phán định được',
  no_verdict: 'không có phán quyết',
};

const CHANGE_LABEL: Record<LineageClaim['change'], string> = {
  kept: 'giữ nguyên',
  reworded: 'viết lại',
  removed: 'bị xóa',
  added: 'thêm mới',
};

const CHANGE_COLOR: Record<LineageClaim['change'], string> = {
  kept: 'var(--text-faint)',
  reworded: 'var(--accent-blue)',
  removed: 'var(--accent-red)',
  added: 'var(--accent-green)',
};

export function StageLineage({ sessionId, runId }: { sessionId: string; runId: string }) {
  const query = useQuery({
    queryKey: ['run-lineage', sessionId, runId],
    queryFn: () => getRunLineage(sessionId, runId),
  });
  if (query.isLoading) {
    return <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Đang tải các giai đoạn…</p>;
  }
  if (query.isError || !query.data) {
    return <p className="text-xs" style={{ color: 'var(--accent-red)' }}>Không tải được các giai đoạn của run.</p>;
  }
  return <LineageBody lineage={query.data} />;
}

export function LineageBody({ lineage }: { lineage: RunStageLineage }) {
  if (!lineage.recorded) {
    return (
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
        Run này chạy trước khi có release policy: không biết câu trả lời đã qua review hay chưa.
      </p>
    );
  }
  return (
    <section className="space-y-3 text-xs" aria-label="Các giai đoạn của câu trả lời">
      <p style={{ color: 'var(--text-muted)' }}>
        Chính sách <span className="font-mono">{lineage.release_policy}</span> · kết quả{' '}
        <span className="font-semibold" style={{ color: 'var(--text)' }}>{lineage.disposition ?? 'chưa ghi'}</span>
        {' · '}review {lineage.review_status ?? 'chưa ghi'}
        {lineage.execution_outcome && lineage.execution_outcome !== 'completed' ? ` · ${lineage.execution_outcome}` : ''}
      </p>

      <ol className="space-y-1.5" aria-label="Bản nháp theo thứ tự">
        {lineage.stages.map((stage) => <StageRow key={stage.index} stage={stage} />)}
        {lineage.stages.length === 0 && (
          <li style={{ color: 'var(--text-faint)' }}>Run không nộp bản nháp nào.</li>
        )}
      </ol>

      {lineage.reviews.map((review) => <ReviewRound key={review.round} review={review} />)}

      {lineage.claims.length > 0 && (
        <div className="space-y-1">
          <p className="font-semibold" style={{ color: 'var(--text)' }}>Từng claim, từ ứng viên đến bản người đọc nhận</p>
          <ul className="space-y-1" aria-label="Lineage của claim">
            {lineage.claims.map((claim) => <ClaimRow key={claim.key} claim={claim} />)}
          </ul>
          {lineage.repair?.removal_only === true && (
            <p style={{ color: 'var(--accent-yellow)' }}>
              Bản sửa chỉ xóa claim, không viết lại: độ chính xác có thể tăng nhưng thông tin bắt buộc có thể đã mất.
            </p>
          )}
        </div>
      )}

      <Missing lineage={lineage} />
    </section>
  );
}

function StageRow({ stage }: { stage: LineageStage }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-md p-2" style={{ backgroundColor: 'var(--surface-alt)' }}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p style={{ color: 'var(--text)' }}>
          <span className="font-medium">{PHASE_LABEL[stage.phase] ?? stage.phase}</span>
          {' · '}{stage.hard_valid ? 'qua validator cứng' : 'bị validator từ chối'}
          {' · '}{stage.claim_count} claim
        </p>
        {stage.is_public
          ? <span style={{ color: 'var(--accent-green)' }}>bản đang phát hành</span>
          : <span style={{ color: 'var(--text-faint)' }}>không phải câu trả lời</span>}
      </div>
      {stage.violation_codes.length > 0 && (
        <p style={{ color: 'var(--accent-yellow)' }}>Lỗi cấu trúc: {stage.violation_codes.join(', ')}</p>
      )}
      {stage.answer_markdown && (
        <>
          <button type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open}
            className="mt-1 underline" style={{ color: 'var(--accent-blue)' }}>
            {open ? 'Ẩn văn bản' : 'Xem văn bản của giai đoạn này'}
          </button>
          {open && (
            <div className="mt-1 space-y-1">
              {!stage.is_public && (
                <p className="italic" style={{ color: 'var(--text-faint)' }}>
                  Bản nháp, chỉ để audit — không phải câu trả lời đã phát hành.
                </p>
              )}
              <p className="whitespace-pre-wrap" style={{ color: 'var(--text-muted)' }}>{stage.answer_markdown}</p>
            </div>
          )}
        </>
      )}
    </li>
  );
}

function ReviewRound({ review }: { review: LineageReview }) {
  const counts = Object.entries(review.counts).map(([verdict, n]) => `${n} ${VERDICT_LABEL[verdict] ?? verdict}`);
  return (
    <div className="space-y-1">
      <p className="font-semibold" style={{ color: 'var(--text)' }}>
        Review vòng {review.round} ({review.phase === 'review_final' ? 'sau sửa' : 'lần đầu'})
        <span className="font-normal" style={{ color: 'var(--text-muted)' }}>
          {' · '}{review.status === 'completed' ? counts.join(', ') || 'không có phán quyết' : `${review.status}: ${review.reason ?? ''}`}
        </span>
      </p>
      {review.verdicts.length > 0 && (
        <ul className="space-y-0.5">
          {review.verdicts.map((verdict, index) => (
            <li key={`${verdict.claim_id}-${index}`} style={{ color: 'var(--text-muted)' }}>
              <span className="font-mono" style={{ color: 'var(--text-faint)' }}>{verdict.key ?? verdict.claim_id}</span>
              {' — '}
              <span style={{ color: verdict.verdict === 'supported' ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>
                {VERDICT_LABEL[verdict.verdict ?? ''] ?? verdict.verdict}
              </span>
              {verdict.reason ? `: ${verdict.reason}` : ''}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ClaimRow({ claim }: { claim: LineageClaim }) {
  return (
    <li className="rounded-md border p-2" style={{ borderColor: 'var(--border)' }}>
      <p>
        <span className="font-mono" style={{ color: 'var(--text-faint)' }}>{claim.key}</span>
        {' · '}
        <span className="font-medium" style={{ color: CHANGE_COLOR[claim.change] }}>{CHANGE_LABEL[claim.change]}</span>
        {claim.verdicts.length > 0 && (
          <span style={{ color: 'var(--text-faint)' }}>
            {' · '}{claim.verdicts.map((v) => `vòng ${String(v.round)}: ${VERDICT_LABEL[String(v.verdict)] ?? String(v.verdict)}`).join('; ')}
          </span>
        )}
      </p>
      {claim.change === 'reworded' ? (
        <>
          <p style={{ color: 'var(--text-faint)' }}><span className="line-through">{claim.candidate_text}</span></p>
          <p style={{ color: 'var(--text)' }}>{claim.final_text}</p>
        </>
      ) : (
        <p className={claim.change === 'removed' ? 'line-through' : ''} style={{ color: claim.change === 'removed' ? 'var(--text-faint)' : 'var(--text)' }}>
          {claim.final_text ?? claim.candidate_text}
        </p>
      )}
    </li>
  );
}

function Missing({ lineage }: { lineage: RunStageLineage }) {
  const missing = lineage.missing;
  const lines = [
    missing.needed_inputs.length > 0 && `Cần thêm đầu vào: ${missing.needed_inputs.join('; ')}`,
    missing.unresolved_claim_ids.length > 0 && `${missing.unresolved_claim_ids.length} claim vẫn bị review chặn`,
    missing.lost_limitation_codes.length > 0 && `Bản sửa làm mất limitation: ${missing.lost_limitation_codes.join(', ')}`,
    missing.failure_code && `Run dừng vì: ${missing.failure_code}`,
  ].filter(Boolean) as string[];
  if (lines.length === 0 && !missing.reason) return null;
  return (
    <div className="space-y-0.5 rounded-md border p-2" style={{ borderColor: 'var(--accent-yellow)' }}>
      <p className="font-semibold" style={{ color: 'var(--text)' }}>Người đọc không nhận được gì, và vì sao</p>
      {missing.reason && <p style={{ color: 'var(--text-muted)' }}>{missing.reason}</p>}
      {lines.map((line) => <p key={line} style={{ color: 'var(--text-muted)' }}>{line}</p>)}
    </div>
  );
}
