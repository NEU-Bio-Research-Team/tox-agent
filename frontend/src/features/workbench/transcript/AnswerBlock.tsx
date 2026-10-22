import { Link } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { getAnswer, getRunResult } from '../../../shared/api/endpoints';
import { AnswerRenderer } from '../answer/AnswerRenderer';
import { ReleaseBanner } from '../answer/ReleaseBanner';
import { ApiError } from '../../../shared/api/types';

/** #47: trạng thái phát hành của một run, đọc từ run chứ không suy từ văn bản.
 * Dùng cho cả câu trả lời đã công bố và kết quả bị giữ lại (không có answer). */
export function RunResultBanner({ sessionId, runId }: { sessionId: string; runId: string }) {
  const result = useRunResult(sessionId, runId);
  return (
    <div className="space-y-1">
      <ReleaseBanner result={result.data} />
      {result.data?.recorded && result.data.disposition !== 'publish' && (
        <Link
          to={`/s/${sessionId}/runs/${runId}?tab=release`}
          className="text-xs underline"
          style={{ color: 'var(--accent-blue)' }}
        >
          xem các giai đoạn và lý do trong audit →
        </Link>
      )}
    </div>
  );
}

function useRunResult(sessionId: string, runId: string | null | undefined) {
  return useQuery({
    queryKey: ['run-result', sessionId, runId],
    queryFn: () => getRunResult(sessionId, runId as string),
    enabled: Boolean(runId),
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 2,
  });
}

export function AnswerBlock({
  sessionId,
  answerId,
  runId,
}: {
  sessionId: string;
  answerId: string;
  runId?: string | null;
}) {
  const query = useQuery({
    queryKey: ['answer', sessionId, answerId],
    queryFn: () => getAnswer(sessionId, answerId),
  });
  // Cùng một câu trả lời có thể là bản đã review, bản đã sửa rồi review, hoặc
  // chỉ là dữ kiện của một kết quả bị giữ lại.
  const result = useRunResult(sessionId, runId);

  if (query.isLoading) {
    return (
      <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
        Đang tải đáp án…
      </p>
    );
  }
  if (query.isError || !query.data) {
    return (
      <p className="text-sm" style={{ color: 'var(--accent-red)' }}>
        Không tải được đáp án {answerId}.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <ReleaseBanner result={result.data} />
      <AnswerRenderer answer={query.data} sessionId={sessionId} />
    </div>
  );
}
