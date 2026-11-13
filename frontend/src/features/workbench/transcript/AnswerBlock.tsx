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
          className="text-xs font-medium text-link underline"
        >
          Xem vì sao →
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
  // chỉ là dữ kiện của một kết quả bị giữ lại. Một message đời cũ không có
  // result_ref: hỏi theo run của chính answer, để server trả "không biết đã
  // review hay chưa" thay vì không hiện gì (§10.3).
  const result = useRunResult(sessionId, runId ?? query.data?.run_id);

  if (query.isLoading) {
    return (
      <p className="text-sm text-ink-secondary">
        Đang tải câu trả lời…
      </p>
    );
  }
  if (query.isError || !query.data) {
    return (
      <p className="text-sm text-danger">
        Không tải được câu trả lời này.
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
