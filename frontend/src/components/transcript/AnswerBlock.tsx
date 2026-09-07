import { useQuery } from '@tanstack/react-query';
import { getAnswer } from '../../lib/api/endpoints';
import { AnswerRenderer } from '../answer/AnswerRenderer';

export function AnswerBlock({ sessionId, answerId }: { sessionId: string; answerId: string }) {
  const query = useQuery({
    queryKey: ['answer', sessionId, answerId],
    queryFn: () => getAnswer(sessionId, answerId),
  });

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

  return <AnswerRenderer answer={query.data} sessionId={sessionId} />;
}
