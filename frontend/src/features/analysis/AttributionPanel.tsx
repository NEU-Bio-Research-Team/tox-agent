import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router';
import { listAttributions } from '../../shared/api/endpoints';
import { LIMITATION_LABEL_VI } from '../../shared/lib/labels';

export function AttributionPanel({ sessionId, analysisId }: { sessionId: string; analysisId: string }) {
  const query = useQuery({
    queryKey: ['attributions', sessionId, analysisId],
    queryFn: () => listAttributions(sessionId, analysisId),
  });

  if (query.isLoading) return null;

  if (query.isError) {
    return (
      <section className="rounded-xl border p-4 text-sm bg-surface text-warning">
        Không tải được attribution. API attribution có thể chưa được deploy cùng phiên bản với giao diện.
      </section>
    );
  }

  if (!query.data || query.data.attributions.length === 0) return null;

  return (
    <section className="space-y-2 rounded-xl border p-4 bg-surface">
      <div>
        <h2 className="text-sm font-semibold text-ink">Attribution theo endpoint</h2>
        <p className="mt-1 text-xs text-ink-secondary">
          {LIMITATION_LABEL_VI.attribution_not_causality}
        </p>
      </div>
      {query.data.attributions.map((attribution) => (
        <div key={attribution.observation_id} className="rounded-lg p-3 text-xs bg-surface-muted">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-2 gap-y-1">
            <p className="font-medium text-ink">
              {attribution.endpoint}{attribution.task ? ` · ${attribution.task}` : ''}
            </p>
            <span style={{ color: 'var(--ink-tertiary)' }}>
              {attribution.status}{attribution.method ? ` · ${attribution.method}` : ''}
            </span>
          </div>
          {attribution.status === 'partial' && (
            <p className="mb-2 text-warning">
              Predictor báo attribution partial; không coi đây là danh sách đóng góp hoàn chỉnh.
            </p>
          )}
          <ul className="space-y-1 font-mono text-ink-secondary">
            {attribution.top_tokens.map((token, index) => (
              <li key={`${token.token}:${index}`} className="flex justify-between gap-3">
                <span className="break-all">{token.token}</span>
                <span>{token.score.toPrecision(4)}</span>
              </li>
            ))}
          </ul>
          <Link
            to={`/s/${sessionId}/observations/${attribution.observation_id}`}
            className="mt-2 inline-block font-medium underline"
            style={{ color: 'var(--link)' }}
          >
            Mở observation attribution →
          </Link>
        </div>
      ))}
    </section>
  );
}
