import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router';
import { listAttributions } from '../../lib/api/endpoints';
import { LIMITATION_LABEL_VI } from '../../lib/labels';

export function AttributionPanel({ sessionId, analysisId }: { sessionId: string; analysisId: string }) {
  const query = useQuery({
    queryKey: ['attributions', sessionId, analysisId],
    queryFn: () => listAttributions(sessionId, analysisId),
  });

  if (query.isLoading) return null;

  if (query.isError) {
    return (
      <section className="rounded-xl border p-4 text-sm" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)', color: 'var(--accent-yellow)' }}>
        Không tải được attribution. API attribution có thể chưa được deploy cùng phiên bản với giao diện.
      </section>
    );
  }

  if (!query.data || query.data.attributions.length === 0) return null;

  return (
    <section className="space-y-2 rounded-xl border p-4" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
      <div>
        <h2 className="text-sm font-semibold" style={{ color: 'var(--text)' }}>Attribution theo endpoint</h2>
        <p className="mt-1 text-xs" style={{ color: 'var(--text-muted)' }}>
          {LIMITATION_LABEL_VI.attribution_not_causality}
        </p>
      </div>
      {query.data.attributions.map((attribution) => (
        <div key={attribution.observation_id} className="rounded-lg p-3 text-xs" style={{ backgroundColor: 'var(--surface-alt)' }}>
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-2 gap-y-1">
            <p className="font-medium" style={{ color: 'var(--text)' }}>
              {attribution.endpoint}{attribution.task ? ` · ${attribution.task}` : ''}
            </p>
            <span style={{ color: 'var(--text-faint)' }}>
              {attribution.status}{attribution.method ? ` · ${attribution.method}` : ''}
            </span>
          </div>
          {attribution.status === 'partial' && (
            <p className="mb-2" style={{ color: 'var(--accent-yellow)' }}>
              Predictor báo attribution partial; không coi đây là danh sách đóng góp hoàn chỉnh.
            </p>
          )}
          <ul className="space-y-1 font-mono" style={{ color: 'var(--text-muted)' }}>
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
            style={{ color: 'var(--accent-blue)' }}
          >
            Mở observation attribution →
          </Link>
        </div>
      ))}
    </section>
  );
}
