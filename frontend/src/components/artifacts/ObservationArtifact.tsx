import { Link } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { getObservation } from '../../lib/api/endpoints';
import { scrollToRun } from '../../lib/scrollToRun';
import { ArtifactUnavailable } from './ArtifactUnavailable';

/**
 * Intermediate tool output / claim source, as a panel viewer instead of the
 * former `ObservationDialog` modal — plan section 8.2.1: artifacts are
 * viewed in the right panel, not stacked dialogs on top of chat.
 */
export function ObservationArtifact({
  sessionId,
  observationId,
  fieldPath,
}: {
  sessionId: string;
  observationId: string;
  fieldPath?: string;
}) {
  const query = useQuery({
    queryKey: ['observation', sessionId, observationId],
    queryFn: () => getObservation(sessionId, observationId),
  });

  if (query.isLoading) {
    return <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Đang tải…</p>;
  }
  if (query.isError || !query.data) {
    return <ArtifactUnavailable artifact="observation" error={query.error} />;
  }

  const observation = query.data;

  return (
    <div className="space-y-4 text-sm">
      <div>
        <p className="font-mono text-sm" style={{ color: 'var(--text)' }}>
          {observation.observation_id}
        </p>
        <p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
          Đầu ra trung gian đã được ToxAgent chọn lọc cho model — không phải raw dump từ predictor.
        </p>
      </div>

      {fieldPath && (
        <div className="rounded-lg border p-3 text-xs" style={{ borderColor: 'var(--accent-blue)', backgroundColor: 'var(--accent-blue-muted)' }}>
          <p className="font-semibold" style={{ color: 'var(--text)' }}>Field path được claim trỏ tới</p>
          <code className="mt-1 block break-all font-mono" style={{ color: 'var(--accent-blue)' }}>{fieldPath}</code>
          <p className="mt-1" style={{ color: 'var(--text-muted)' }}>
            Đây là path chính xác validator đối chiếu trên canonical observation. Model projection bên dưới là bản đã giới hạn; raw payload chỉ hiện với auditor.
          </p>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs" style={{ color: 'var(--text-faint)' }}>
        <span>kind: {observation.kind}</span>
        <span>producer: {observation.producer}</span>
        <span>schema: {observation.schema_version}</span>
        <Link to={`/s/${sessionId}/runs/${observation.run_id}`} className="font-mono underline" style={{ color: 'var(--accent-blue)' }}>
          run {observation.run_id} →
        </Link>
        <button
          type="button"
          onClick={() => scrollToRun(observation.run_id)}
          className="font-medium underline"
          style={{ color: 'var(--accent-blue)' }}
        >
          về lượt chat tạo kết quả
        </button>
      </div>

      <div>
        <p className="mb-1.5 text-xs font-medium" style={{ color: 'var(--text)' }}>
          model_projection
        </p>
        <pre
          className="overflow-x-auto rounded-md p-3 text-xs"
          style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}
        >
          {JSON.stringify(observation.model_projection, null, 2)}
        </pre>
      </div>

      {observation.canonical_payload && (
        <div>
          <p className="mb-1.5 text-xs font-medium" style={{ color: 'var(--text)' }}>
            canonical_payload <span style={{ color: 'var(--text-faint)' }}>(quyền auditor)</span>
          </p>
          <pre
            className="overflow-x-auto rounded-md p-3 text-xs"
            style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}
          >
            {JSON.stringify(observation.canonical_payload, null, 2)}
          </pre>
        </div>
      )}

      {observation.required_limitations.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-medium" style={{ color: 'var(--text)' }}>
            required_limitations
          </p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{observation.required_limitations.join(', ')}</p>
        </div>
      )}

      <p className="border-t pt-2 font-mono text-xs" style={{ borderColor: 'var(--border)', color: 'var(--text-faint)' }}>
        sha256 {observation.content_sha256.slice(0, 16)}…
      </p>
    </div>
  );
}
