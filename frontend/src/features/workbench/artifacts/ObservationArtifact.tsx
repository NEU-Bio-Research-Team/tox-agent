import { Link } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { Activity } from 'lucide-react';
import { getObservation } from '../../../shared/api/endpoints';
import { LIMITATION_LABEL_VI } from '../../../shared/lib/labels';
import { Callout } from '../../../shared/ui/callout';
import { scrollToRun } from '../lib/scrollToRun';
import { ArtifactUnavailable } from './ArtifactUnavailable';

/** The value at a dotted path (`sections.herg.probability_blocker`), if it is a
 * plain number, string or boolean. Anything else is not a value to point at. */
function valueAt(root: unknown, path: string | undefined): string | null {
  if (!path) return null;
  let node: unknown = root;
  for (const key of path.replace(/\[(\d+)\]/g, '.$1').split('.').filter(Boolean)) {
    if (node === null || typeof node !== 'object') return null;
    node = (node as Record<string, unknown>)[key];
  }
  return typeof node === 'number' || typeof node === 'string' || typeof node === 'boolean' ? String(node) : null;
}

/**
 * "Kết quả gốc của mô hình": the output the answer was written from, as a panel
 * viewer instead of the former `ObservationDialog` modal — plan section 8.2.1:
 * artifacts are viewed in the right panel, not stacked dialogs on top of chat.
 * The value the answer quotes is pulled out and shown first; the full data, the
 * ids and the checksum are behind two closed disclosures.
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
    return <p className="text-sm text-ink-secondary">Đang tải…</p>;
  }
  if (query.isError || !query.data) {
    return <ArtifactUnavailable artifact="kết quả gốc" error={query.error} />;
  }

  const observation = query.data;
  const quoted = valueAt(observation.model_projection, fieldPath);

  return (
    <div className="space-y-4 text-sm">
      <header className="space-y-1.5">
        <h2 className="flex items-center gap-2 text-base font-semibold text-ink">
          <Activity className="size-4 text-brand-ink" aria-hidden="true" />
          Kết quả gốc của mô hình
        </h2>
        <p className="text-13 leading-snug text-ink-secondary">
          Đây là kết quả mà câu trả lời được viết từ đó. ToxAgent chỉ giữ lại phần cần thiết, không phải toàn bộ đầu ra của mô hình.
        </p>
        <button type="button" onClick={() => scrollToRun(observation.run_id)} className="text-xs font-medium text-link underline">
          Về tin nhắn tạo ra kết quả này
        </button>
      </header>

      {fieldPath ? (
        <div className="rounded-xl border border-brand-border bg-brand-soft p-4">
          <p className="text-xs font-semibold uppercase tracking-wider text-ink-tertiary">Con số được câu trả lời trích</p>
          {quoted !== null ? (
            <p className="mt-1 font-mono text-3xl font-bold tracking-tight text-ink tabular-nums">{quoted}</p>
          ) : (
            <p className="mt-1 text-13 text-ink-secondary">Xem trong dữ liệu đầy đủ bên dưới.</p>
          )}
          <p className="mt-2 text-xs text-ink-secondary">Hệ thống đã đối chiếu con số này với đúng vị trí đó trong kết quả gốc.</p>
        </div>
      ) : null}

      {observation.required_limitations.length > 0 && (
        <Callout tone="warning" title="Lưu ý khi đọc kết quả này" className="text-13">
          <ul className="space-y-1 text-ink-secondary">
            {observation.required_limitations.map((code) => (
              <li key={code}>• {(LIMITATION_LABEL_VI as Record<string, string>)[code] ?? code}</li>
            ))}
          </ul>
        </Callout>
      )}

      <details className="rounded-lg border border-line bg-canvas-subtle px-3 py-2 text-xs">
        <summary className="cursor-pointer font-medium text-ink-secondary">Xem dữ liệu đầy đủ</summary>
        <pre className="mt-2 overflow-x-auto rounded-md bg-surface-muted p-3 text-ink-secondary">
          {JSON.stringify(observation.model_projection, null, 2)}
        </pre>
        {observation.canonical_payload && (
          <>
            <p className="mt-3 font-medium text-ink">Dữ liệu gốc đầy đủ (dành cho người kiểm toán)</p>
            <pre className="mt-1 overflow-x-auto rounded-md bg-surface-muted p-3 text-ink-secondary">
              {JSON.stringify(observation.canonical_payload, null, 2)}
            </pre>
          </>
        )}
      </details>

      <details className="rounded-lg border border-line bg-canvas-subtle px-3 py-2 text-xs">
        <summary className="cursor-pointer font-medium text-ink-secondary">Chi tiết kỹ thuật</summary>
        <dl className="mt-2 space-y-1.5 font-mono text-ink-secondary">
          <div><dt className="text-ink-tertiary">mã</dt><dd className="break-all">{observation.observation_id}</dd></div>
          {fieldPath ? <div><dt className="text-ink-tertiary">vị trí trích</dt><dd className="break-all">{fieldPath}</dd></div> : null}
          <div><dt className="text-ink-tertiary">loại · nguồn tạo · lược đồ</dt><dd>{observation.kind} · {observation.producer} · {observation.schema_version}</dd></div>
          <div>
            <dt className="text-ink-tertiary">lượt chạy</dt>
            <dd className="break-all"><Link to={`/s/${sessionId}/runs/${observation.run_id}`} className="underline">{observation.run_id}</Link></dd>
          </div>
          <div><dt className="text-ink-tertiary">sha256</dt><dd>{observation.content_sha256.slice(0, 16)}…</dd></div>
        </dl>
      </details>
    </div>
  );
}
