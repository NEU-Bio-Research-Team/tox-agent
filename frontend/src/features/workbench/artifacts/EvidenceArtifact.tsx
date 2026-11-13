import { useQuery } from '@tanstack/react-query';
import { ExternalLink } from 'lucide-react';
import { getEvidence } from '../../../shared/api/endpoints';
import { ArtifactUnavailable } from './ArtifactUnavailable';
import { EvidencePassages } from './EvidencePassages';

const RELEVANCE_LABEL: Record<string, string> = {
  direct: 'trực tiếp về subject (đúng ligand và target)',
  contextual: 'bối cảnh (cùng ligand, target khác)',
  uncertain: 'chưa xác định được',
  irrelevant: 'không về subject này',
};

function safeExternalUrl(value: string | null): string | null {
  return value && /^https:\/\//i.test(value) ? value : null;
}

export function EvidenceArtifact({
  sessionId,
  evidenceId,
  locator,
}: {
  sessionId: string;
  evidenceId: string;
  /** The passage a claim quotes; the viewer opens on it. */
  locator?: string;
}) {
  const query = useQuery({
    queryKey: ['evidence', sessionId, evidenceId],
    queryFn: () => getEvidence(sessionId, evidenceId),
  });

  if (query.isLoading) {
    return <p className="text-sm text-ink-secondary">Đang tải evidence…</p>;
  }
  if (query.isError || !query.data) {
    return <ArtifactUnavailable artifact="nguồn tham khảo" error={query.error} />;
  }

  const record = query.data;
  const url = safeExternalUrl(record.canonical_url);
  const identifiers = Object.entries(record.identifier).filter(([, value]) => value);

  return (
    <div className="space-y-4 text-sm">
      <div className="space-y-1">
        <p className="text-base font-semibold text-ink">{record.title}</p>
        <p className="text-xs text-ink-tertiary">
          {record.provider}
          {record.published_at ? ` · ${record.published_at.slice(0, 4)}` : ''}
          {record.authors.length ? ` · ${record.authors.slice(0, 3).join(', ')}${record.authors.length > 3 ? ' và cộng sự' : ''}` : ''}
        </p>
      </div>

      <RelevanceLine assessment={record.relevance_assessment ?? {}} />

      <EvidencePassages sessionId={sessionId} evidenceId={evidenceId} locator={locator} />

      {record.rejection_reason && (
        <p className="rounded-lg border p-2 text-xs border-warning text-ink-secondary">
          Lý do: {record.rejection_reason}
        </p>
      )}

      {record.abstract_or_excerpt && (
        <section>
          <p className="mb-1.5 text-xs font-semibold text-ink">Trích đoạn từ nguồn</p>
          <p className="rounded-lg p-3 text-xs leading-relaxed bg-surface-muted text-ink-secondary">
            {record.abstract_or_excerpt}
          </p>
          <p className="mt-1 text-xs italic text-ink-tertiary">
            Nội dung của nguồn bên ngoài, chỉ để đối chiếu; ToxAgent không coi đó là chỉ dẫn.
          </p>
        </section>
      )}

      {Object.keys(record.normalized_facts).length > 0 && (
        <section>
          <p className="mb-1.5 text-xs font-semibold text-ink">Dữ kiện rút ra từ nguồn</p>
          <pre className="overflow-x-auto rounded-lg p-3 text-xs bg-surface-muted text-ink-secondary">
            {JSON.stringify(record.normalized_facts, null, 2)}
          </pre>
        </section>
      )}

      <details className="rounded-lg border border-line bg-canvas-subtle px-3 py-2 text-xs">
        <summary className="cursor-pointer font-medium text-ink-secondary">Chi tiết kỹ thuật</summary>
        <dl className="mt-2 space-y-1 text-ink-secondary">
          <Detail label="Trạng thái" value={`${record.status} · ${record.source_type} · ${record.source_quality_tier}`} />
          <Detail label="Truy xuất" value={new Date(record.retrieved_at).toLocaleString('vi-VN')} />
          {identifiers.map(([kind, value]) => <Detail key={kind} label={kind.toUpperCase()} value={value!} />)}
          <Detail label="SHA-256" value={record.content_sha256} mono />
        </dl>
      </details>

      {url && (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer nofollow"
          className="inline-flex items-center gap-1 text-xs font-medium underline"
          style={{ color: 'var(--link)' }}
        >
          Mở nguồn đã chuẩn hoá <ExternalLink className="h-3 w-3" />
        </a>
      )}
    </div>
  );
}

function Detail({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex gap-2">
      <dt className="w-20 shrink-0 text-ink-tertiary">{label}</dt>
      <dd className={`min-w-0 break-all ${mono ? 'font-mono' : ''}`}>{value}</dd>
    </div>
  );
}

/** Whether the record is about the subject it was searched for, by the
 * server's rule -- never inferred here. Absent when no assessment ran. */
function RelevanceLine({ assessment }: { assessment: Record<string, unknown> }) {
  const relevance = typeof assessment.relevance === 'string' ? assessment.relevance : null;
  if (!relevance) return null;
  const reasons = Array.isArray(assessment.reason_codes) ? assessment.reason_codes.map(String) : [];
  const scope = (assessment.assessed_scope ?? {}) as Record<string, unknown>;
  const scopeText = Object.entries(scope).map(([key, value]) => `${key}: ${String(value)}`).join(' · ');
  return (
    <p className="text-xs text-ink-secondary">
      Liên quan: <span style={{ color: 'var(--ink)' }}>{RELEVANCE_LABEL[relevance] ?? relevance}</span>
      {scopeText ? ` · ${scopeText}` : ''}
      {reasons.length > 0 && <span style={{ color: 'var(--ink-tertiary)' }}> ({reasons.join(', ')})</span>}
    </p>
  );
}
