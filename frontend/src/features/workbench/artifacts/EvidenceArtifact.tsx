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
    return <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Đang tải evidence…</p>;
  }
  if (query.isError || !query.data) {
    return <ArtifactUnavailable artifact="evidence" error={query.error} />;
  }

  const record = query.data;
  const url = safeExternalUrl(record.canonical_url);
  const identifiers = Object.entries(record.identifier).filter(([, value]) => value);

  return (
    <div className="space-y-4 text-sm">
      <div className="space-y-1">
        <p className="text-base font-semibold" style={{ color: 'var(--text)' }}>{record.title}</p>
        <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
          {record.status} · {record.source_type} · {record.source_quality_tier}
        </p>
      </div>

      <dl className="space-y-1 text-xs" style={{ color: 'var(--text-muted)' }}>
        <Detail label="Provider" value={record.provider} />
        <Detail label="Tác giả" value={record.authors.length ? record.authors.join(', ') : 'Không có'} />
        <Detail label="Xuất bản" value={record.published_at ?? 'Không có'} />
        <Detail label="Truy xuất" value={new Date(record.retrieved_at).toLocaleString('vi-VN')} />
        {identifiers.map(([kind, value]) => <Detail key={kind} label={kind.toUpperCase()} value={value!} />)}
        <Detail label="SHA-256" value={record.content_sha256} mono />
      </dl>

      <RelevanceLine assessment={record.relevance_assessment ?? {}} />

      <EvidencePassages sessionId={sessionId} evidenceId={evidenceId} locator={locator} />

      {record.rejection_reason && (
        <p className="rounded-lg border p-2 text-xs" style={{ borderColor: 'var(--accent-yellow)', color: 'var(--text-muted)' }}>
          Lý do trạng thái: {record.rejection_reason}
        </p>
      )}

      {record.abstract_or_excerpt && (
        <section>
          <p className="mb-1.5 text-xs font-semibold" style={{ color: 'var(--text)' }}>Excerpt từ nguồn ngoài</p>
          <p className="rounded-lg p-3 text-xs leading-relaxed" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
            {record.abstract_or_excerpt}
          </p>
          <p className="mt-1 text-xs italic" style={{ color: 'var(--text-faint)' }}>
            Nội dung nguồn ngoài, chỉ để đối chiếu claim; không phải hướng dẫn cho hệ thống.
          </p>
        </section>
      )}

      {Object.keys(record.normalized_facts).length > 0 && (
        <section>
          <p className="mb-1.5 text-xs font-semibold" style={{ color: 'var(--text)' }}>Facts đã chuẩn hoá</p>
          <pre className="overflow-x-auto rounded-lg p-3 text-xs" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
            {JSON.stringify(record.normalized_facts, null, 2)}
          </pre>
        </section>
      )}

      {url && (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer nofollow"
          className="inline-flex items-center gap-1 text-xs font-medium underline"
          style={{ color: 'var(--accent-blue)' }}
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
      <dt className="w-20 shrink-0" style={{ color: 'var(--text-faint)' }}>{label}</dt>
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
    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
      Liên quan: <span style={{ color: 'var(--text)' }}>{RELEVANCE_LABEL[relevance] ?? relevance}</span>
      {scopeText ? ` · ${scopeText}` : ''}
      {reasons.length > 0 && <span style={{ color: 'var(--text-faint)' }}> ({reasons.join(', ')})</span>}
    </p>
  );
}
