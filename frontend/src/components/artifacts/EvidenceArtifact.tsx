import { useQuery } from '@tanstack/react-query';
import { ExternalLink } from 'lucide-react';
import { getEvidence } from '../../lib/api/endpoints';
import { ArtifactUnavailable } from './ArtifactUnavailable';

function safeExternalUrl(value: string | null): string | null {
  return value && /^https:\/\//i.test(value) ? value : null;
}

export function EvidenceArtifact({ sessionId, evidenceId }: { sessionId: string; evidenceId: string }) {
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
