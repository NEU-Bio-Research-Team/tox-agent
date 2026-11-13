import { Link } from 'react-router';
import type { Claim } from '../../../shared/api/types';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '../../../shared/ui/hover-card';
import { getDeveloperModeEnabled } from '../../../shared/lib/preferences';
import { artifactPath } from '../hooks/useArtifactSelection';
import { describeLocator } from '../lib/describeLocator';

/** N-3: citation lives at the claim, not the paragraph. A number in the answer
 * opens a card that says where it came from and links to the original result or
 * passage. The technical fields (field path, transform, ids) are shown only to
 * someone who turned on Run Details in Settings. */
export function ClaimChip({ claim, sessionId, index }: { claim: Claim; sessionId: string; index: number }) {
  const technical = getDeveloperModeEnabled();
  return (
    <HoverCard openDelay={150}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          className="mx-0.5 inline-flex items-center gap-0.5 rounded border-b border-dashed border-brand px-0.5 font-mono text-[0.95em] font-medium text-link"
        >
          {claim.rendered_value}
          <sup className="text-[0.65em]">{index + 1}</sup>
        </button>
      </HoverCardTrigger>
      <HoverCardContent className="w-80 space-y-2 bg-surface text-xs">
        <p className="text-sm font-semibold text-ink">Con số này đến từ đâu</p>
        {claim.text ? <p className="text-13 leading-snug text-ink-secondary">{claim.text}</p> : null}

        <div className="space-y-1">
          {claim.observation_id && (
            <Link
              to={artifactPath(sessionId, { kind: 'observation', entityId: claim.observation_id, fieldPath: claim.field_path })}
              className="block font-medium text-link underline"
            >
              Xem kết quả gốc của mô hình →
            </Link>
          )}
          {claim.citation_ids.map((evidenceId, i) => (
            <Link key={evidenceId} to={`/s/${sessionId}/evidence/${evidenceId}`} className="block font-medium text-link underline">
              Xem nguồn tham khảo {claim.citation_ids.length > 1 ? i + 1 : ''} →
            </Link>
          ))}
          {(claim.evidence_spans ?? []).map((span) => (
            <Link
              key={`${span.evidence_id}:${span.locator}`}
              to={artifactPath(sessionId, { kind: 'evidence', entityId: span.evidence_id, locator: span.locator })}
              className="block font-medium text-link underline"
            >
              Xem đoạn trích · {describeLocator(span.locator)} →
            </Link>
          ))}
        </div>

        <p className="border-t border-line pt-2 text-success">✓ Con số này đã được đối chiếu với nguồn của nó</p>

        {technical ? (
          <dl className="space-y-1 border-t border-line pt-2">
            <Row label="giá trị hiển thị" value={claim.rendered_value ?? '—'} />
            <Row label="loại · phép biến đổi" value={`${claim.kind} · ${claim.transform}`} />
            {claim.source_value !== undefined && <Row label="giá trị nguồn" value={String(claim.source_value)} />}
            {claim.field_path && <Row label="field_path" value={claim.field_path} wrap />}
            {claim.observation_id && <Row label="observation" value={claim.observation_id} wrap />}
          </dl>
        ) : null}
      </HoverCardContent>
    </HoverCard>
  );
}

function Row({ label, value, wrap }: { label: string; value: string; wrap?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className="text-ink-tertiary">{label}</dt>
      <dd className={`text-right font-mono text-ink-secondary ${wrap ? 'break-all' : 'truncate'}`}>{value}</dd>
    </div>
  );
}
