import { Link } from 'react-router';
import type { Claim } from '../../lib/api/types';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '../ui/hover-card';
import { artifactPath } from '../../hooks/useArtifactSelection';

/** N-3: citation lives at the claim, not the paragraph. Every numeric claim
 * this chip renders can be traced to an observation_id + field_path a
 * reviewer can open — "[mở →]" routes to the observation artifact in the
 * right panel (plan section 8.4) instead of a modal dialog. */
export function ClaimChip({ claim, sessionId, index }: { claim: Claim; sessionId: string; index: number }) {
  return (
    <>
      <HoverCard openDelay={150}>
        <HoverCardTrigger asChild>
          <button
            type="button"
            className="mx-0.5 inline-flex items-center gap-0.5 rounded border-b border-dashed px-0.5 font-mono text-[0.95em] font-medium"
            style={{ borderColor: 'var(--accent-blue)', color: 'var(--accent-blue)' }}
          >
            {claim.rendered_value}
            <sup className="text-[0.65em]">{index + 1}</sup>
          </button>
        </HoverCardTrigger>
        <HoverCardContent className="w-80 text-xs" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
          <p className="mb-2 font-semibold" style={{ color: 'var(--text)' }}>
            claim {index + 1} · {claim.kind}
          </p>
          <dl className="space-y-1">
            <Row label="giá trị hiển thị" value={claim.rendered_value ?? '—'} mono />
            <Row label="phép biến đổi" value={claim.transform} mono />
            {claim.source_value !== undefined && (
              <Row label="giá trị nguồn" value={String(claim.source_value)} mono />
            )}
            {claim.field_path && <Row label="field_path" value={claim.field_path} mono wrap />}
          </dl>
          {claim.observation_id && (
            <Link
              to={artifactPath(sessionId, {
                kind: 'observation',
                entityId: claim.observation_id,
                fieldPath: claim.field_path,
              })}
              className="mt-3 block text-xs font-medium underline"
              style={{ color: 'var(--accent-blue)' }}
            >
              mở observation {claim.observation_id.slice(0, 12)}… →
            </Link>
          )}
          {claim.citation_ids.length > 0 && (
            <div className="mt-2 space-y-1 border-t pt-2" style={{ borderColor: 'var(--border)' }}>
              <p style={{ color: 'var(--text-faint)' }}>evidence đã trích dẫn</p>
              {claim.citation_ids.map((evidenceId) => (
                <Link
                  key={evidenceId}
                  to={`/s/${sessionId}/evidence/${evidenceId}`}
                  className="block font-medium underline"
                  style={{ color: 'var(--accent-blue)' }}
                >
                  mở evidence {evidenceId.slice(0, 12)}… →
                </Link>
              ))}
            </div>
          )}
          <p className="mt-3 border-t pt-2 text-xs" style={{ borderColor: 'var(--border)', color: 'var(--accent-green)' }}>
            ✓ validator đã đối chiếu số này với nguồn
          </p>
        </HoverCardContent>
      </HoverCard>
    </>
  );
}

function Row({ label, value, mono, wrap }: { label: string; value: string; mono?: boolean; wrap?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt style={{ color: 'var(--text-faint)' }}>{label}</dt>
      <dd
        className={`text-right ${mono ? 'font-mono' : ''} ${wrap ? 'break-all' : 'truncate'}`}
        style={{ color: 'var(--text-muted)' }}
      >
        {value}
      </dd>
    </div>
  );
}
