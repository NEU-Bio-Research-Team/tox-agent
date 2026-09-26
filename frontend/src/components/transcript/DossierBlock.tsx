import { useQuery } from '@tanstack/react-query';
import { getRunDossier } from '../../lib/api/endpoints';
import type { HypothesisStatus } from '../../lib/api/types';

/**
 * The decision dossier under a decision-support answer (W9-05, RETHINK
 * §4.4 step 3). The answer is one view of the run's dossier; this block is the
 * compact other half: which hypotheses the turn weighed and where they stand,
 * what the case can and cannot say, and what would change it. The full record
 * is on the investigation board.
 */

const STATUS_LABEL: Record<HypothesisStatus, string> = {
  open: 'đang mở',
  supported: 'được ủng hộ',
  weakened: 'bị yếu đi',
  refuted: 'bị bác bỏ',
  unresolvable: 'chưa thể giải quyết',
};

export function DossierBlock({ sessionId, runId }: { sessionId: string; runId: string }) {
  const dossier = useQuery({
    queryKey: ['dossier', sessionId, runId],
    queryFn: () => getRunDossier(sessionId, runId),
    staleTime: Infinity, // a run's dossier is written once and never changes
  });
  if (dossier.isLoading || dossier.isError || !dossier.data) return null;
  const d = dossier.data;
  const empty = d.hypotheses.length === 0 && d.conclusion.can_say.length === 0
    && d.conclusion.cannot_say.length === 0;
  if (empty) return null;
  const blocking = d.open_uncertainties.filter((u) => u.severity === 'blocking').length;

  return (
    <section aria-label="Hồ sơ quyết định" className="mt-3 space-y-2 rounded-lg border p-3 text-xs"
      style={{ borderColor: 'var(--border)', color: 'var(--text)' }}>
      <p className="font-semibold uppercase tracking-wide" style={{ color: 'var(--text-faint)' }}>Hồ sơ quyết định</p>
      {d.hypotheses.length > 0 && (
        <ul className="space-y-1">
          {d.hypotheses.map((h) => (
            <li key={h.id}>
              <span className="font-medium">{h.id}</span> {h.statement}{' '}
              <span style={{ color: 'var(--text-muted)' }}>
                ({STATUS_LABEL[h.status] ?? h.status}; ủng hộ {h.evidence_for.length}, phản bác {h.evidence_against.length})
              </span>
            </li>
          ))}
        </ul>
      )}
      {d.conclusion.can_say.length > 0 && (
        <div>
          <p className="font-medium">Có thể nói</p>
          <ul className="list-disc pl-4">
            {d.conclusion.can_say.map((line, index) => (
              <li key={index}>{line.text} <span style={{ color: 'var(--text-faint)' }}>({line.sources.length} nguồn)</span></li>
            ))}
          </ul>
        </div>
      )}
      {d.conclusion.cannot_say.length > 0 && (
        <div>
          <p className="font-medium">Chưa thể nói</p>
          <ul className="list-disc pl-4">{d.conclusion.cannot_say.map((line, index) => <li key={index}>{line}</li>)}</ul>
        </div>
      )}
      {d.conclusion.what_would_change.length > 0 && (
        <div>
          <p className="font-medium">Điều gì sẽ thay đổi nhận định</p>
          <ul className="list-disc pl-4">{d.conclusion.what_would_change.map((line, index) => <li key={index}>{line}</li>)}</ul>
        </div>
      )}
      <p style={{ color: 'var(--text-faint)' }}>
        {d.open_uncertainties.length} điều chưa biết{blocking ? ` (${blocking} chặn kết luận)` : ''}
        {' · '}có bằng chứng độc lập trực tiếp: {d.coverage.with_independent_direct_evidence}/{d.coverage.hypotheses}
        {!d.data_scope.external_search ? ' · chỉ dữ liệu nội bộ' : ''}
        {' · '}xem đầy đủ ở tab Điều tra
      </p>
    </section>
  );
}
