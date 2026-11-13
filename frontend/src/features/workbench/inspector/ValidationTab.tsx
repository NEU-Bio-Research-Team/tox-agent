import { useQuery } from '@tanstack/react-query';
import { listAllEventsForRun } from '../../../shared/api/endpoints';
import type { Violation } from '../../../shared/api/types';
import { ViolationList } from './ViolationList';

interface CandidateRow {
  generation: number;
  violations: Violation[];
  accepted: boolean;
  isFallback: boolean;
}

/** The durable source here is the outbox (P1-2's `events:list`), not the live
 * in-memory ticks from useSessionEvents — a run opened after the browser
 * missed the run entirely must show the exact same history as one watched
 * live. */
export function ValidationTab({ sessionId, runId }: { sessionId: string; runId: string }) {
  const query = useQuery({
    queryKey: ['run-events', sessionId, runId],
    queryFn: () => listAllEventsForRun(sessionId, runId),
  });

  if (query.isLoading) {
    return <p className="text-xs text-ink-secondary">Đang tải…</p>;
  }
  if (query.isError || !query.data) {
    return <p className="text-xs text-danger">Không tải được lịch sử kiểm định.</p>;
  }

  const candidates = new Map<number, CandidateRow>();
  for (const event of query.data) {
    if (event.type === 'answer.rejected') {
      const generation = Number(event.payload.candidate_generation ?? 0);
      candidates.set(generation, {
        generation,
        violations: (event.payload.violations as Violation[] | undefined) ?? [],
        accepted: false,
        isFallback: false,
      });
    }
    if (event.type === 'answer.accepted') {
      const generation = Number(event.payload.candidate_generation ?? 0);
      const existing = candidates.get(generation);
      candidates.set(generation, {
        generation,
        violations: existing?.violations ?? [],
        accepted: true,
        isFallback: Boolean(event.payload.is_fallback),
      });
    }
  }

  const rows = [...candidates.values()].sort((a, b) => a.generation - b.generation);

  if (rows.length === 0) {
    return (
      <p className="text-xs text-ink-tertiary">
        Không có lượt sửa nào — model qua validator ngay lần đầu, hoặc run này không phải report_qa/evidence_research.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {rows.map((row) => (
        <div key={row.generation}>
          <p className="mb-2 text-xs font-semibold text-ink">
            Candidate {row.generation} —{' '}
            {row.accepted ? (
              <span style={{ color: row.isFallback ? 'var(--warning)' : 'var(--success)' }}>
                {row.isFallback ? 'được nhận (dự phòng)' : 'được nhận'}
              </span>
            ) : (
              <span style={{ color: 'var(--danger)' }}>bị bác ({row.violations.length} vi phạm)</span>
            )}
          </p>
          {row.violations.length > 0 && (
            <details className="rounded-lg border p-2.5 bg-surface-muted">
              <summary className="cursor-pointer text-xs font-medium text-link">
                Chi tiết {row.violations.length} vi phạm
              </summary>
              <div className="mt-2">
                <ViolationList violations={row.violations} />
              </div>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}
