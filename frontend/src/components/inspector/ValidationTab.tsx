import { useQuery } from '@tanstack/react-query';
import { listAllEventsForRun } from '../../lib/api/endpoints';
import type { Violation } from '../../lib/api/types';
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
    return <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Đang tải…</p>;
  }
  if (query.isError || !query.data) {
    return <p className="text-xs" style={{ color: 'var(--accent-red)' }}>Không tải được lịch sử kiểm định.</p>;
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
      <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
        Không có lượt sửa nào — model qua validator ngay lần đầu, hoặc run này không phải report_qa/evidence_research.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {rows.map((row) => (
        <div key={row.generation}>
          <p className="mb-2 text-xs font-semibold" style={{ color: 'var(--text)' }}>
            Candidate {row.generation} —{' '}
            {row.accepted ? (
              <span style={{ color: row.isFallback ? 'var(--accent-yellow)' : 'var(--accent-green)' }}>
                {row.isFallback ? 'được nhận (dự phòng)' : 'được nhận'}
              </span>
            ) : (
              <span style={{ color: 'var(--accent-red)' }}>bị bác ({row.violations.length} vi phạm)</span>
            )}
          </p>
          {row.violations.length > 0 && (
            <details className="rounded-lg border p-2.5" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface-alt)' }}>
              <summary className="cursor-pointer text-xs font-medium" style={{ color: 'var(--accent-blue)' }}>
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
