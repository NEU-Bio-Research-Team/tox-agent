import type { Applicability } from '../../lib/api/types';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '../ui/tooltip';

/** SCI-07: applicability is a fixed rule (element_rules_v1), never a learned
 * out-of-distribution score. The caption is not optional decoration — it is
 * the thing that stops a reader from over-trusting "ok" as "in-distribution". */
export function ApplicabilityChip({ applicability }: { applicability: Applicability }) {
  const ok = applicability.status === 'ok';
  return (
    <div className="rounded-xl border p-4" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
      <h3 className="mb-2 text-sm font-semibold" style={{ color: 'var(--text)' }}>
        Applicability
      </h3>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium"
              style={{
                borderColor: 'var(--border)',
                color: ok ? 'var(--text)' : 'var(--accent-yellow)',
              }}
            >
              {applicability.status} · {applicability.method}
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs text-xs">
            {applicability.reasons.length > 0 ? applicability.reasons.join('; ') : 'Không có lý do bổ sung.'}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <p className="mt-2 text-xs" style={{ color: 'var(--text-faint)' }}>
        Đánh giá bằng luật cố định, KHÔNG phải out-of-distribution học được.
      </p>
    </div>
  );
}
