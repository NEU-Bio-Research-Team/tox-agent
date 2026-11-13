import { CircleCheck, TriangleAlert } from 'lucide-react';
import type { Applicability } from '../../shared/api/types';
import { StatusPill } from '../../shared/ui/status-pill';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '../../shared/ui/tooltip';

/** SCI-07: applicability is a fixed rule (element_rules_v1), never a learned
 * out-of-distribution score. The caption is not optional decoration — it is
 * the thing that stops a reader from over-trusting "ok" as "in-distribution". */
export function ApplicabilityChip({ applicability }: { applicability: Applicability }) {
  const ok = applicability.status === 'ok';
  return (
    <div className="space-y-1.5">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="inline-flex">
              <StatusPill tone={ok ? 'success' : 'warning'} className="text-xs font-medium">
                {ok ? <CircleCheck className="size-3.5" /> : <TriangleAlert className="size-3.5" />}
                Applicability {applicability.status}
                <span className="font-mono text-2xs">· {applicability.method}</span>
              </StatusPill>
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs text-xs">
            {applicability.reasons.length > 0 ? applicability.reasons.join('; ') : 'Không có lý do bổ sung.'}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <p className="text-xs text-ink-tertiary">Đánh giá bằng luật cố định, KHÔNG phải out-of-distribution học được.</p>
    </div>
  );
}
