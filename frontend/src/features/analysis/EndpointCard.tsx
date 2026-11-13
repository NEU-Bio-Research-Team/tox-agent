import type { HergSection, ClintoxSection } from '../../shared/api/types';
import { StatusPill } from '../../shared/ui/status-pill';
import { labelTone } from './labelTone';

/** The probability against its threshold. The tick is the threshold the label was
 * decided with; its caption sits above it so the two cannot be confused. */
function ProbabilityBar({ value, threshold }: { value: number; threshold: number }) {
  const pct = Math.min(Math.max(value, 0), 1) * 100;
  const thresholdPct = Math.min(Math.max(threshold, 0), 1) * 100;
  return (
    <div className="relative pt-5">
      <span
        className="absolute top-0 -translate-x-1/2 whitespace-nowrap font-mono text-2xs text-ink-secondary"
        style={{ left: `${Math.min(Math.max(thresholdPct, 8), 92)}%` }}
      >
        ngưỡng {threshold}
      </span>
      <div
        role="img"
        aria-label={`Xác suất ${pct.toFixed(0)} phần trăm, ngưỡng ${thresholdPct.toFixed(0)} phần trăm`}
        className="relative h-2.5 w-full rounded-full bg-line"
      >
        <div className="h-full rounded-full bg-brand" style={{ width: `${pct}%` }} />
        <div className="absolute -top-[5px] h-5 w-0.5 rounded-sm bg-ink" style={{ left: `${thresholdPct}%` }} title={`ngưỡng ${threshold}`} />
      </div>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-2xs font-semibold uppercase tracking-wider text-ink-tertiary">{label}</dt>
      <dd className="mt-1 truncate font-mono text-13 font-medium text-ink" title={value}>{value}</dd>
    </div>
  );
}

interface EndpointCardProps {
  title: string;
  section: HergSection | ClintoxSection;
  probability: number;
  /** What the percentage is the probability of ("xác suất blocker"). */
  probabilityLabel?: string;
}

/** One measurement, on its own: the value, the threshold it was read against, the
 * model that produced it. Per ADR-0002 nothing here is combined with another card. */
export function EndpointCard({ title, section, probability, probabilityLabel = 'xác suất' }: EndpointCardProps) {
  return (
    <article className="space-y-4 rounded-xl border border-line bg-card p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-ink">{title}</h3>
          <p className="mt-0.5 text-13 text-ink-secondary">{section.measurement}</p>
        </div>
        <StatusPill tone={labelTone(section.label)} className="shrink-0 text-xs">{section.label}</StatusPill>
      </div>

      <div className="space-y-1">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="text-4xl font-bold leading-none tracking-tight text-ink tabular-nums">{(probability * 100).toFixed(2)}%</span>
          <span className="text-13 text-ink-secondary">{probabilityLabel}</span>
        </div>
        <ProbabilityBar value={probability} threshold={section.threshold} />
      </div>

      <dl className="grid grid-cols-3 gap-3 border-t border-line pt-4">
        <Meta label="Ngưỡng" value={String(section.threshold)} />
        <Meta label="Nguồn ngưỡng" value={section.threshold_source} />
        <Meta label="Mô hình" value={section.model_id} />
      </dl>
    </article>
  );
}
