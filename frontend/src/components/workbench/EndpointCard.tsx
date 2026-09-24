import type { HergSection, ClintoxSection } from '../../lib/api/types';

function ProbabilityBar({ value, threshold }: { value: number; threshold: number }) {
  const pct = Math.min(Math.max(value, 0), 1) * 100;
  const thresholdPct = Math.min(Math.max(threshold, 0), 1) * 100;
  return (
    <div className="relative h-1.5 w-full rounded-full" style={{ backgroundColor: 'var(--border)' }}>
      <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: 'var(--accent-blue)' }} />
      <div
        className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2"
        style={{ left: `${thresholdPct}%`, backgroundColor: 'var(--text-faint)' }}
        title={`ngưỡng ${threshold}`}
      />
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span style={{ color: 'var(--text-faint)' }}>{label}</span>
      <span className="max-w-[60%] truncate text-right font-mono" style={{ color: 'var(--text-muted)' }} title={value}>
        {value}
      </span>
    </div>
  );
}

interface EndpointCardProps {
  title: string;
  section: HergSection | ClintoxSection;
  probability: number;
}

export function EndpointCard({ title, section, probability }: EndpointCardProps) {
  return (
    <div className="rounded-xl border p-4" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
      <div className="mb-1 flex items-center justify-between">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
          {title}
        </h3>
        <span
          className="rounded-full border px-2 py-0.5 text-xs font-medium"
          style={{ borderColor: 'var(--border)', color: 'var(--text)' }}
        >
          {section.label}
        </span>
      </div>
      <p className="mb-3 text-xs" style={{ color: 'var(--text-faint)' }}>
        {section.measurement}
      </p>
      <p className="mb-2 font-mono text-2xl font-semibold" style={{ color: 'var(--text)' }}>
        {(probability * 100).toFixed(2)}%
      </p>
      <div className="mb-3">
        <ProbabilityBar value={probability} threshold={section.threshold} />
      </div>
      <div className="space-y-1.5 border-t pt-3" style={{ borderColor: 'var(--border)' }}>
        <Row label="ngưỡng" value={String(section.threshold)} />
        <Row label="nguồn ngưỡng" value={section.threshold_source} />
        <Row label="model" value={section.model_id} />
      </div>
    </div>
  );
}
