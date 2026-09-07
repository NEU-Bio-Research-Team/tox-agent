import type { Tox21Section } from '../../lib/api/types';

/**
 * A mapping rendered as a table, never a count. SCI-05: the number of Tox21
 * assays active across chemically unrelated targets is not a severity, so
 * this component must never expose a derived "N/12 active" figure anywhere —
 * not as a summary line, not as a badge, not as a sort key default.
 */
export function Tox21AssayTable({ section }: { section: Tox21Section }) {
  const tasks = Object.entries(section.assays);

  return (
    <div className="rounded-xl border p-4" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
      <div className="mb-1 flex items-center justify-between">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
          Tox21
        </h3>
        <span className="text-xs" style={{ color: 'var(--text-faint)' }}>
          {section.task_order_version}
        </span>
      </div>
      <p className="mb-3 text-xs" style={{ color: 'var(--text-faint)' }}>
        {section.measurement}
      </p>
      <div className="max-h-72 space-y-1 overflow-y-auto pr-1">
        {tasks.map(([task, assay]) => (
          <div
            key={task}
            className="flex items-center justify-between rounded-md px-2 py-1.5 text-xs"
            style={{ backgroundColor: 'var(--surface-alt)' }}
          >
            <span className="font-mono" style={{ color: 'var(--text)' }}>
              {task}
            </span>
            <div className="flex items-center gap-2">
              <span className="font-mono" style={{ color: 'var(--text-muted)' }}>
                {assay.probability_activity.toFixed(3)}
              </span>
              <span
                className="rounded-full border px-1.5 py-0.5"
                style={{ borderColor: 'var(--border)', color: assay.active ? 'var(--text)' : 'var(--text-faint)' }}
              >
                {assay.active ? 'active' : 'inactive'}
              </span>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-3 border-t pt-2 text-xs" style={{ borderColor: 'var(--border)', color: 'var(--text-faint)' }}>
        model {section.model_id} · 12 assay độc lập, không có chỉ số tổng hợp
      </p>
    </div>
  );
}
