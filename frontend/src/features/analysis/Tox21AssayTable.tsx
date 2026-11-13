import type { Tox21Section } from '../../shared/api/types';
import { StatusPill } from '../../shared/ui/status-pill';
import { labelTone } from './labelTone';

/**
 * A mapping rendered as a table, never a count. SCI-05: the number of Tox21
 * assays active across chemically unrelated targets is not a severity, so
 * this component must never expose a derived "N/12 active" figure anywhere —
 * not as a summary line, not as a badge, not as a sort key default. Rows keep
 * the model's fixed task order; they are not ranked by probability.
 */
export function Tox21AssayTable({ section }: { section: Tox21Section }) {
  const tasks = Object.entries(section.assays);

  return (
    <article className="space-y-3 rounded-xl border border-line bg-card p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-ink">Tox21</h3>
          <p className="mt-0.5 text-13 text-ink-secondary">{section.measurement}</p>
        </div>
        <span className="shrink-0 font-mono text-2xs text-ink-secondary">{section.task_order_version}</span>
      </div>

      <div className="max-h-[26rem] overflow-y-auto">
        <table className="w-full border-collapse text-13">
          <caption className="sr-only">Kết quả từng assay Tox21, theo thứ tự cố định của mô hình</caption>
          <thead>
            <tr className="text-left text-2xs font-semibold uppercase tracking-wider text-ink-tertiary">
              <th scope="col" className="pb-2 pr-3 font-semibold">Assay</th>
              <th scope="col" className="w-[38%] pb-2 pr-3 font-semibold">Xác suất</th>
              <th scope="col" className="pb-2 text-right font-semibold">Nhãn</th>
            </tr>
          </thead>
          <tbody>
            {tasks.map(([task, assay]) => (
              <tr key={task} className="border-t border-line">
                <td className="py-2.5 pr-3 font-mono font-medium text-ink">{task}</td>
                <td className="py-2.5 pr-3">
                  <div className="flex items-center gap-2.5">
                    <div className="h-1.5 flex-1 rounded-full bg-line" aria-hidden="true">
                      <div className="h-full rounded-full bg-brand" style={{ width: `${Math.min(Math.max(assay.probability_activity, 0), 1) * 100}%` }} />
                    </div>
                    <span className="w-11 text-right font-mono text-xs text-ink-secondary tabular-nums">{assay.probability_activity.toFixed(3)}</span>
                  </div>
                </td>
                <td className="py-2.5 text-right">
                  <StatusPill tone={assay.active ? labelTone('active') : 'neutral'} className="font-medium">
                    {assay.active ? 'active' : 'inactive'}
                  </StatusPill>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="border-t border-line pt-3 text-xs text-ink-tertiary">
        model {section.model_id} · 12 assay độc lập, không có chỉ số tổng hợp
      </p>
    </article>
  );
}
