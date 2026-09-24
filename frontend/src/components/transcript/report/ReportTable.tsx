import type { ReportSourceClass, ReportTable as ReportTableData } from '../../../lib/api/types';

/** Every piece of report content says which of the four kinds of claim it is
 * (spec 3.4). Carrying that into the badge means a predictor number and a
 * published one are never presented as the same kind of statement. */
export const SOURCE_CLASS_LABEL: Record<ReportSourceClass, string> = {
  structure_fact: 'Cấu trúc',
  predictor_fact: 'Model',
  explanation_fact: 'Explainer',
  external_evidence: 'Tài liệu ngoài',
  agent_synthesis: 'Tổng hợp của agent',
  recommendation: 'Đề xuất',
};

export function SourceClassBadge({ sourceClass }: { sourceClass: ReportSourceClass }) {
  return (
    <span
      className="rounded-full border px-2 py-0.5 text-[11px]"
      style={{ color: 'var(--text-muted)', borderColor: 'var(--border-subtle)' }}
    >
      {SOURCE_CLASS_LABEL[sourceClass] ?? sourceClass}
    </span>
  );
}

/** A table the artifact carries as columns and already-rendered rows.
 *
 * `overflow-x-auto` on the wrapper rather than letting the table set the page
 * width: a report is read in a resizable side panel and on a phone, and a wide
 * predictor table must scroll inside its own box instead of making the whole
 * transcript scroll sideways.
 */
export function ReportTable({ table }: { table: ReportTableData }) {
  return (
    <div className="my-3 space-y-1">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-sm font-medium">{table.title}</p>
        <SourceClassBadge sourceClass={table.source_class} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] border-collapse text-sm">
          <thead>
            <tr className="text-left">
              {table.columns.map((column) => (
                <th
                  key={column}
                  className="border-b py-1.5 pr-3 font-medium"
                  style={{ borderColor: 'var(--border-subtle)' }}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="border-b py-1.5 pr-3 tabular-nums"
                    style={{ borderColor: 'var(--border-subtle)' }}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
