import { ShieldCheck } from 'lucide-react';
import { AnalysisPanel } from '../analysis/AnalysisPanel';
import { Callout } from '../../shared/ui/callout';
import { Skeleton } from '../../shared/ui/skeleton';
import { Link } from 'react-router';
import { errorMessageVi } from '../../shared/lib/labels';
import type { Endpoint, QuickPredictBatchResult, QuickPredictCompareResult, QuickPredictResult } from '../../shared/api/types';
import { ENDPOINT_DESCRIPTION, ENDPOINT_LABEL, ENDPOINT_ORDER, type EndpointCapability } from './endpointLabels';

const WHAT_YOU_WILL_SEE: Record<Endpoint, string> = {
  herg: 'Bạn sẽ thấy: xác suất blocker, ngưỡng đã dùng và mô hình đã chạy.',
  tox21: 'Bạn sẽ thấy: một dòng cho mỗi assay, kèm xác suất và nhãn.',
  clintox: 'Bạn sẽ thấy: xác suất, ngưỡng đã dùng và mô hình đã chạy.',
};

/** Before a run: what each measurement will report, so the pane is never blank. */
function ExpectedResults({ capabilities }: { capabilities: ReadonlyArray<EndpointCapability> }) {
  const known = capabilities.length
    ? [...capabilities].sort((a, b) => ENDPOINT_ORDER.indexOf(a.id) - ENDPOINT_ORDER.indexOf(b.id)).map((c) => ({ id: c.id, enabled: c.enabled }))
    : ENDPOINT_ORDER.map((id) => ({ id, enabled: id !== 'clintox' }));
  return (
    <div className="space-y-4">
      {known.map(({ id, enabled }) => (
        <article
          key={id}
          className={enabled ? 'space-y-1.5 rounded-xl border border-line bg-card p-5' : 'space-y-1.5 rounded-xl border border-line bg-surface-muted p-5'}
        >
          <div className="flex items-center justify-between gap-3">
            <h3 className={enabled ? 'text-15 font-semibold text-ink' : 'text-15 font-semibold text-ink-tertiary'}>{ENDPOINT_LABEL[id]}</h3>
            <span className="text-xs text-ink-tertiary">{enabled ? 'chờ phân tích' : 'không có ở triển khai này'}</span>
          </div>
          <p className={enabled ? 'text-13 text-ink' : 'text-13 text-ink-tertiary'}>{enabled ? `${ENDPOINT_DESCRIPTION[id]}.` : 'Phép đo chưa được phục vụ nên hiển thị là không có sẵn, không bị ẩn.'}</p>
          {enabled ? <p className="text-xs text-ink-secondary">{WHAT_YOU_WILL_SEE[id]}</p> : null}
        </article>
      ))}
      <Callout tone="brand" icon={<ShieldCheck />}>
        <strong className="font-semibold">Không có điểm tổng.</strong> Mỗi phép đo được báo riêng, kèm mô hình và ngưỡng đã dùng.{' '}
        <Link to="/docs/reading-results">Cách đọc kết quả</Link>
      </Callout>
    </div>
  );
}

/**
 * The right-hand pane: what will come back, then the per-measurement results. A
 * batch shows one block per molecule; a model comparison shows one block per
 * model. Nothing here combines measurements (ADR-0002).
 */
export function ResultsPane({
  capabilities,
  loading,
  error,
  result,
  batchResult,
  compareResult,
}: {
  capabilities: ReadonlyArray<EndpointCapability>;
  loading: boolean;
  error: { message: string } | null;
  result: QuickPredictResult | null;
  batchResult: QuickPredictBatchResult | null;
  compareResult: QuickPredictCompareResult | null;
}) {
  return (
    <section aria-label="Kết quả" aria-live="polite" className="min-w-0 space-y-4">
      <h2 className="text-base font-semibold text-ink">Kết quả</h2>

      {error ? <Callout tone="danger" role="alert" title="Không phân tích được">{error.message}</Callout> : null}

      {loading ? (
        <div className="space-y-4" aria-busy="true" aria-label="Đang phân tích">
          <Skeleton className="h-52 rounded-xl" />
          <Skeleton className="h-72 rounded-xl" />
        </div>
      ) : compareResult ? (
        <div className="space-y-3">
          <p className="text-sm font-medium text-ink">So sánh theo mô hình</p>
          <div className="grid gap-4 2xl:grid-cols-2">
            {compareResult.comparisons.map((comparison) => (
              <div key={`${comparison.endpoint}:${comparison.model_id}`} className="min-w-0 space-y-2">
                <p className="font-mono text-xs text-link">{comparison.endpoint} · {comparison.model_id}</p>
                <AnalysisPanel analysis={comparison.result} hideMolecule />
              </div>
            ))}
          </div>
        </div>
      ) : batchResult ? (
        <div className="space-y-6">
          {batchResult.errors.length > 0 ? (
            <Callout tone="danger" title={`${batchResult.errors.length} phân tử lỗi`}>
              <ul className="space-y-0.5 font-mono text-xs">
                {batchResult.errors.map((entry) => (
                  <li key={entry.index}>
                    #{entry.index} {entry.input_smiles || '(rỗng)'} — {errorMessageVi(entry.error, entry.error)}
                  </li>
                ))}
              </ul>
            </Callout>
          ) : null}
          {batchResult.results.map((entry, index) => (
            <div key={`${entry.canonical_smiles}:${index}`} className="space-y-3">
              <p className="break-all font-mono text-xs text-ink-secondary">#{index + 1} · {entry.canonical_smiles}</p>
              <AnalysisPanel analysis={entry} hideMolecule />
            </div>
          ))}
        </div>
      ) : result ? (
        <AnalysisPanel analysis={result} hideMolecule />
      ) : (
        <ExpectedResults capabilities={capabilities} />
      )}
    </section>
  );
}
