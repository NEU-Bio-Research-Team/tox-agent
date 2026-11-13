import { useState } from 'react';
import { ChevronDown, Sparkles } from 'lucide-react';
import { explainPrediction } from '../../shared/api/endpoints';
import type { AtomAttribution } from '../../shared/api/types';
import { ApiError } from '../../shared/api/types';
import { LIMITATION_LABEL_VI, errorMessageVi } from '../../shared/lib/labels';
import { Button } from '../../shared/ui/button';
import { AtomHighlightDepiction } from './AtomHighlightDepiction';
import { NativeSelect } from '../../shared/ui/native-select';

type LoadState =
  | { phase: 'idle' }
  | { phase: 'loading' }
  | { phase: 'done'; data: AtomAttribution }
  | { phase: 'error'; message: string };

/**
 * Atom-level XAI for one served endpoint. Stateless — it calls
 * `POST /v1/predict/explain` on demand, so it works identically on the
 * Quick Predict page and inside the session AnalysisPanel.
 *
 * The magnitude ramp is single-hue (D-XAI-7); the legend and the
 * `attribution_not_causality` limitation are shown at content level, never
 * hidden, because the visual must not be read as a mechanism.
 */
export function ExplainPanel({
  canonicalSmiles,
  endpoint,
  tox21Tasks,
  modelId,
}: {
  canonicalSmiles: string;
  endpoint: 'herg' | 'tox21';
  tox21Tasks?: string[];
  /** The release that produced the result on screen. An explanation of a
   * different release would describe a prediction nobody is looking at. */
  modelId?: string;
}) {
  const [open, setOpen] = useState(false);
  const [task, setTask] = useState<string>(tox21Tasks?.[0] ?? '');
  const [state, setState] = useState<LoadState>({ phase: 'idle' });

  const needsTask = endpoint === 'tox21';
  const run = async () => {
    setState({ phase: 'loading' });
    try {
      const data = await explainPrediction({
        smiles: canonicalSmiles,
        endpoint,
        task: needsTask ? task : undefined,
        model_id: modelId,
      });
      setState({ phase: 'done', data });
    } catch (err) {
      const message =
        err instanceof ApiError ? errorMessageVi(err.code, err.message) : 'Không giải thích được.';
      setState({ phase: 'error', message });
    }
  };

  const label = needsTask ? `Giải thích (XAI) · tox21` : `Giải thích (XAI) · ${endpoint}`;

  return (
    <section className="-mt-1 rounded-xl border border-line bg-canvas-subtle">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left"
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-ink">
          <Sparkles className="h-3.5 w-3.5 text-link" />
          {label}
        </span>
        <ChevronDown
          className="h-4 w-4 transition-transform"
          style={{ color: 'var(--ink-tertiary)', transform: open ? 'rotate(180deg)' : undefined }}
        />
      </button>

      {open && (
        <div className="space-y-3 border-t p-4">
          <div className="flex flex-wrap items-center gap-2">
            {needsTask && (
              <NativeSelect
                value={task}
                onChange={(e) => setTask(e.target.value)}
                aria-label="Chọn assay Tox21"
                className="h-8 w-auto text-xs"
              >
                {(tox21Tasks ?? []).map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </NativeSelect>
            )}
            <Button size="sm" onClick={run} disabled={state.phase === 'loading' || (needsTask && !task)}>
              {state.phase === 'loading' ? 'Đang giải thích…' : 'Giải thích'}
            </Button>
          </div>

          {state.phase === 'error' && (
            <p className="text-xs text-danger">
              {state.message}
            </p>
          )}

          {state.phase === 'done' && (
            <ExplainResult data={state.data} />
          )}

          <p className="text-xs text-ink-tertiary">
            Độ lớn attribution gradient×embedding — không phải cơ chế, không phải quan hệ nhân quả.
          </p>
        </div>
      )}
    </section>
  );
}

function ExplainResult({ data }: { data: AtomAttribution }) {
  if (data.status === 'failed') {
    return (
      <p className="text-xs text-danger">
        Predictor báo attribution failed; không có highlight nào được hiển thị.
      </p>
    );
  }

  const unmappedPct =
    data.unmapped_importance != null ? (data.unmapped_importance * 100).toFixed(0) : null;

  return (
    <div className="space-y-3">
      {data.status === 'partial' && (
        <p className="text-xs text-warning">
          Kết quả best-effort — attribution vượt ngân sách thời gian; không coi đây là danh sách đóng góp hoàn chỉnh.
        </p>
      )}

      <AtomHighlightDepiction
        smiles={data.canonical_smiles ?? data.input_smiles}
        atomOrderVersion={data.atom_order_version}
        atoms={data.atoms}
        bonds={data.bonds}
        depictionSvg={data.depiction_svg}
      />

      {unmappedPct != null && (
        <p className="text-xs text-ink-secondary">
          {unmappedPct}% trọng số nằm ở liên kết/tô-pô, không gán được cho nguyên tử.
        </p>
      )}

      {data.bonds?.some((bond) => bond.source === 'adjacent_atom_derived') && (
        <p className="text-xs text-ink-secondary">
          Màu của liên kết không có token trực tiếp được suy ra từ hai nguyên tử kề; đây là hỗ trợ đọc hình, không phải attribution trực tiếp.
        </p>
      )}

      <details className="text-xs">
        <summary className="cursor-pointer text-ink-tertiary">
          Xem attribution theo token ({data.tokens.length})
        </summary>
        <ul className="mt-1 space-y-0.5 font-mono text-ink-secondary">
          {data.tokens.map((token, index) => (
            <li key={`${token.token}:${index}`} className="flex justify-between gap-3">
              <span className="break-all">{token.token}</span>
              <span>{token.importance.toPrecision(4)}</span>
            </li>
          ))}
        </ul>
      </details>

      <p
        className="rounded-md px-2 py-1.5 text-xs"
        style={{ backgroundColor: 'var(--surface-muted)', color: 'var(--ink-secondary)' }}
      >
        {LIMITATION_LABEL_VI.attribution_not_causality}
      </p>
    </div>
  );
}
