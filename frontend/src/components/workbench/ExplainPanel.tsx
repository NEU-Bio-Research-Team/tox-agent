import { useState } from 'react';
import { ChevronDown, Sparkles } from 'lucide-react';
import { explainPrediction } from '../../lib/api/endpoints';
import type { AtomAttribution } from '../../lib/api/types';
import { ApiError } from '../../lib/api/types';
import { LIMITATION_LABEL_VI, errorMessageVi } from '../../lib/labels';
import { Button } from '../ui/button';
import { AtomHighlightDepiction } from './AtomHighlightDepiction';

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
}: {
  canonicalSmiles: string;
  endpoint: 'herg' | 'tox21';
  tox21Tasks?: string[];
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
    <section
      className="rounded-xl border"
      style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 p-4 text-left"
      >
        <span className="flex items-center gap-2 text-sm font-semibold" style={{ color: 'var(--text)' }}>
          <Sparkles className="h-3.5 w-3.5" style={{ color: 'var(--accent-blue)' }} />
          {label}
        </span>
        <ChevronDown
          className="h-4 w-4 transition-transform"
          style={{ color: 'var(--text-faint)', transform: open ? 'rotate(180deg)' : undefined }}
        />
      </button>

      {open && (
        <div className="space-y-3 border-t p-4" style={{ borderColor: 'var(--border)' }}>
          <div className="flex flex-wrap items-center gap-2">
            {needsTask && (
              <select
                value={task}
                onChange={(e) => setTask(e.target.value)}
                aria-label="Chọn assay Tox21"
                className="rounded-md border px-2 py-1 text-xs"
                style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface-alt)', color: 'var(--text)' }}
              >
                {(tox21Tasks ?? []).map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            )}
            <Button size="sm" onClick={run} disabled={state.phase === 'loading' || (needsTask && !task)}>
              {state.phase === 'loading' ? 'Đang giải thích…' : 'Giải thích'}
            </Button>
          </div>

          {state.phase === 'error' && (
            <p className="text-xs" style={{ color: 'var(--accent-red)' }}>
              {state.message}
            </p>
          )}

          {state.phase === 'done' && (
            <ExplainResult data={state.data} />
          )}

          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
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
      <p className="text-xs" style={{ color: 'var(--accent-red)' }}>
        Predictor báo attribution failed; không có highlight nào được hiển thị.
      </p>
    );
  }

  const unmappedPct =
    data.unmapped_importance != null ? (data.unmapped_importance * 100).toFixed(0) : null;

  return (
    <div className="space-y-3">
      {data.status === 'partial' && (
        <p className="text-xs" style={{ color: 'var(--accent-yellow)' }}>
          Kết quả best-effort — attribution vượt ngân sách thời gian; không coi đây là danh sách đóng góp hoàn chỉnh.
        </p>
      )}

      <AtomHighlightDepiction
        smiles={data.canonical_smiles ?? data.input_smiles}
        atomOrderVersion={data.atom_order_version}
        atoms={data.atoms}
      />

      {unmappedPct != null && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {unmappedPct}% trọng số nằm ở liên kết/tô-pô, không gán được cho nguyên tử.
        </p>
      )}

      <details className="text-xs">
        <summary className="cursor-pointer" style={{ color: 'var(--text-faint)' }}>
          Xem attribution theo token ({data.tokens.length})
        </summary>
        <ul className="mt-1 space-y-0.5 font-mono" style={{ color: 'var(--text-muted)' }}>
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
        style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}
      >
        {LIMITATION_LABEL_VI.attribution_not_causality}
      </p>
    </div>
  );
}
