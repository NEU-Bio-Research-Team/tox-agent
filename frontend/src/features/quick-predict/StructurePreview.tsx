import { useState } from 'react';
import { Check, Copy, FlaskConical } from 'lucide-react';
import { MoleculeDepiction } from '../analysis/MoleculeDepiction';

/** The structure that will be analysed, drawn client-side. The string beneath it
 * is the one the request will carry. */
export function StructurePreview({ smiles, rendered }: { smiles: string; rendered: string }) {
  const [copied, setCopied] = useState(false);
  const ready = smiles !== '' && rendered === smiles;

  return (
    <div className="space-y-2">
      {smiles === '' ? (
        <div className="flex h-44 flex-col items-center justify-center gap-2 rounded-[14px] border-[1.5px] border-dashed border-line-strong bg-canvas-subtle px-6 text-center text-13 text-ink-tertiary">
          <FlaskConical className="size-6" aria-hidden="true" />
          <span>Cấu trúc sẽ hiện ở đây khi có SMILES</span>
        </div>
      ) : (
        <div className="flex h-44 items-center justify-center overflow-hidden rounded-[14px] border border-line bg-depiction">
          {ready ? <MoleculeDepiction smiles={rendered} size={156} /> : <span className="text-13 text-ink-tertiary">Đang vẽ cấu trúc…</span>}
        </div>
      )}
      {smiles !== '' ? (
        <div className="flex items-center justify-between gap-2">
          <span className="shrink-0 text-xs text-ink-tertiary">SMILES sẽ gửi</span>
          <span className="flex min-w-0 items-center gap-1 font-mono text-xs text-ink">
            <span className="truncate" title={smiles}>{smiles}</span>
            <button
              type="button"
              aria-label="Sao chép SMILES"
              className="flex size-6 shrink-0 items-center justify-center rounded-md text-ink-secondary hover:bg-surface-muted"
              onClick={() => {
                void navigator.clipboard?.writeText(smiles);
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1200);
              }}
            >
              {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
            </button>
          </span>
        </div>
      ) : null}
    </div>
  );
}
