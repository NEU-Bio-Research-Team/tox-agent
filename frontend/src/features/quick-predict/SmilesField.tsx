import { CircleAlert, CircleCheck, Info, X } from 'lucide-react';
import { Label } from '../../shared/ui/label';
import { Segmented } from '../../shared/ui/segmented';
import { Textarea } from '../../shared/ui/textarea';
import { cn } from '../../shared/ui/utils';

export const EXAMPLE_MOLECULES = [
  { name: 'Caffeine', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C' },
  { name: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
  { name: 'Ethanol', smiles: 'CCO' },
  { name: 'Haloperidol', smiles: 'O=C(CCCN1CCC(c2ccc(Cl)cc2)(O)CC1)c1ccc(F)cc1' },
];

export type Validation = { tone: 'valid' | 'invalid' | 'note'; message: string } | null;

const VALIDATION_STYLE = {
  valid: { icon: CircleCheck, className: 'text-success' },
  invalid: { icon: CircleAlert, className: 'text-danger' },
  note: { icon: Info, className: 'text-warning' },
} as const;

/** The "SMILES" tab: one molecule or a list, with what is known about the string
 * (the page cannot validate chemistry; the predictor does) and the examples. */
export function SmilesField({
  smiles,
  onSmiles,
  batchMode,
  onBatchMode,
  batchText,
  onBatchText,
  batchCount,
  validation,
  invalid,
  loading,
  onSubmit,
}: {
  smiles: string;
  onSmiles: (value: string) => void;
  batchMode: boolean;
  onBatchMode: (batch: boolean) => void;
  batchText: string;
  onBatchText: (value: string) => void;
  batchCount: number;
  validation: Validation;
  invalid: boolean;
  loading: boolean;
  /** Ctrl/Cmd+Enter in either field. */
  onSubmit: () => void;
}) {
  const submitOnShortcut = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      onSubmit();
    }
  };
  const feedback = validation ? VALIDATION_STYLE[validation.tone] : null;

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-3">
          <Label htmlFor={batchMode ? 'qp-batch' : 'qp-smiles'} className="text-13 font-semibold">
            {batchMode ? (
              <>
                Danh sách SMILES <span className="font-normal text-ink-tertiary">· mỗi dòng một chuỗi · {batchCount} phân tử</span>
              </>
            ) : (
              'Chuỗi SMILES'
            )}
          </Label>
          <Segmented<'one' | 'many'>
            size="sm"
            label="Số lượng phân tử"
            value={batchMode ? 'many' : 'one'}
            onValueChange={(value) => onBatchMode(value === 'many')}
            options={[
              { value: 'one', label: 'Một chất', disabled: loading },
              { value: 'many', label: 'Danh sách', disabled: loading },
            ]}
          />
        </div>

        {batchMode ? (
          <Textarea
            id="qp-batch"
            aria-label="Danh sách SMILES"
            rows={6}
            className="min-h-[148px] rounded-lg border-line-strong px-3.5 py-3 font-mono text-sm"
            placeholder={'CCO\nCC(=O)Oc1ccccc1C(=O)O'}
            value={batchText}
            disabled={loading}
            spellCheck={false}
            autoCorrect="off"
            autoComplete="off"
            onChange={(event) => onBatchText(event.target.value)}
            onKeyDown={submitOnShortcut}
          />
        ) : (
          <div className="relative">
            <Textarea
              id="qp-smiles"
              rows={3}
              value={smiles}
              onChange={(event) => onSmiles(event.target.value)}
              onKeyDown={submitOnShortcut}
              placeholder="Dán hoặc gõ SMILES, ví dụ CC(=O)Oc1ccccc1C(=O)O"
              className={cn(
                'min-h-[84px] rounded-lg px-3.5 py-3 pr-9 font-mono text-sm',
                invalid ? 'border-danger' : validation?.tone === 'valid' ? 'border-brand-border' : 'border-line-strong',
              )}
              aria-invalid={invalid ? true : undefined}
              disabled={loading}
              spellCheck={false}
              autoCorrect="off"
              autoComplete="off"
            />
            {smiles && !loading ? (
              <button
                type="button"
                onClick={() => onSmiles('')}
                aria-label="Xoá SMILES"
                className="absolute right-2.5 top-2.5 rounded-md p-1 text-ink-tertiary transition-colors hover:bg-surface-muted hover:text-ink"
              >
                <X className="size-4" />
              </button>
            ) : null}
          </div>
        )}

        {feedback && validation ? (
          <p className={cn('flex items-center gap-1.5 text-xs', feedback.className)} role={validation.tone === 'invalid' ? 'alert' : 'status'}>
            <feedback.icon className="size-3.5 shrink-0" aria-hidden="true" />
            {validation.message}
          </p>
        ) : !batchMode ? (
          <p className="text-xs text-ink-tertiary">Chưa có SMILES</p>
        ) : null}
      </div>

      {!batchMode ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-0.5 text-xs text-ink-tertiary">Ví dụ</span>
          {EXAMPLE_MOLECULES.map((molecule) => (
            <button
              type="button"
              key={molecule.name}
              onClick={() => onSmiles(molecule.smiles)}
              disabled={loading}
              className={cn(
                'h-7 rounded-full border px-3 text-xs transition-colors disabled:opacity-50',
                smiles.trim() === molecule.smiles
                  ? 'border-brand-border bg-brand-soft font-semibold text-brand-ink'
                  : 'border-line bg-card text-ink hover:bg-surface-muted',
              )}
            >
              {molecule.name}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
