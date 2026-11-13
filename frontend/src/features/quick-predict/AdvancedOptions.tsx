import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import type { Endpoint } from '../../shared/api/types';
import { Input } from '../../shared/ui/input';
import { Label } from '../../shared/ui/label';
import { NativeSelect } from '../../shared/ui/native-select';
import { cn } from '../../shared/ui/utils';
import { ENDPOINT_LABEL, modelLabel, type EndpointCapability } from './endpointLabels';

/** Model choice, model comparison and (for an expert) the hERG threshold. Closed
 * by default: the measurements themselves are on the page, these are not. */
export function AdvancedOptions({
  capabilities,
  selected,
  modelSelection,
  onModelSelection,
  compareMode,
  onCompareMode,
  compareSelection,
  onCompareSelection,
  batchActive,
  expertMode,
  thresholdHerg,
  onThresholdHerg,
  disabled,
  open,
  onOpenChange,
}: {
  capabilities: ReadonlyArray<EndpointCapability>;
  selected: ReadonlyArray<Endpoint>;
  modelSelection: Partial<Record<Endpoint, string>>;
  onModelSelection: (next: Partial<Record<Endpoint, string>>) => void;
  compareMode: boolean;
  onCompareMode: (next: boolean) => void;
  compareSelection: Partial<Record<Endpoint, string[]>>;
  onCompareSelection: (next: Partial<Record<Endpoint, string[]>>) => void;
  batchActive: boolean;
  expertMode: boolean;
  thresholdHerg: string;
  onThresholdHerg: (next: string) => void;
  disabled?: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [id] = useState(() => `adv-${Math.random().toString(36).slice(2, 8)}`);
  const active = capabilities.filter((endpoint) => endpoint.enabled && selected.includes(endpoint.id));
  const sharedModel = selected.length > 1 && new Set(selected.map((endpoint) => modelSelection[endpoint])).size === 1;

  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => onOpenChange(!open)}
        className="flex w-full items-center justify-between gap-3 text-left"
      >
        <span>
          <span className="block text-13 font-semibold text-ink">Tuỳ chọn nâng cao</span>
          <span className="mt-0.5 block text-xs text-ink-secondary">Ngưỡng hERG, chọn mô hình, so sánh mô hình</span>
        </span>
        <ChevronDown className={cn('size-4 shrink-0 text-ink-secondary transition-transform', open && 'rotate-180')} aria-hidden="true" />
      </button>

      {open ? (
        <div id={id} className="mt-4 space-y-4">
          <div className="space-y-2">
            {active.map((endpoint) => {
              const models = endpoint.models ?? [];
              const chosen = models.find((model) => model.model_id === modelSelection[endpoint.id]);
              return (
                <div key={endpoint.id} className="space-y-1.5 rounded-xl border border-line bg-canvas-subtle px-3.5 py-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-13 font-semibold text-ink">{ENDPOINT_LABEL[endpoint.id]}</span>
                    {!compareMode ? (
                      <div className="w-52">
                        <NativeSelect
                          aria-label={`Mô hình cho ${ENDPOINT_LABEL[endpoint.id]}`}
                          className="h-8 text-xs"
                          value={modelSelection[endpoint.id] ?? ''}
                          disabled={disabled}
                          onChange={(event) => onModelSelection({ ...modelSelection, [endpoint.id]: event.target.value })}
                        >
                          {models.map((model) => (
                            <option key={model.model_id} value={model.model_id}>{modelLabel(model, endpoint.id)}</option>
                          ))}
                        </NativeSelect>
                      </div>
                    ) : null}
                  </div>
                  {!compareMode && chosen?.notes ? <p className="text-2xs leading-snug text-ink-tertiary">{chosen.notes}</p> : null}
                  {compareMode ? (
                    <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-13 text-ink">
                      {models.map((model) => {
                        const checked = (compareSelection[endpoint.id] ?? []).includes(model.model_id);
                        return (
                          <label key={model.model_id} className="flex items-center gap-1.5">
                            <input
                              type="checkbox"
                              className="accent-brand"
                              checked={checked}
                              disabled={disabled}
                              onChange={() => {
                                const existing = compareSelection[endpoint.id] ?? [];
                                const next = checked ? existing.filter((item) => item !== model.model_id) : [...existing, model.model_id];
                                onCompareSelection({ ...compareSelection, [endpoint.id]: next });
                              }}
                            />
                            <span title={model.notes || undefined}>{modelLabel(model, endpoint.id)}</span>
                          </label>
                        );
                      })}
                    </div>
                  ) : null}
                </div>
              );
            })}
            {sharedModel ? <p className="text-xs text-link">Các phép đo đã chọn dùng chung một mô hình; predictor chỉ chạy suy luận một lần.</p> : null}
          </div>

          {!batchActive ? (
            <label className="flex items-center gap-2 text-13 text-ink">
              <input type="checkbox" className="accent-brand" checked={compareMode} disabled={disabled} onChange={(event) => onCompareMode(event.target.checked)} />
              So sánh mô hình (mỗi mô hình giữ kết quả riêng)
            </label>
          ) : null}

          {expertMode ? (
            <div className="space-y-1.5">
              <Label htmlFor="qp-threshold" className="text-13 font-semibold">Ghi đè ngưỡng hERG (chuyên gia)</Label>
              <p className="text-xs text-ink-secondary">Backend từ chối ghi đè nếu danh tính đang đăng nhập không có role expert.</p>
              <Input
                id="qp-threshold"
                className="h-9 w-40 font-mono text-sm"
                placeholder="vd. 0.3"
                inputMode="decimal"
                value={thresholdHerg}
                disabled={disabled}
                onChange={(event) => onThresholdHerg(event.target.value)}
              />
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
