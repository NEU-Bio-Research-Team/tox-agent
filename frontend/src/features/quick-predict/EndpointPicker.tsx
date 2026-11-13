import type { Endpoint } from '../../shared/api/types';
import { cn } from '../../shared/ui/utils';
import { ENDPOINT_DESCRIPTION, ENDPOINT_LABEL, ENDPOINT_ORDER, modelLabel, type EndpointCapability } from './endpointLabels';

/**
 * The measurements to run. Each is a separate measurement with its own model
 * (ADR-0002), so each is its own row. One the deployment does not serve stays
 * in the list, disabled and labelled, instead of vanishing.
 */
export function EndpointPicker({
  capabilities,
  selected,
  onChange,
  modelSelection,
  disabled,
}: {
  capabilities: ReadonlyArray<EndpointCapability>;
  selected: ReadonlyArray<Endpoint>;
  onChange: (next: Endpoint[]) => void;
  modelSelection: Partial<Record<Endpoint, string>>;
  disabled?: boolean;
}) {
  const rows = [...capabilities].sort((a, b) => ENDPOINT_ORDER.indexOf(a.id) - ENDPOINT_ORDER.indexOf(b.id));

  return (
    <fieldset className="m-0 min-w-0 space-y-2 border-0 p-0">
      <legend className="p-0 text-13 font-semibold text-ink">Phép đo</legend>
      <p className="text-xs text-ink-secondary">Mỗi phép đo chạy riêng, với mô hình riêng.</p>
      {rows.length === 0 ? <p className="text-xs text-ink-secondary">Danh sách phép đo của triển khai chưa tải xong.</p> : null}
      {rows.map((endpoint) => {
        const checked = selected.includes(endpoint.id);
        const model = (endpoint.models ?? []).find((item) => item.model_id === modelSelection[endpoint.id]);
        const available = endpoint.enabled;
        // At least one measurement must stay selected.
        const lastOne = checked && selected.length === 1;
        return (
          <label
            key={endpoint.id}
            title={endpoint.blocked_reason ?? undefined}
            className={cn(
              'flex items-center gap-3 rounded-lg border px-3.5 py-3 transition-colors',
              !available && 'cursor-not-allowed border-line bg-surface-muted',
              available && checked && 'cursor-pointer border-brand-border bg-brand-soft',
              available && !checked && 'cursor-pointer border-line bg-card hover:bg-surface-muted',
            )}
          >
            <input
              type="checkbox"
              className="size-4 accent-brand"
              checked={available && checked}
              disabled={!available || disabled || lastOne}
              onChange={() => onChange(checked ? selected.filter((item) => item !== endpoint.id) : [...selected, endpoint.id])}
            />
            <span className="min-w-0 flex-1">
              <span className={cn('block text-sm font-semibold', available ? 'text-ink' : 'text-ink-tertiary')}>{ENDPOINT_LABEL[endpoint.id]}</span>
              <span className={cn('block text-xs', available ? 'text-ink-secondary' : 'text-ink-tertiary')}>{ENDPOINT_DESCRIPTION[endpoint.id]}</span>
            </span>
            {available ? (
              model ? <span className="shrink-0 font-mono text-2xs text-ink-secondary">{modelLabel(model, endpoint.id)}</span> : null
            ) : (
              <span className="shrink-0 text-2xs text-ink-tertiary">không có ở triển khai này</span>
            )}
          </label>
        );
      })}
    </fieldset>
  );
}
