import type { RuntimeUsageEvent, RunDetail } from '../../../shared/api/types';

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b py-2 text-xs last:border-0">
      <span style={{ color: 'var(--ink-tertiary)' }}>{label}</span>
      <span className="max-w-[60%] truncate text-right font-mono text-ink" title={value}>
        {value}
      </span>
    </div>
  );
}

export function RuntimeManifestTab({ run }: { run: RunDetail }) {
  const configuration = run.configuration_snapshot;
  if (!run.runtime) {
    return (
      <div className="space-y-3 text-xs text-ink-tertiary">
        <p>Run này không có runtime binding (lane deterministic không mở agent runtime).</p>
        <ConfigurationSnapshot configuration={configuration} />
        {run.potentially_billed && <PotentiallyBilledWarning />}
      </div>
    );
  }
  const runtime = run.runtime;
  return (
    <div>
      <Row label="runtime_binding_id" value={runtime.runtime_binding_id} />
      <Row label="runtime_kind" value={runtime.runtime_kind} />
      <Row label="runtime_version" value={runtime.runtime_version} />
      <Row label="provider_id" value={runtime.provider_id} />
      <Row label="model_id" value={runtime.model_id} />
      <Row label="connection_id" value={runtime.connection_id ?? 'runtime mặc định'} />
      <Row label="profile_hash" value={runtime.profile_hash} />
      <Row label="tool_schema_hash" value={runtime.tool_schema_hash} />
      <Row label="system_prompt_hash" value={runtime.system_prompt_hash} />
      <Row label="potentially_billed" value={String(run.potentially_billed)} />
      {run.potentially_billed && <PotentiallyBilledWarning />}
      <UsageReports run={run} />
      <ConfigurationSnapshot configuration={configuration} />
    </div>
  );
}

function ConfigurationSnapshot({ configuration }: { configuration: RunDetail['configuration_snapshot'] }) {
  if (!configuration) {
    return (
      <p className="mt-4 text-xs italic text-ink-tertiary">
        Run này được tạo trước khi hệ thống lưu snapshot cấu hình.
      </p>
    );
  }
  const bindings = Object.entries(configuration.predictor_bindings);
  return (
    <section className="mt-4 space-y-2 border-t pt-3" aria-label="Cấu hình đã pin">
      <p className="text-xs font-medium text-ink">Cấu hình đã pin khi chạy</p>
      <Row label="AI profile" value={configuration.ai_profile_id ?? 'runtime mặc định'} />
      {bindings.length === 0 ? (
        <p className="text-xs text-ink-tertiary">Predictor dùng lựa chọn tự động.</p>
      ) : bindings.map(([endpoint, modelId]) => (
        <Row key={endpoint} label={`predictor.${endpoint}`} value={modelId} />
      ))}
      <p className="text-xs text-ink-tertiary">
        Đã pin lúc {new Date(configuration.created_at).toLocaleString('vi-VN')}; thay đổi cấu hình phiên sau đó không ảnh hưởng kết quả này.
      </p>
    </section>
  );
}

function PotentiallyBilledWarning() {
  return (
    <p className="mt-3 rounded-lg border p-2 text-xs border-warning text-ink-secondary">
      Nhà cung cấp có thể đã nhận lượt này trước khi run kết thúc. Chi phí thực tế chưa thể xác nhận từ trạng thái run.
    </p>
  );
}

function UsageReports({ run }: { run: RunDetail }) {
  if (run.usage.status === 'unknown') {
    return (
      <p className="mt-3 text-xs italic text-ink-tertiary">
        Runtime chưa báo usage/cost. Đây là “không biết”, không phải chi phí bằng 0.
      </p>
    );
  }

  return (
    <div className="mt-4 space-y-3 border-t pt-3">
      <p className="text-xs font-medium text-ink">Usage do runtime báo</p>
      {run.usage.events.map((event) => <UsageReport key={event.usage_event_id} event={event} />)}
    </div>
  );
}

function UsageReport({ event }: { event: RuntimeUsageEvent }) {
  const tokenRows = [
    ['input', event.tokens.input],
    ['output', event.tokens.output],
    ['reasoning', event.tokens.reasoning],
    ['cache read', event.tokens.cache_read],
    ['cache write', event.tokens.cache_write],
    ['total', event.tokens.total],
  ] as const;
  const cost = event.cost.amount !== null && event.cost.currency !== null
    ? `${event.cost.amount} ${event.cost.currency}`
    : 'không biết';

  return (
    <div className="rounded-lg border p-2 text-xs">
      <p className="mb-1 font-mono text-ink-secondary">
        {event.provider_id} · {event.model_id}
      </p>
      <p className="mb-1 text-ink-tertiary">
        {new Date(event.reported_at).toLocaleString('vi-VN')}
      </p>
      {tokenRows.map(([label, value]) => <Row key={label} label={`tokens.${label}`} value={value === null ? 'không biết' : String(value)} />)}
      <Row label="cost" value={cost} />
    </div>
  );
}
