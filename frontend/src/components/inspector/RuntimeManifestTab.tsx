import type { RuntimeUsageEvent, RunDetail } from '../../lib/api/types';

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b py-2 text-xs last:border-0" style={{ borderColor: 'var(--border)' }}>
      <span style={{ color: 'var(--text-faint)' }}>{label}</span>
      <span className="max-w-[60%] truncate text-right font-mono" style={{ color: 'var(--text)' }} title={value}>
        {value}
      </span>
    </div>
  );
}

export function RuntimeManifestTab({ run }: { run: RunDetail }) {
  if (!run.runtime) {
    return (
      <div className="space-y-3 text-xs" style={{ color: 'var(--text-faint)' }}>
        <p>Run này không có runtime binding (lane deterministic không mở agent runtime).</p>
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
      <Row label="profile_hash" value={runtime.profile_hash} />
      <Row label="tool_schema_hash" value={runtime.tool_schema_hash} />
      <Row label="system_prompt_hash" value={runtime.system_prompt_hash} />
      <Row label="potentially_billed" value={String(run.potentially_billed)} />
      {run.potentially_billed && <PotentiallyBilledWarning />}
      <UsageReports run={run} />
    </div>
  );
}

function PotentiallyBilledWarning() {
  return (
    <p className="mt-3 rounded-lg border p-2 text-xs" style={{ borderColor: 'var(--accent-yellow)', color: 'var(--text-muted)' }}>
      Nhà cung cấp có thể đã nhận lượt này trước khi run kết thúc. Chi phí thực tế chưa thể xác nhận từ trạng thái run.
    </p>
  );
}

function UsageReports({ run }: { run: RunDetail }) {
  if (run.usage.status === 'unknown') {
    return (
      <p className="mt-3 text-xs italic" style={{ color: 'var(--text-faint)' }}>
        Runtime chưa báo usage/cost. Đây là “không biết”, không phải chi phí bằng 0.
      </p>
    );
  }

  return (
    <div className="mt-4 space-y-3 border-t pt-3" style={{ borderColor: 'var(--border)' }}>
      <p className="text-xs font-medium" style={{ color: 'var(--text)' }}>Usage do runtime báo</p>
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
    <div className="rounded-lg border p-2 text-xs" style={{ borderColor: 'var(--border)' }}>
      <p className="mb-1 font-mono" style={{ color: 'var(--text-muted)' }}>
        {event.provider_id} · {event.model_id}
      </p>
      <p className="mb-1" style={{ color: 'var(--text-faint)' }}>
        {new Date(event.reported_at).toLocaleString('vi-VN')}
      </p>
      {tokenRows.map(([label, value]) => <Row key={label} label={`tokens.${label}`} value={value === null ? 'không biết' : String(value)} />)}
      <Row label="cost" value={cost} />
    </div>
  );
}
