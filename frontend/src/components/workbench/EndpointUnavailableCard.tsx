import { AlertTriangle } from 'lucide-react';

const REASON_VI: Record<string, string> = {
  clintox: 'ClinTox không được phục vụ — thiếu tokenizer cho checkpoint hiện tại.',
};

export function EndpointUnavailableCard({ endpoint }: { endpoint: string }) {
  return (
    <div
      className="rounded-xl border border-dashed p-4"
      style={{ backgroundColor: 'var(--surface-alt)', borderColor: 'var(--border)' }}
    >
      <div className="mb-1 flex items-center gap-2">
        <AlertTriangle className="h-4 w-4" style={{ color: 'var(--accent-yellow)' }} />
        <h3 className="text-sm font-semibold uppercase" style={{ color: 'var(--text-muted)' }}>
          {endpoint} — không khả dụng
        </h3>
      </div>
      <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
        {REASON_VI[endpoint] ?? 'Endpoint này hiện không được phục vụ cho phân tử này.'}
      </p>
    </div>
  );
}
