import { RefreshCw } from 'lucide-react';
import type { RecoveryBanner as RecoveryBannerData } from '../hooks/useSessionEvents';

export function RecoveryBanner({ banner }: { banner: RecoveryBannerData }) {
  return (
    <div
      className="my-2 flex items-start gap-2 rounded-lg border p-3 text-xs"
      style={{ borderColor: 'var(--warning)', backgroundColor: 'var(--surface-muted)' }}
    >
      <RefreshCw className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning" />
      <span style={{ color: 'var(--ink-secondary)' }}>
        Runtime mất kết nối giữa lượt (run <code className="font-mono">{banner.originalRunId.slice(0, 12)}…</code>).
        Đã tạo run khôi phục <code className="font-mono">{banner.recoveryRunId.slice(0, 12)}…</code> — run gốc vẫn giữ
        nguyên trong lịch sử, không bị nối văn bản âm thầm.
      </span>
    </div>
  );
}
