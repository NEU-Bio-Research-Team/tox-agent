import { Info } from 'lucide-react';
import type { Limitation } from '../../lib/api/types';
import { LIMITATION_LABEL_VI } from '../../lib/labels';

/** N-5: a limitation is content, never a collapsed footer disclaimer. */
export function LimitationBlock({ limitations }: { limitations: Limitation[] }) {
  if (limitations.length === 0) return null;
  return (
    <div className="rounded-lg border-l-2 p-3" style={{ borderColor: 'var(--accent-yellow)', backgroundColor: 'var(--surface-alt)' }}>
      <div className="mb-1.5 flex items-center gap-1.5">
        <Info className="h-3.5 w-3.5" style={{ color: 'var(--accent-yellow)' }} />
        <span className="text-xs font-semibold" style={{ color: 'var(--text)' }}>
          Giới hạn của kết quả này
        </span>
      </div>
      <ul className="space-y-1">
        {limitations.map((limitation) => (
          <li key={limitation.code} className="text-xs" style={{ color: 'var(--text-muted)' }}>
            • {LIMITATION_LABEL_VI[limitation.code] ?? limitation.text}
          </li>
        ))}
      </ul>
    </div>
  );
}
