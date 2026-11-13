import type { Limitation } from '../../../shared/api/types';
import { LIMITATION_LABEL_VI } from '../../../shared/lib/labels';
import { Callout } from '../../../shared/ui/callout';

/** N-5: a limitation is content, never a collapsed footer disclaimer. */
export function LimitationBlock({ limitations }: { limitations: Limitation[] }) {
  if (limitations.length === 0) return null;
  return (
    <Callout tone="warning" title="Giới hạn của kết quả này" className="text-13">
      <ul className="space-y-1 text-ink-secondary">
        {limitations.map((limitation) => (
          <li key={limitation.code}>• {LIMITATION_LABEL_VI[limitation.code] ?? limitation.text}</li>
        ))}
      </ul>
    </Callout>
  );
}
