import { AlertTriangle } from 'lucide-react';
import type { Message } from '../../lib/api/types';
import { ERROR_CODE_LABEL_VI } from '../../lib/labels';

/** A `system_event` message (e.g. startup reconciliation failing an
 * orphaned run) used to have no render branch at all in the transcript — the
 * failure was recorded server-side but invisible to whoever was looking at
 * the session. Styled like `AnalysisSystemCard`'s failure row. */
export function SystemEventCard({ message }: { message: Message }) {
  const errorPart = message.parts.find((p) => p.type === 'error');
  if (!errorPart) return null;
  const code = errorPart.content.code as string | undefined;
  const detail = errorPart.content.message as string | undefined;

  return (
    <div className="my-2 flex items-start gap-2 rounded-lg px-3 py-2 text-xs" style={{ backgroundColor: 'var(--surface-alt)' }}>
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" style={{ color: 'var(--accent-red)' }} />
      <span style={{ color: 'var(--text-muted)' }}>
        {code ? (ERROR_CODE_LABEL_VI[code] ?? code) : 'Sự kiện hệ thống'}
        {detail ? ` — ${detail}` : ''}
      </span>
    </div>
  );
}
