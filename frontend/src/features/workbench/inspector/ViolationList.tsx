import { XCircle } from 'lucide-react';
import type { Violation } from '../../../shared/api/types';

export function ViolationList({ violations }: { violations: Violation[] }) {
  return (
    <ul className="space-y-2">
      {violations.map((violation, index) => (
        <li key={index} className="rounded-lg p-2.5 text-xs bg-surface-muted">
          <div className="mb-1 flex items-center gap-1.5">
            <XCircle className="h-3.5 w-3.5 shrink-0 text-danger" />
            <span className="font-mono font-semibold text-ink">
              {violation.code}
            </span>
            {violation.path && (
              <span className="font-mono text-ink-tertiary">
                · {violation.path}
              </span>
            )}
          </div>
          <p style={{ color: 'var(--ink-secondary)' }}>{violation.message}</p>
          {(violation.expected !== undefined || violation.actual !== undefined) && (
            <div className="mt-1 flex gap-4 font-mono text-ink-tertiary">
              {violation.expected !== undefined && <span>mong đợi: {String(violation.expected)}</span>}
              {violation.actual !== undefined && <span>nhận: {String(violation.actual)}</span>}
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}
