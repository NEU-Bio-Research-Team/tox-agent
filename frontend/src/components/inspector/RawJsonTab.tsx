import type { RunDetail } from '../../lib/api/types';

export function RawJsonTab({ run }: { run: RunDetail }) {
  return (
    <pre
      className="overflow-x-auto rounded-lg p-3 text-xs"
      style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}
    >
      {JSON.stringify(run, null, 2)}
    </pre>
  );
}
