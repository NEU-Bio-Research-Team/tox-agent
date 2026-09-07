import type { ConnectionStatus } from '../../lib/store/eventBus';

const LABEL: Record<ConnectionStatus, string> = {
  connecting: 'đang kết nối',
  live: 'live',
  reconnecting: 'đang kết nối lại',
  offline: 'mất kết nối',
};

const COLOR: Record<ConnectionStatus, string> = {
  connecting: 'var(--text-faint)',
  live: 'var(--accent-green)',
  reconnecting: 'var(--accent-yellow)',
  offline: 'var(--accent-red)',
};

export function ConnectionIndicator({ status }: { status: ConnectionStatus }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium" style={{ color: COLOR[status] }}>
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: COLOR[status], ...(status === 'live' ? {} : { opacity: 0.7 }) }}
      />
      {LABEL[status]}
    </span>
  );
}
