import type { ConnectionStatus } from '../../shared/events/eventBus';

const LABEL: Record<ConnectionStatus, string> = {
  connecting: 'đang kết nối',
  live: 'live',
  reconnecting: 'đang kết nối lại',
  offline: 'mất kết nối',
};

const COLOR: Record<ConnectionStatus, string> = {
  connecting: 'var(--ink-tertiary)',
  live: 'var(--success)',
  reconnecting: 'var(--warning)',
  offline: 'var(--danger)',
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
