import type { ReactNode } from 'react';
import { SidebarTrigger } from '../ui/sidebar';
import { Separator } from '../ui/separator';
import { ConnectionIndicator } from './ConnectionIndicator';
import type { ConnectionStatus } from '../../lib/store/eventBus';

/** Top bar inside the chat column (plan section 8.2 wireframe row 1). Not a
 * page-wide Navbar — the sidebar toggle lives here so mobile/tablet users
 * always have one, regardless of which region (rail vs Sheet) the shadcn
 * sidebar primitive is currently rendering. */
export function WorkspaceHeader({
  title,
  subtitle,
  status,
  actions,
}: {
  title: ReactNode;
  subtitle?: string;
  status?: ConnectionStatus;
  actions?: ReactNode;
}) {
  return (
    <header
      className="flex h-14 shrink-0 items-center gap-2 border-b px-3 md:px-4"
      style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface)' }}
    >
      <SidebarTrigger />
      <Separator orientation="vertical" className="h-5" />
      <div className="min-w-0 flex-1">
        <h1 className="truncate text-sm font-semibold" style={{ color: 'var(--text)' }}>
          {title}
        </h1>
        {subtitle && (
          <p className="truncate font-mono text-xs" style={{ color: 'var(--text-faint)' }}>
            {subtitle}
          </p>
        )}
      </div>
      {status && <ConnectionIndicator status={status} />}
      {actions}
    </header>
  );
}
