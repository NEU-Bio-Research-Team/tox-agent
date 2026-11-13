import { NavLink } from 'react-router';
import type { SessionListRow } from '../../shared/api/types';
import { RUN_STATUS_LABEL_VI } from '../../shared/lib/labels';
import { SidebarMenuButton, SidebarMenuItem } from '../../shared/ui/sidebar';

/** Compact row used in the rail's "Gần đây" group: one real link per row, so it
 * works from the keyboard and opens in a new tab. A running session carries a
 * pulsing dot; the others carry none, so the dot always means "working". */
export function SessionSidebarRow({ row }: { row: SessionListRow }) {
  const running = row.active_run !== null;
  const title = row.title ?? 'Session mới';
  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        tooltip={running ? `${title} — ${RUN_STATUS_LABEL_VI[row.active_run!.status]}` : title}
        className="h-8 text-13 text-ink-secondary data-[active=true]:text-sidebar-accent-foreground"
      >
        <NavLink to={`/s/${row.session_id}`}>
          {({ isActive }) => (
            <>
              {running ? (
                <span className="size-[7px] shrink-0 animate-pulse rounded-full bg-brand" aria-hidden="true" />
              ) : (
                <span className="size-[7px] shrink-0" aria-hidden="true" />
              )}
              <span className="truncate" style={{ fontWeight: isActive ? 600 : 400 }}>
                {title}
              </span>
            </>
          )}
        </NavLink>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}
