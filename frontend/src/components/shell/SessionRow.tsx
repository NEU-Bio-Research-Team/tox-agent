import { NavLink } from 'react-router';
import type { SessionListRow } from '../../lib/api/types';
import { RUN_STATUS_LABEL_VI } from '../../lib/labels';
import { relativeTimeVi } from '../../lib/dateGroups';
import { SidebarMenuButton, SidebarMenuItem } from '../ui/sidebar';

function statusDotColor(row: SessionListRow): string {
  if (row.active_run) return 'var(--accent-blue)';
  if (row.status === 'active') return 'var(--accent-green)';
  return 'var(--text-faint)';
}

/** Compact row used in the sidebar's recent-sessions groups (plan 8.1: one
 * real `<a>`/`NavLink` per row, for keyboard nav and open-in-new-tab). */
export function SessionSidebarRow({ row }: { row: SessionListRow }) {
  return (
    <SidebarMenuItem>
      <SidebarMenuButton asChild tooltip={row.title ?? row.session_id}>
        <NavLink to={`/s/${row.session_id}`}>
          {({ isActive }) => (
            <>
              <span
                className="h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ backgroundColor: statusDotColor(row) }}
              />
              <span className="truncate" style={{ fontWeight: isActive ? 600 : 400 }}>
                {row.title ?? row.session_id}
              </span>
            </>
          )}
        </NavLink>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

/** Fuller row used on `/sessions` (plan 8.1: title, status badge, preview,
 * relative time — the "expanded history view" that shares the same query
 * cache as the sidebar's list instead of owning a second one). */
export function SessionCardRow({ row, onOpen }: { row: SessionListRow; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="w-full rounded-xl border p-4 text-left transition-shadow hover:shadow-md"
      style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: statusDotColor(row) }} />
            <p className="truncate text-sm font-medium" style={{ color: 'var(--text)' }}>
              {row.title ?? row.session_id}
            </p>
          </div>
          {row.last_message_preview && (
            <p className="mt-1 truncate text-sm" style={{ color: 'var(--text-muted)' }}>
              {row.last_message_preview}
            </p>
          )}
          <p className="mt-1.5 text-xs" style={{ color: 'var(--text-faint)' }}>
            {row.session_id} · {row.run_count} run · cập nhật {relativeTimeVi(row.updated_at)}
          </p>
        </div>
        {row.active_run && (
          <span
            className="shrink-0 rounded-full px-2 py-1 text-xs font-medium"
            style={{ backgroundColor: 'var(--accent-blue-muted)', color: 'var(--accent-blue)' }}
          >
            {RUN_STATUS_LABEL_VI[row.active_run.status]}
          </span>
        )}
      </div>
    </button>
  );
}
