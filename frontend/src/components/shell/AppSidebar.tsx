import { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { Info, Loader2, Plus, Search, Settings } from 'lucide-react';
import logoImage from '../../assets/logo-tox.png';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
  SidebarTrigger,
} from '../ui/sidebar';
import { Input } from '../ui/input';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createSession } from '../../lib/api/endpoints';
import { useSessionsList } from '../../hooks/useSessionsList';
import { groupByRecency } from '../../lib/dateGroups';
import { SessionSidebarRow } from './SessionRow';
import { RUN_STATUS_LABEL_VI } from '../../lib/labels';

export function AppSidebar() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [search, setSearch] = useState('');
  const { rows, isLoading, isError, hasNextPage, isFetchingNextPage, fetchNextPage } = useSessionsList();

  const createMutation = useMutation({
    mutationFn: () => createSession({ preferred_language: 'vi' }),
    onSuccess: (session) => {
      void queryClient.invalidateQueries({ queryKey: ['sessions'] });
      navigate(`/s/${session.session_id}`);
    },
  });

  const filtered = useMemo(() => {
    if (!search.trim()) return rows;
    const needle = search.toLowerCase();
    return rows.filter((row) =>
      `${row.title ?? ''} ${row.last_message_preview ?? ''} ${row.session_id}`.toLowerCase().includes(needle),
    );
  }, [rows, search]);

  const runningSessions = useMemo(() => rows.filter((row) => row.active_run !== null), [rows]);
  const groups = useMemo(() => groupByRecency(filtered, (row) => row.updated_at), [filtered]);

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <div className="flex items-center justify-between px-1 py-1">
          <Link to="/" className="flex min-w-0 items-center gap-2 group-data-[collapsible=icon]:hidden">
            <img src={logoImage} alt="ToxAgent" className="h-6 w-6 shrink-0" />
            <span className="truncate text-sm font-semibold" style={{ color: 'var(--text)' }}>
              ToxAgent
            </span>
          </Link>
          <SidebarTrigger />
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  tooltip="Session mới"
                  onClick={() => createMutation.mutate()}
                  disabled={createMutation.isPending}
                >
                  {createMutation.isPending ? (
                    <Loader2 className="animate-spin" />
                  ) : (
                    <Plus />
                  )}
                  <span>{createMutation.isPending ? 'Đang tạo…' : 'Session mới'}</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
          <div className="relative px-1">
            <Search
              className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2"
              style={{ color: 'var(--text-faint)' }}
            />
            <Input
              placeholder="Tìm session…"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              className="h-8 pl-8 text-xs"
              aria-label="Tìm session trong danh sách đã tải"
            />
          </div>
        </SidebarGroup>

        {runningSessions.length > 0 && (
          <SidebarGroup>
            <SidebarGroupLabel>Tác vụ đang chạy</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {runningSessions.map((row) => (
                  <SidebarMenuItem key={row.session_id}>
                    <SidebarMenuButton asChild tooltip={`${row.title ?? row.session_id} — ${RUN_STATUS_LABEL_VI[row.active_run!.status]}`}>
                      <Link to={`/s/${row.session_id}`}>
                        <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full" style={{ backgroundColor: 'var(--accent-blue)' }} />
                        <span className="truncate">{row.title ?? row.session_id}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}

        <SidebarSeparator />

        {isLoading && (
          <p className="px-3 py-2 text-xs group-data-[collapsible=icon]:hidden" style={{ color: 'var(--text-muted)' }}>
            Đang tải…
          </p>
        )}
        {isError && (
          <p className="px-3 py-2 text-xs group-data-[collapsible=icon]:hidden" style={{ color: 'var(--accent-red)' }}>
            Không tải được danh sách session.
          </p>
        )}
        {!isLoading && filtered.length === 0 && (
          <p className="px-3 py-2 text-xs group-data-[collapsible=icon]:hidden" style={{ color: 'var(--text-faint)' }}>
            {search.trim() ? 'Không có kết quả tìm kiếm trong danh sách đã tải.' : 'Chưa có session nào.'}
          </p>
        )}

        {groups.map((group) => (
          <SidebarGroup key={group.label}>
            <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {group.rows.map((row) => (
                  <SessionSidebarRow key={row.session_id} row={row} />
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}

        {hasNextPage && !search.trim() && (
          <button
            type="button"
            onClick={() => void fetchNextPage()}
            disabled={isFetchingNextPage}
            className="mx-2 mb-2 rounded-md px-2 py-1.5 text-left text-xs group-data-[collapsible=icon]:hidden"
            style={{ color: 'var(--accent-blue)' }}
          >
            {isFetchingNextPage ? 'Đang tải thêm…' : 'Tải thêm'}
          </button>
        )}
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="Tất cả sessions">
              <Link to="/sessions">
                <Search />
                <span>Tất cả sessions</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="Cài đặt">
              <Link to="/settings">
                <Settings />
                <span>Cài đặt</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="Giới thiệu">
              <Link to="/about">
                <Info />
                <span>Giới thiệu</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
