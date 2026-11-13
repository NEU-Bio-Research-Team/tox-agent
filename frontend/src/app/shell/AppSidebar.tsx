import { useMemo } from 'react';
import { Link, useLocation } from 'react-router';
import { BookOpen, ChevronRight, FlaskConical, Loader2, MessageSquare, Plus, Search, Settings } from 'lucide-react';
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
  SidebarTrigger,
} from '../../shared/ui/sidebar';
import { Logo } from '../../shared/ui/logo';
import { useSessionsList } from '../../features/sessions/useSessionsList';
import { useCreateSession } from '../../features/sessions/useCreateSession';
import { SessionSidebarRow } from './SessionRow';
import { AccountMenu } from './AccountMenu';
import { ThemeToggle } from './ThemeToggle';
import { openCommandPalette } from './CommandPalette';

const RECENT_LIMIT = 5;

const NAV = [
  { to: '/analyze', label: 'Phân tích nhanh', icon: FlaskConical, match: ['/analyze', '/predict'] },
  { to: '/sessions', label: 'Sessions', icon: MessageSquare, match: ['/sessions'] },
  { to: '/docs', label: 'Tài liệu', icon: BookOpen, match: ['/docs'] },
] as const;

/**
 * The workspace rail (plan section 6.3): the mark and search, "Session mới", the
 * three destinations, the sessions in flight and the latest few, then Cài đặt and
 * the account. Full history lives on the Sessions page, so the rail no longer
 * repeats it in date groups.
 */
export function AppSidebar() {
  const { pathname } = useLocation();
  const { rows, isLoading, isError } = useSessionsList();
  const create = useCreateSession();

  const recents = useMemo(() => {
    const running = rows.filter((row) => row.active_run !== null);
    const rest = rows.filter((row) => row.active_run === null);
    return [...running, ...rest].slice(0, RECENT_LIMIT);
  }, [rows]);

  const isActive = (match: ReadonlyArray<string>) => match.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="gap-3">
        <div className="flex items-center gap-1 px-1 pt-1">
          <Link to="/" aria-label="ToxAgent" className="flex min-w-0 flex-1 items-center pl-1 group-data-[collapsible=icon]:hidden">
            <Logo withWordmark />
          </Link>
          <button
            type="button"
            onClick={openCommandPalette}
            aria-label="Tìm trang hoặc session (Ctrl+K)"
            title="Tìm (Ctrl+K)"
            className="flex size-8 shrink-0 items-center justify-center rounded-lg text-ink-secondary outline-none transition-colors hover:bg-sidebar-accent hover:text-ink focus-visible:ring-2 focus-visible:ring-sidebar-ring group-data-[collapsible=icon]:hidden"
          >
            <Search className="size-4" />
          </button>
          <SidebarTrigger />
        </div>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              tooltip="Session mới"
              onClick={() => create.mutate()}
              disabled={create.isPending}
              className="h-10 justify-center rounded-lg bg-brand font-semibold text-brand-foreground hover:bg-brand/90 hover:text-brand-foreground active:bg-brand/90 active:text-brand-foreground group-data-[collapsible=icon]:justify-center"
            >
              {create.isPending ? <Loader2 className="animate-spin" /> : <Plus />}
              <span>{create.isPending ? 'Đang tạo…' : 'Session mới'}</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <nav aria-label="Điều hướng chính">
            <SidebarMenu>
              {NAV.map((item) => (
                <SidebarMenuItem key={item.to}>
                  <SidebarMenuButton asChild tooltip={item.label} isActive={isActive(item.match)} className="h-9 text-sm">
                    <Link to={item.to} aria-current={isActive(item.match) ? 'page' : undefined}>
                      <item.icon />
                      <span>{item.label}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
            </nav>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
          <SidebarGroupLabel>Gần đây</SidebarGroupLabel>
          <SidebarGroupContent>
            {isLoading ? <p className="px-2 py-1.5 text-xs text-ink-secondary">Đang tải…</p> : null}
            {isError ? <p className="px-2 py-1.5 text-xs text-danger">Không tải được danh sách session.</p> : null}
            {!isLoading && !isError && recents.length === 0 ? (
              <p className="px-2 py-1.5 text-xs text-ink-tertiary">Chưa có session nào.</p>
            ) : null}
            <SidebarMenu>
              {recents.map((row) => (
                <SessionSidebarRow key={row.session_id} row={row} />
              ))}
              {rows.length > 0 ? (
                <SidebarMenuItem>
                  <SidebarMenuButton asChild className="h-8 text-13 font-medium text-link">
                    <Link to="/sessions">
                      <span className="size-[7px] shrink-0" aria-hidden="true" />
                      <span>Xem tất cả</span>
                      <ChevronRight className="ml-auto size-3.5" />
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ) : null}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="gap-1 border-t border-sidebar-border pt-3">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="Cài đặt" isActive={pathname === '/settings'} className="h-9 text-sm">
              <Link to="/settings" aria-current={pathname === '/settings' ? 'page' : undefined}>
                <Settings />
                <span>Cài đặt</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        <div className="flex items-center gap-1">
          <AccountMenu />
          <ThemeToggle className="size-8 text-ink-secondary group-data-[collapsible=icon]:hidden" />
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
