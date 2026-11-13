import { type ReactNode, useState } from 'react';
import { SidebarInset, SidebarProvider } from '../../shared/ui/sidebar';
import { AppSidebar } from './AppSidebar';
import { CommandPalette } from './CommandPalette';
import { getStoredSidebarCollapsed, setLayoutPreferences } from '../../shared/lib/preferences';
import { useBreakpoint } from '../../shared/hooks/useBreakpoint';

/**
 * The full-viewport workspace frame (plan section 6.3): the rail on the left and
 * everything else — a page, or the chat and artifacts — inside `SidebarInset`.
 * `100dvh`, not a max width: this frame fills the viewport. Every signed-in page
 * (Phân tích nhanh, Sessions, Cài đặt, the session workbench) renders inside it,
 * so there is one navigation and one palette across them.
 */
export function WorkspaceLayout({ children }: { children: ReactNode }) {
  const breakpoint = useBreakpoint();
  // Tablet (768-1279px) starts collapsed to the icon rail so the chat and
  // artifacts columns keep their room; desktop starts expanded. A stored
  // preference means the user already chose, which then wins.
  const [open, setOpen] = useState(() => {
    const stored = getStoredSidebarCollapsed();
    if (stored !== null) return !stored;
    return breakpoint !== 'tablet';
  });

  return (
    <SidebarProvider
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        setLayoutPreferences({ sidebarCollapsed: !next });
      }}
      style={{ height: '100dvh' } as React.CSSProperties}
      className="overflow-hidden"
    >
      <AppSidebar />
      <SidebarInset className="min-w-0 overflow-hidden">{children}</SidebarInset>
      <CommandPalette />
    </SidebarProvider>
  );
}
