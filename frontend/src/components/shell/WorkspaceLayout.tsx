import { type ReactNode, useState } from 'react';
import { SidebarInset, SidebarProvider } from '../ui/sidebar';
import { AppSidebar } from './AppSidebar';
import { getStoredSidebarCollapsed, setLayoutPreferences } from '../../lib/preferences';
import { useBreakpoint } from '../../hooks/useBreakpoint';

/**
 * The full-viewport three-region shell (plan section 8.2): sidebar left
 * (this component owns its collapse-to-rail/Sheet behavior via the shadcn
 * `Sidebar` primitive), everything else — chat, artifacts — is `children`
 * inside `SidebarInset`. `100dvh` per section 8.2's sizing table, not
 * `max-w-6xl`: this shell is meant to fill the viewport, unlike the public
 * landing/about pages which keep Navbar/Footer and a centered max width.
 */
export function WorkspaceLayout({ children }: { children: ReactNode }) {
  const breakpoint = useBreakpoint();
  // Section 8.2.2: tablet (768–1279px) starts collapsed to the icon rail so
  // the chat/artifacts columns keep their room; desktop starts expanded.
  // A stored preference means the user already made a choice, which then
  // wins over that default regardless of viewport.
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
    </SidebarProvider>
  );
}
