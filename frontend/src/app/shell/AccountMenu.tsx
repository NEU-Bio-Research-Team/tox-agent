import { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { LogOut, Settings, User } from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from '../../shared/ui/popover';
import { getToken } from '../../shared/api/client';
import { queryClient } from '../../shared/lib/queryClient';
import { identityFromToken } from '../../features/auth/identity';
import { signOut } from '../../features/auth/firebaseAuth';

/** The signed-in chip at the foot of the rail. The identity is read from the
 * token's claims for display only; the control plane verifies every request. */
export function AccountMenu() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const identity = identityFromToken(getToken());
  const label = identity?.name ?? identity?.email ?? 'Đã đăng nhập';

  // Signing out drops the token and everything fetched with it: a cached
  // session list belongs to the identity that loaded it.
  const logout = () => {
    signOut();
    queryClient.clear();
    setOpen(false);
    navigate('/');
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label="Tài khoản"
          className="flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-2 py-1.5 text-left outline-none transition-colors hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-sidebar-ring"
        >
          <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-brand-soft-strong text-brand-ink">
            <User className="size-3.5" />
          </span>
          <span className="truncate text-13 text-ink group-data-[collapsible=icon]:hidden">{label}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" side="top" className="w-60 p-1.5">
        <div className="space-y-0.5 px-2 py-1.5">
          <p className="truncate text-sm font-medium text-ink">{identity?.name ?? 'Đã đăng nhập'}</p>
          {identity?.email ? <p className="truncate text-xs text-ink-secondary">{identity.email}</p> : null}
        </div>
        <div className="my-1 h-px bg-line" />
        <Link
          to="/settings"
          onClick={() => setOpen(false)}
          className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-ink transition-colors hover:bg-surface-muted"
        >
          <Settings className="size-4" />
          Cài đặt
        </Link>
        <button
          type="button"
          onClick={logout}
          className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm text-danger transition-colors hover:bg-surface-muted"
        >
          <LogOut className="size-4" />
          Đăng xuất
        </button>
      </PopoverContent>
    </Popover>
  );
}
