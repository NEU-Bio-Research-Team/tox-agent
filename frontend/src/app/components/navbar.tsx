import { ExternalLink, FlaskConical, LogOut, Menu, Moon, Settings, Sun, User } from 'lucide-react';
import { Button } from './ui/button';
import { useEffect, useState } from 'react';
import { Link, NavLink, useLocation, useNavigate } from 'react-router';
import logoImage from '../../assets/logo-tox.png';
import { RELEASE_NOTES_EVENT } from '../release-notes';
import { useAuth } from './contexts/auth-context';
import { APP_BUILD_TIME, APP_VERSION } from '../build-info';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from './ui/sheet';

const REPO_URL = 'https://github.com/NEU-Bio-Research-Team/tox-agent';
const NAV_LINKS = [
  { to: '/about', label: 'About' },
  { to: '/settings', label: 'Settings' },
] as const;
const BUILD_TIME_LABEL = `${new Date(APP_BUILD_TIME).toLocaleString('en-GB', {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
  timeZone: 'UTC',
})} UTC`;
const BUILD_LABEL = `v${APP_VERSION} • ${BUILD_TIME_LABEL}`;

export function Navbar() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    if (typeof window === 'undefined') {
      return 'light';
    }

    const storedTheme = window.localStorage.getItem('theme');
    if (storedTheme === 'dark' || storedTheme === 'light') {
      return storedTheme;
    }

    return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
  });
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { user, logout, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    window.localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  const toggleTheme = () => {
    setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'));
  };

  const handleLogout = () => {
    logout();
    setMobileMenuOpen(false);
    navigate('/');
  };

  const openReleaseNotes = () => {
    window.dispatchEvent(new Event(RELEASE_NOTES_EVENT));
  };

  const navLinkClassName = ({ isActive }: { isActive: boolean }) =>
    `text-sm font-medium transition-colors ${
      isActive ? 'text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
    }`;

  const mobileNavLinkClassName = ({ isActive }: { isActive: boolean }) =>
    `rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
      isActive
        ? 'bg-[var(--surface-alt)] text-[var(--text)]'
        : 'text-[var(--text-muted)] hover:bg-[var(--surface-alt)] hover:text-[var(--text)]'
    }`;

  return (
    <nav className="sticky top-0 z-50 border-b backdrop-blur-xl bg-[var(--bg)]/80" style={{ borderColor: 'var(--border)' }}>
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-3 px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <Link to="/" className="flex min-w-0 items-center gap-3 transition-opacity hover:opacity-80">
            <img src={logoImage} alt="ToxAgent Logo" className="h-8 w-auto shrink-0" />
            <div className="hidden min-w-0 sm:block">
              <p className="truncate text-sm font-semibold" style={{ color: 'var(--text)' }}>
                ToxAgent
              </p>
              <p className="truncate text-xs" style={{ color: 'var(--text-muted)' }}>
                Molecular toxicity intelligence
              </p>
            </div>
          </Link>
          <button
            type="button"
            onClick={openReleaseNotes}
            className="hidden items-center rounded-full border px-3 py-1 text-[11px] font-medium uppercase tracking-wide transition-colors lg:inline-flex"
            style={{
              borderColor: 'var(--border)',
              color: 'var(--text-muted)',
              backgroundColor: 'color-mix(in srgb, var(--surface) 82%, transparent)',
            }}
            title={`${BUILD_LABEL} • Click to view release notes`}
          >
            <span className="mr-2">v{APP_VERSION}</span>
            <span
              className="inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px] font-semibold"
              style={{
                borderColor: 'var(--accent-blue)',
                color: 'var(--accent-blue)',
                backgroundColor: 'var(--accent-blue-muted)',
              }}
            >
              <FlaskConical className="w-3 h-3" />
              Beta
            </span>
            <span className="hidden md:inline ml-2">{BUILD_TIME_LABEL}</span>
          </button>
        </div>

        <div className="hidden items-center gap-5 md:flex">
          {NAV_LINKS.map((link) => (
            <NavLink key={link.to} to={link.to} className={navLinkClassName}>
              {link.label}
            </NavLink>
          ))}
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-sm font-medium text-[var(--text-muted)] transition-colors hover:text-[var(--text)]"
          >
            GitHub
            <ExternalLink className="h-3 w-3" />
          </a>

          <div className="h-4 w-px" style={{ backgroundColor: 'var(--border)' }} />

          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="w-8 h-8 rounded-full"
            style={{ color: 'var(--text-muted)' }}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>

          {isAuthenticated ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  className="flex items-center gap-2 rounded-lg px-3 py-1.5 transition-colors hover:bg-[var(--surface-alt)]"
                  style={{ color: 'var(--text)' }}
                >
                  <User className="h-4 w-4" />
                  <span className="max-w-32 truncate text-sm font-medium">{user?.name || user?.email}</span>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="end"
                className="w-56"
                style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
              >
                <DropdownMenuLabel>
                  <div className="space-y-1">
                    <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>
                      {user?.name || 'Signed in'}
                    </p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {user?.email}
                    </p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator style={{ backgroundColor: 'var(--border)' }} />
                <DropdownMenuItem asChild>
                  <Link to="/settings" className="cursor-pointer" style={{ color: 'var(--text)' }}>
                    <Settings className="h-4 w-4" />
                    Settings
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={handleLogout}
                  variant="destructive"
                  className="cursor-pointer"
                >
                  <LogOut className="h-4 w-4" />
                  Logout
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Button
              size="sm"
              asChild
              style={{ backgroundColor: 'var(--accent-blue)', color: '#ffffff' }}
            >
              <Link to="/login">Sign in</Link>
            </Button>
          )}
        </div>

        <div className="flex items-center gap-2 md:hidden">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="h-9 w-9 rounded-full"
            style={{ color: 'var(--text-muted)' }}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>

          <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9 rounded-full"
                style={{ color: 'var(--text)' }}
              >
                <Menu className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent
              side="right"
              className="border-l p-0"
              style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
            >
              <SheetHeader className="border-b px-6 py-5 text-left" style={{ borderColor: 'var(--border)' }}>
                <SheetTitle style={{ color: 'var(--text)' }}>Navigation</SheetTitle>
                <SheetDescription style={{ color: 'var(--text-muted)' }}>
                  Browse the app and manage your account.
                </SheetDescription>
              </SheetHeader>

              <div className="flex h-full flex-col px-6 py-5">
                <div className="mb-6">
                  <button
                    type="button"
                    onClick={openReleaseNotes}
                    className="flex w-full items-center justify-between rounded-xl border px-4 py-3 text-left transition-colors"
                    style={{
                      borderColor: 'var(--border)',
                      backgroundColor: 'var(--surface-alt)',
                      color: 'var(--text)',
                    }}
                  >
                    <div>
                      <p className="text-sm font-semibold">Release notes</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {BUILD_LABEL}
                      </p>
                    </div>
                    <FlaskConical className="h-4 w-4" style={{ color: 'var(--accent-blue)' }} />
                  </button>
                </div>

                <div className="space-y-2">
                  <NavLink to="/" end className={mobileNavLinkClassName}>
                    Home
                  </NavLink>
                  {NAV_LINKS.map((link) => (
                    <NavLink key={link.to} to={link.to} className={mobileNavLinkClassName}>
                      {link.label}
                    </NavLink>
                  ))}
                  <a
                    href={REPO_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between rounded-lg px-3 py-2 text-sm font-medium text-[var(--text-muted)] transition-colors hover:bg-[var(--surface-alt)] hover:text-[var(--text)]"
                  >
                    <span>GitHub</span>
                    <ExternalLink className="h-4 w-4" />
                  </a>
                </div>

                <div className="mt-auto border-t pt-5" style={{ borderColor: 'var(--border)' }}>
                  {isAuthenticated ? (
                    <div className="space-y-3">
                      <div className="rounded-xl px-4 py-3" style={{ backgroundColor: 'var(--surface-alt)' }}>
                        <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>
                          {user?.name || 'Signed in'}
                        </p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                          {user?.email}
                        </p>
                      </div>
                      <Button
                        variant="outline"
                        asChild
                        className="w-full justify-start"
                        style={{ borderColor: 'var(--border)', color: 'var(--text)' }}
                      >
                        <Link to="/settings">
                          <Settings className="h-4 w-4" />
                          Settings
                        </Link>
                      </Button>
                      <Button
                        variant="outline"
                        onClick={handleLogout}
                        className="w-full justify-start"
                        style={{ borderColor: 'rgba(239,68,68,0.35)', color: 'var(--accent-red)' }}
                      >
                        <LogOut className="h-4 w-4" />
                        Logout
                      </Button>
                    </div>
                  ) : (
                    <Button
                      className="w-full"
                      asChild
                      style={{ backgroundColor: 'var(--accent-blue)', color: '#ffffff' }}
                    >
                      <Link to="/login">Sign in</Link>
                    </Button>
                  )}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </nav>
  );
}
