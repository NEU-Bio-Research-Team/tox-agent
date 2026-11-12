import { useEffect, useState } from 'react';
import { Link, NavLink, useLocation, useNavigate } from 'react-router';
import { ExternalLink, FlaskConical, LogOut, Menu, Moon, Settings, Star, Sun, User } from 'lucide-react';
import logoImage from '../../assets/logo-tox.png';
import { Button } from '../../shared/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '../../shared/ui/popover';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '../../shared/ui/sheet';
import { getToken, setToken } from '../../shared/api/client';
import { queryClient } from '../../shared/lib/queryClient';
import { applyTheme, getTheme, type Theme } from '../../shared/lib/theme';
import { identityFromToken } from '../../features/auth/identity';
import { APP_VERSION, BUILD_TIME_LABEL } from '../build-info';
import { ReleaseNotesModal } from './ReleaseNotesModal';

const REPO_URL = 'https://github.com/NEU-Bio-Research-Team/tox-agent';
const RATE_US_URL = 'https://docs.google.com/forms/d/e/1FAIpQLScgcYA17KI_UdS4JrHqK7d8HupeUQ-MsgRB3XUBeGQN1MbpdQ/viewform';

/** The production bar carried About, Settings and Documents. Sessions is the
 * one addition: it is the way into the workbench, which the production site
 * did not have and which must stay reachable from every page with this bar. */
const NAV_LINKS = [
  { to: '/sessions', label: 'Sessions' },
  { to: '/about', label: 'About' },
  { to: '/settings', label: 'Settings' },
  { to: '/documents', label: 'Documents' },
] as const;

const BUILD_LABEL = `v${APP_VERSION} • ${BUILD_TIME_LABEL}`;

/**
 * The production navigation bar, on this app's own state.
 *
 * Layout, labels and styling are the production site's. What sits behind each
 * control is this codebase's: the session is the bearer token `RequireToken`
 * manages, the theme is the `dark` class theme.css already styles, and every
 * link is a route this router serves.
 */
export function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();
  const [theme, setTheme] = useState<Theme>(() => getTheme());
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [releaseNotesOpen, setReleaseNotesOpen] = useState(false);

  const identity = identityFromToken(getToken());
  const isAuthenticated = identity !== null;
  const displayName = identity?.name ?? identity?.email ?? 'Signed in';

  useEffect(() => {
    setMobileMenuOpen(false);
    setUserMenuOpen(false);
  }, [location.pathname]);

  const toggleTheme = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    setTheme(next);
  };

  // Signing out drops the token and everything fetched with it: a cached
  // session list belongs to the identity that loaded it.
  const handleLogout = () => {
    setToken(null);
    queryClient.clear();
    setMobileMenuOpen(false);
    setUserMenuOpen(false);
    navigate('/');
  };

  const openReleaseNotes = () => {
    setMobileMenuOpen(false);
    setReleaseNotesOpen(true);
  };

  const navLinkClassName = ({ isActive }: { isActive: boolean }) =>
    `text-sm font-medium transition-colors ${
      isActive ? 'text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
    }`;

  const mobileNavLinkClassName = ({ isActive }: { isActive: boolean }) =>
    `block rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
      isActive
        ? 'bg-[var(--surface-alt)] text-[var(--text)]'
        : 'text-[var(--text-muted)] hover:bg-[var(--surface-alt)] hover:text-[var(--text)]'
    }`;

  const themeTitle = `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`;

  return (
    <nav
      className="legacy-ui sticky top-0 z-50 border-b backdrop-blur-xl"
      style={{ borderColor: 'var(--border)', backgroundColor: 'color-mix(in srgb, var(--bg) 80%, transparent)' }}
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-3 px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <Link to="/analyze" className="flex min-w-0 items-center gap-3 transition-opacity hover:opacity-80">
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
              <FlaskConical className="h-3 w-3" />
              Beta
            </span>
            <span className="ml-2 hidden xl:inline">{BUILD_TIME_LABEL}</span>
          </button>
        </div>

        <div className="hidden items-center gap-5 md:flex">
          {NAV_LINKS.map((link) => (
            <NavLink key={link.to} to={link.to} className={navLinkClassName}>
              {link.label}
            </NavLink>
          ))}

          <a
            href={RATE_US_URL}
            className="inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm font-semibold transition-all hover:-translate-y-[1px]"
            style={{
              borderColor: '#0891b2',
              backgroundColor: '#22d3ee',
              color: '#083344',
              boxShadow: '0 6px 14px rgba(8,145,178,0.28)',
            }}
            title="Rate ToxAgent"
          >
            <Star className="h-3.5 w-3.5" />
            Rate us!
          </a>

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
            className="h-8 w-8 rounded-full"
            style={{ color: 'var(--text-muted)' }}
            title={themeTitle}
            aria-label={themeTitle}
          >
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>

          {isAuthenticated ? (
            <Popover open={userMenuOpen} onOpenChange={setUserMenuOpen}>
              <PopoverTrigger asChild>
                <button
                  type="button"
                  className="flex items-center gap-2 rounded-lg px-3 py-1.5 transition-colors hover:bg-[var(--surface-alt)]"
                  style={{ color: 'var(--text)' }}
                  aria-label="Account menu"
                >
                  <User className="h-4 w-4" />
                  <span className="max-w-32 truncate text-sm font-medium">{displayName}</span>
                </button>
              </PopoverTrigger>
              <PopoverContent
                align="end"
                className="legacy-ui w-56 p-1"
                style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
              >
                <div className="space-y-1 px-2 py-1.5">
                  <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>
                    {identity?.name ?? 'Signed in'}
                  </p>
                  {identity?.email && (
                    <p className="truncate text-xs" style={{ color: 'var(--text-muted)' }}>
                      {identity.email}
                    </p>
                  )}
                </div>
                <div className="my-1 h-px" style={{ backgroundColor: 'var(--border)' }} />
                <Link
                  to="/settings"
                  className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-[var(--surface-alt)]"
                  style={{ color: 'var(--text)' }}
                >
                  <Settings className="h-4 w-4" />
                  Settings
                </Link>
                <button
                  type="button"
                  onClick={handleLogout}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-[var(--surface-alt)]"
                  style={{ color: 'var(--accent-red)' }}
                >
                  <LogOut className="h-4 w-4" />
                  Logout
                </button>
              </PopoverContent>
            </Popover>
          ) : (
            <Button size="sm" asChild style={{ backgroundColor: 'var(--accent-blue)', color: '#ffffff' }}>
              <Link to="/analyze">Sign in</Link>
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
            title={themeTitle}
            aria-label={themeTitle}
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
                aria-label="Menu"
              >
                <Menu className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent
              side="right"
              className="legacy-ui gap-0 border-l p-0"
              style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
            >
              <SheetHeader className="border-b px-6 py-5 text-left" style={{ borderColor: 'var(--border)' }}>
                <SheetTitle style={{ color: 'var(--text)' }}>Navigation</SheetTitle>
                <SheetDescription style={{ color: 'var(--text-muted)' }}>
                  Browse the app and manage your account.
                </SheetDescription>
              </SheetHeader>

              <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-6 py-5">
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
                  <NavLink to="/analyze" end className={mobileNavLinkClassName}>
                    Home
                  </NavLink>
                  {NAV_LINKS.map((link) => (
                    <NavLink key={link.to} to={link.to} className={mobileNavLinkClassName}>
                      {link.label}
                    </NavLink>
                  ))}
                  <a
                    href={RATE_US_URL}
                    className="flex items-center justify-between rounded-lg border px-3 py-2 text-sm font-semibold transition-colors"
                    style={{ borderColor: '#0891b2', backgroundColor: '#22d3ee', color: '#083344' }}
                  >
                    <span>Rate us!</span>
                    <Star className="h-4 w-4" />
                  </a>
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
                          {identity?.name ?? 'Signed in'}
                        </p>
                        {identity?.email && (
                          <p className="truncate text-xs" style={{ color: 'var(--text-muted)' }}>
                            {identity.email}
                          </p>
                        )}
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
                    <Button className="w-full" asChild style={{ backgroundColor: 'var(--accent-blue)', color: '#ffffff' }}>
                      <Link to="/analyze">Sign in</Link>
                    </Button>
                  )}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>

      <ReleaseNotesModal open={releaseNotesOpen} onOpenChange={setReleaseNotesOpen} />
    </nav>
  );
}
