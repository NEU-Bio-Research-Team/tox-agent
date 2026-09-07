import { Link, NavLink, useNavigate } from 'react-router';
import { LogOut, Menu } from 'lucide-react';
import { useState } from 'react';
import logoImage from '../../assets/logo-tox.png';
import { Button } from '../ui/button';
import { getToken, setToken } from '../../lib/api/client';

const NAV_LINKS = [
  { to: '/predict', label: 'Phân tích nhanh' },
  { to: '/sessions', label: 'Sessions' },
  { to: '/about', label: 'About' },
  { to: '/settings', label: 'Settings' },
] as const;

export function Navbar() {
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const connected = Boolean(getToken());

  return (
    <header
      className="sticky top-0 z-40 border-b backdrop-blur"
      style={{ backgroundColor: 'color-mix(in srgb, var(--surface) 92%, transparent)', borderColor: 'var(--border)' }}
    >
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 md:px-6">
        <Link to="/" className="flex items-center gap-2">
          <img src={logoImage} alt="ToxAgent" className="h-8 w-8" />
          <span className="text-base font-semibold" style={{ color: 'var(--text)' }}>
            ToxAgent
          </span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {NAV_LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                `rounded-md px-3 py-2 text-sm font-medium transition-colors ${isActive ? '' : 'hover:opacity-80'}`
              }
              style={({ isActive }) => ({
                color: isActive ? 'var(--accent-blue)' : 'var(--text-muted)',
                backgroundColor: isActive ? 'var(--accent-blue-muted)' : 'transparent',
              })}
            >
              {link.label}
            </NavLink>
          ))}
          {connected && (
            <Button
              variant="ghost"
              size="sm"
              className="gap-1.5"
              onClick={() => {
                setToken(null);
                navigate('/');
              }}
            >
              <LogOut className="h-3.5 w-3.5" />
              Ngắt kết nối
            </Button>
          )}
        </nav>

        <button
          type="button"
          className="rounded-md p-2 md:hidden"
          onClick={() => setMenuOpen((v) => !v)}
          aria-label="Menu"
        >
          <Menu className="h-5 w-5" style={{ color: 'var(--text)' }} />
        </button>
      </div>

      {menuOpen && (
        <div className="border-t px-4 py-3 md:hidden" style={{ borderColor: 'var(--border)' }}>
          {NAV_LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              onClick={() => setMenuOpen(false)}
              className="block rounded-md px-3 py-2 text-sm font-medium"
              style={{ color: 'var(--text-muted)' }}
            >
              {link.label}
            </NavLink>
          ))}
        </div>
      )}
    </header>
  );
}
