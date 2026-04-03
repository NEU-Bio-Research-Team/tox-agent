import { Moon, Sun, FlaskConical } from 'lucide-react';
import { Button } from './ui/button';
import { useState, useEffect } from 'react';
import { Link } from 'react-router';
import logoImage from '../../assets/a654c40bdf5d3906916ebeed588d27aa413d5bd4.png';
import { RELEASE_NOTES_EVENT } from '../release-notes';

const REPO_URL = 'https://github.com/NEU-Bio-Research-Team/tox-agent';
const BUILD_TIME_LABEL = `${new Date(__APP_BUILD_TIME__).toLocaleString('en-GB', {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
  timeZone: 'UTC',
})} UTC`;
const BUILD_LABEL = `v${__APP_VERSION__} • ${BUILD_TIME_LABEL}`;

export function Navbar() {
  const [theme, setTheme] = useState<'dark' | 'light'>('light');

  useEffect(() => {
    // Apply theme class to document
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  const openReleaseNotes = () => {
    window.dispatchEvent(new Event(RELEASE_NOTES_EVENT));
  };

  return (
    <nav className="sticky top-0 z-50 border-b backdrop-blur-xl bg-[var(--bg)]/80" style={{ borderColor: 'var(--border)' }}>
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo and App Name */}
        <div className="flex items-center gap-3">
          <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <img src={logoImage} alt="ToxAgent Logo" className="h-8" />
          </Link>
          <button
            type="button"
            onClick={openReleaseNotes}
            className="hidden sm:inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-medium tracking-wide uppercase transition-colors"
            style={{
              borderColor: 'var(--border)',
              color: 'var(--text-muted)',
              backgroundColor: 'color-mix(in srgb, var(--surface) 82%, transparent)',
            }}
            title={`${BUILD_LABEL} • Click to view release notes`}
          >
            <span className="mr-2">v{__APP_VERSION__}</span>
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

        {/* Nav Links */}
        <div className="flex items-center gap-6">
          <Link 
            to="/about" 
            className="text-sm font-medium transition-colors hover:underline"
            style={{ color: 'var(--text-muted)' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            About
          </Link>
          <a 
            href={REPO_URL}
            target="_blank" 
            rel="noopener noreferrer"
            className="text-sm font-medium flex items-center gap-1 transition-colors hover:underline"
            style={{ color: 'var(--text-muted)' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            GitHub
            <svg className="w-3 h-3" viewBox="0 0 12 12" fill="none">
              <path d="M10 2L2 10M10 2H4M10 2V8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </a>
          <Link 
            to="/settings" 
            className="text-sm font-medium transition-colors hover:underline"
            style={{ color: 'var(--text-muted)' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            Settings
          </Link>
          
          <div style={{ width: '1px', height: '16px', backgroundColor: 'var(--border)' }} />
          
          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="w-8 h-8 rounded-full"
            style={{ color: 'var(--text-muted)' }}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? (
              <Sun className="w-4 h-4 transition-transform rotate-0 hover:rotate-90" />
            ) : (
              <Moon className="w-4 h-4 transition-transform rotate-0 hover:-rotate-90" />
            )}
          </Button>
        </div>
      </div>
    </nav>
  );
}