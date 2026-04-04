import { Moon, Sun, Github, User, LogOut } from 'lucide-react';
import { Button } from './ui/button';
import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router';
import { useAuth } from './contexts/auth-context';
import logoImage from '../../../assets/logo-tox.png';

export function Navbar() {
  const [theme, setTheme] = useState<'dark' | 'light'>('light');
  const [showUserMenu, setShowUserMenu] = useState(false);
  const { user, logout, isAuthenticated } = useAuth();
  const navigate = useNavigate();

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

  const handleLogout = () => {
    logout();
    setShowUserMenu(false);
    navigate('/login');
  };

  return (
    <nav className="sticky top-0 z-50 border-b backdrop-blur-xl bg-[var(--bg)]/80" style={{ borderColor: 'var(--border)' }}>
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo and App Name */}
        <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <img src={logoImage} alt="ToxAgent Logo" className="h-8" />
        </Link>

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
            href="https://github.com" 
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

          {/* User Menu */}
          {isAuthenticated ? (
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-[var(--surface-alt)] transition-colors"
                style={{ color: 'var(--text)' }}
              >
                <User className="w-4 h-4" />
                <span className="text-sm font-medium">{user?.name}</span>
              </button>
              
              {showUserMenu && (
                <>
                  <div 
                    className="fixed inset-0 z-40"
                    onClick={() => setShowUserMenu(false)}
                  />
                  <div 
                    className="absolute right-0 top-full mt-2 w-48 rounded-lg shadow-lg border z-50"
                    style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
                  >
                    <div className="p-3 border-b" style={{ borderColor: 'var(--border)' }}>
                      <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>{user?.name}</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{user?.email}</p>
                    </div>
                    <button
                      onClick={handleLogout}
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-[var(--surface-alt)] transition-colors"
                      style={{ color: 'var(--accent-red)' }}
                    >
                      <LogOut className="w-4 h-4" />
                      Logout
                    </button>
                  </div>
                </>
              )}
            </div>
          ) : (
            <Link to="/login">
              <Button
                size="sm"
                style={{ backgroundColor: 'var(--accent-blue)', color: '#ffffff' }}
              >
                Sign in
              </Button>
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
}
