import { useState } from 'react';
import { useNavigate, Link } from 'react-router';
import { useAuth } from '../components/contexts/auth-context';
import { Button } from '../components/ui/button';
import { AlertCircle } from 'lucide-react';
import logoImage from '../../assets/logo-tox.png';

export function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const success = await login(email, password);
    
    if (success) {
      navigate('/analyze');
    } else {
      setError('Invalid email or password');
    }
    
    setLoading(false);
  };

  return (
    <div 
      className="min-h-screen flex items-center justify-center px-6"
      style={{ backgroundColor: 'var(--bg)' }}
    >
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <img src={logoImage} alt="ToxAgent Logo" className="h-16 mx-auto mb-4" />
          <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text)' }}>
            Welcome back
          </h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Sign in to your ToxAgent account
          </p>
        </div>

        {/* Login Form */}
        <div 
          className="rounded-xl p-8"
          style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
        >
          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <div 
                className="flex items-center gap-2 p-3 rounded-lg text-sm"
                style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', color: 'var(--accent-red)' }}
              >
                <AlertCircle className="w-4 h-4" />
                {error}
              </div>
            )}

            <div>
              <label 
                htmlFor="email" 
                className="block text-sm font-medium mb-2"
                style={{ color: 'var(--text)' }}
              >
                Email
              </label>
              <input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg border outline-none transition-colors"
                style={{ 
                  backgroundColor: 'var(--surface-alt)',
                  borderColor: 'var(--border)',
                  color: 'var(--text)'
                }}
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label 
                htmlFor="password" 
                className="block text-sm font-medium mb-2"
                style={{ color: 'var(--text)' }}
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg border outline-none transition-colors"
                style={{ 
                  backgroundColor: 'var(--surface-alt)',
                  borderColor: 'var(--border)',
                  color: 'var(--text)'
                }}
                placeholder="••••••••"
              />
            </div>

            <Button
              type="submit"
              disabled={loading}
              className="w-full py-2.5"
              style={{ backgroundColor: 'var(--accent-blue)', color: '#ffffff' }}
            >
              {loading ? 'Signing in...' : 'Sign in'}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Don't have an account?{' '}
              <Link 
                to="/register" 
                className="font-medium hover:underline"
                style={{ color: 'var(--accent-blue)' }}
              >
                Create account
              </Link>
            </p>
          </div>
        </div>

        {/* Demo Credentials */}
        <div 
          className="mt-6 p-4 rounded-lg text-center"
          style={{ backgroundColor: 'var(--surface-alt)', border: '1px solid var(--border)' }}
        >
          <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
            Demo credentials:
          </p>
          <p className="text-xs font-mono" style={{ color: 'var(--text-faint)' }}>
            Email: demo@toxagent.ai • Password: demo123
          </p>
        </div>
      </div>
    </div>
  );
}
