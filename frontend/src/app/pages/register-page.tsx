import { useState } from 'react';
import { useNavigate, Link } from 'react-router';
import { useAuth } from '../components/contexts/auth-context';
import { Button } from '../components/ui/button';
import { AlertCircle, CheckCircle2 } from 'lucide-react';
import logoImage from '../../assets/logo-tox.png';

export function RegisterPage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);
    const success = await register(email, password, name);
    
    if (success) {
      navigate('/analyze');
    } else {
      setError('Email already exists');
    }
    
    setLoading(false);
  };

  return (
    <div 
      className="min-h-screen flex items-center justify-center px-6 py-12"
      style={{ backgroundColor: 'var(--bg)' }}
    >
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <img src={logoImage} alt="ToxAgent Logo" className="h-16 mx-auto mb-4" />
          <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text)' }}>
            Create your account
          </h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Start analyzing molecular toxicity with AI
          </p>
        </div>

        {/* Registration Form */}
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
                htmlFor="name" 
                className="block text-sm font-medium mb-2"
                style={{ color: 'var(--text)' }}
              >
                Full Name
              </label>
              <input
                id="name"
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg border outline-none transition-colors"
                style={{ 
                  backgroundColor: 'var(--surface-alt)',
                  borderColor: 'var(--border)',
                  color: 'var(--text)'
                }}
                placeholder="John Doe"
              />
            </div>

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

            <div>
              <label 
                htmlFor="confirmPassword" 
                className="block text-sm font-medium mb-2"
                style={{ color: 'var(--text)' }}
              >
                Confirm Password
              </label>
              <input
                id="confirmPassword"
                type="password"
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
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
              {loading ? 'Creating account...' : 'Create account'}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Already have an account?{' '}
              <Link 
                to="/login" 
                className="font-medium hover:underline"
                style={{ color: 'var(--accent-blue)' }}
              >
                Sign in
              </Link>
            </p>
          </div>
        </div>

        {/* Features */}
        <div className="mt-6 space-y-2">
          {['Fast toxicity predictions', 'Detailed molecular analysis', 'Comprehensive reports'].map((feature) => (
            <div key={feature} className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
              <CheckCircle2 className="w-4 h-4" style={{ color: 'var(--accent-green)' }} />
              {feature}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
