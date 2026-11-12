import { type FormEvent, useState } from 'react';
import { Button } from '../../shared/ui/button';
import { Input } from '../../shared/ui/input';
import { Label } from '../../shared/ui/label';
import { type FirebaseAuthConfig, signInWithPassword, signUpWithPassword } from './firebaseAuth';

/** Email and password against the project's existing accounts. Success is
 * announced by `firebaseAuth`; the gate that rendered this re-reads the token. */
export function FirebaseSignIn({ config }: { config: FirebaseAuthConfig }) {
  const [mode, setMode] = useState<'sign-in' | 'sign-up'>('sign-in');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!email.trim() || !password) {
      setError('Nhập email và mật khẩu.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await (mode === 'sign-in' ? signInWithPassword : signUpWithPassword)(config, email.trim(), password);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Đăng nhập thất bại.');
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center px-6" style={{ backgroundColor: 'var(--bg)' }}>
      <form
        onSubmit={(event) => void submit(event)}
        className="w-full max-w-sm rounded-2xl border p-8"
        style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
      >
        <h1 className="mb-1 text-lg font-semibold" style={{ color: 'var(--text)' }}>
          {mode === 'sign-in' ? 'Đăng nhập ToxAgent' : 'Tạo tài khoản ToxAgent'}
        </h1>
        <p className="mb-6 text-sm" style={{ color: 'var(--text-muted)' }}>
          {mode === 'sign-in' ? 'Dùng tài khoản ToxAgent bạn đã có.' : 'Mật khẩu cần ít nhất 6 ký tự.'}
        </p>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="auth-email">Email</Label>
            <Input
              id="auth-email"
              type="email"
              autoComplete="email"
              autoFocus
              value={email}
              onChange={(event) => {
                setEmail(event.target.value);
                setError(null);
              }}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="auth-password">Mật khẩu</Label>
            <Input
              id="auth-password"
              type="password"
              autoComplete={mode === 'sign-in' ? 'current-password' : 'new-password'}
              value={password}
              onChange={(event) => {
                setPassword(event.target.value);
                setError(null);
              }}
            />
          </div>
        </div>
        {error && (
          <p role="alert" className="mt-3 text-sm" style={{ color: 'var(--accent-red)' }}>
            {error}
          </p>
        )}
        <Button type="submit" className="mt-6 w-full" disabled={busy}>
          {busy ? 'Đang xử lý…' : mode === 'sign-in' ? 'Đăng nhập' : 'Tạo tài khoản'}
        </Button>
        <button
          type="button"
          className="mt-4 w-full text-center text-sm underline-offset-4 hover:underline"
          style={{ color: 'var(--text-muted)' }}
          onClick={() => {
            setMode(mode === 'sign-in' ? 'sign-up' : 'sign-in');
            setError(null);
          }}
        >
          {mode === 'sign-in' ? 'Chưa có tài khoản? Tạo tài khoản' : 'Đã có tài khoản? Đăng nhập'}
        </button>
      </form>
    </div>
  );
}
