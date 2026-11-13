import { type FormEvent, useState } from 'react';
import { Button } from '../../shared/ui/button';
import { Input } from '../../shared/ui/input';
import { Label } from '../../shared/ui/label';
import { type FirebaseAuthConfig, signInWithPassword, signUpWithPassword } from './firebaseAuth';
import { GoogleSignInError, googleSignInAvailable, signInWithGoogle } from './googleSignIn';

function GoogleMark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 48 48" className="h-4 w-4">
      <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9.1 3.6l6.8-6.8C35.8 2.4 30.3 0 24 0 14.6 0 6.5 5.4 2.6 13.2l7.9 6.2C12.4 13.7 17.7 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.7c-.6 3-2.3 5.5-4.8 7.2l7.5 5.8c4.4-4.1 7.1-10.1 7.1-17.5z" />
      <path fill="#FBBC05" d="M10.5 28.6c-.5-1.5-.8-3-.8-4.6s.3-3.1.8-4.6l-7.9-6.2C.9 16.4 0 20.1 0 24s.9 7.6 2.6 10.8l7.9-6.2z" />
      <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.5-5.8c-2.1 1.4-4.9 2.3-8.4 2.3-6.3 0-11.6-4.2-13.5-9.9l-7.9 6.2C6.5 42.6 14.6 48 24 48z" />
    </svg>
  );
}

/** Email and password against the project's existing accounts. Success is
 * announced by `firebaseAuth`; the gate that rendered this re-reads the token. */
export function FirebaseSignIn({ config }: { config: FirebaseAuthConfig }) {
  const [mode, setMode] = useState<'sign-in' | 'sign-up'>('sign-in');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canUseGoogle = googleSignInAvailable(config);

  const google = async () => {
    setBusy(true);
    setError(null);
    try {
      await signInWithGoogle(config);
    } catch (cause) {
      // Closing the window is a choice, not a failure: say nothing.
      if (!(cause instanceof GoogleSignInError && cause.cancelled)) {
        setError(cause instanceof Error ? cause.message : 'Đăng nhập bằng Google thất bại.');
      }
      setBusy(false);
    }
  };

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
    <div className="flex min-h-screen items-center justify-center px-6 bg-canvas-subtle">
      <form
        onSubmit={(event) => void submit(event)}
        className="w-full max-w-sm rounded-2xl border p-8"
        style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--line)' }}
      >
        <h1 className="mb-1 text-lg font-semibold text-ink">
          {mode === 'sign-in' ? 'Đăng nhập ToxAgent' : 'Tạo tài khoản ToxAgent'}
        </h1>
        <p className="mb-6 text-sm text-ink-secondary">
          {mode === 'sign-in' ? 'Dùng tài khoản ToxAgent bạn đã có.' : 'Mật khẩu cần ít nhất 6 ký tự.'}
        </p>
        {canUseGoogle && (
          <>
            <Button type="button" variant="outline" className="w-full gap-2" disabled={busy} onClick={() => void google()}>
              <GoogleMark />
              Tiếp tục với Google
            </Button>
            <div className="my-5 flex items-center gap-3 text-xs text-ink-tertiary">
              <span className="h-px flex-1" style={{ backgroundColor: 'var(--line)' }} />
              hoặc
              <span className="h-px flex-1" style={{ backgroundColor: 'var(--line)' }} />
            </div>
          </>
        )}
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
          <p role="alert" className="mt-3 text-sm text-danger">
            {error}
          </p>
        )}
        <Button type="submit" className="mt-6 w-full" disabled={busy}>
          {busy ? 'Đang xử lý…' : mode === 'sign-in' ? 'Đăng nhập' : 'Tạo tài khoản'}
        </Button>
        <button
          type="button"
          className="mt-4 w-full text-center text-sm underline-offset-4 hover:underline"
          style={{ color: 'var(--ink-secondary)' }}
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
