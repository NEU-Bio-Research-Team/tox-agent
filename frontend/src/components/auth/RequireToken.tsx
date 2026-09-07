import { type FormEvent, type ReactNode, useState } from 'react';
import { getToken, setToken } from '../../lib/api/client';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';

/**
 * D-2: the control plane's only auth path today is a Bearer token —
 * `StaticTokenAuth` for internal alpha, refused outright in production by
 * `SecuritySettings`. There is no signup/login flow to build against yet, so
 * this gate is the honest MVP substitute: paste the token an operator was
 * given, keep it in localStorage, done. Swapping in an OIDC redirect later
 * changes only what sits behind this component, not anything that reads
 * `getToken()`.
 */
export function RequireToken({ children }: { children: ReactNode }) {
  const [token, setLocalToken] = useState(() => getToken());
  const [draft, setDraft] = useState('');
  const [error, setError] = useState<string | null>(null);

  if (token) return <>{children}</>;

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed) {
      setError('Nhập access token trước khi tiếp tục.');
      return;
    }
    setToken(trimmed);
    setLocalToken(trimmed);
  };

  return (
    <div
      className="flex min-h-screen items-center justify-center px-6"
      style={{ backgroundColor: 'var(--bg)' }}
    >
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm rounded-2xl border p-8"
        style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
      >
        <h1 className="mb-1 text-lg font-semibold" style={{ color: 'var(--text)' }}>
          Kết nối ToxAgent
        </h1>
        <p className="mb-6 text-sm" style={{ color: 'var(--text-muted)' }}>
          Dán access token được cấp cho phiên internal alpha.
        </p>
        <div className="space-y-2">
          <Label htmlFor="token">Access token</Label>
          <Input
            id="token"
            type="password"
            autoFocus
            value={draft}
            onChange={(event) => {
              setDraft(event.target.value);
              setError(null);
            }}
            placeholder="dev-local"
          />
        </div>
        {error && (
          <p className="mt-2 text-sm" style={{ color: 'var(--accent-red)' }}>
            {error}
          </p>
        )}
        <Button type="submit" className="mt-6 w-full">
          Kết nối
        </Button>
      </form>
    </div>
  );
}
