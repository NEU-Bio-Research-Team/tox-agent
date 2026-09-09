import { type FormEvent, type ReactNode, useEffect, useState } from 'react';
import { getToken, setToken } from '../../lib/api/client';
import { oidcConfig } from '../../lib/auth/config';
import { completeLogin, isCallback, beginLogin } from '../../lib/auth/pkce';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';

/**
 * D-2, and the swap it anticipated (K12).
 *
 * The paste-a-token gate was written as "the honest MVP substitute" while the
 * control plane's only auth path was `StaticTokenAuth`, with the note that
 * swapping in an OIDC redirect would change only what sits behind this
 * component. The control plane now verifies an identity provider's token and
 * refuses to start in production without one, so this is that swap: when the
 * build has a provider configured, sign-in is authorization code with PKCE;
 * when it does not — local development, where the server accepts the static
 * shim — the paste gate stays, because a login form pointing at nothing is
 * worse than a form that says what it wants.
 *
 * Everything downstream still reads `getToken()`, unchanged.
 */
export function RequireToken({ children }: { children: ReactNode }) {
  const [token, setLocalToken] = useState(() => getToken());
  const [draft, setDraft] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [signingIn, setSigningIn] = useState(false);
  const provider = oidcConfig();

  // Completing a redirect back from the provider. The code is stripped from
  // the address bar afterwards: it is single-use and has been used, and
  // leaving it there puts it in history and in any link the user copies.
  useEffect(() => {
    if (!provider || !isCallback(window.location.href)) return;
    let cancelled = false;
    setSigningIn(true);
    completeLogin(provider, window.location.href)
      .then((accessToken) => {
        if (cancelled) return;
        setToken(accessToken);
        setLocalToken(accessToken);
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(cause instanceof Error ? cause.message : 'Đăng nhập thất bại.');
      })
      .finally(() => {
        if (cancelled) return;
        setSigningIn(false);
        window.history.replaceState({}, '', window.location.pathname);
      });
    return () => {
      cancelled = true;
    };
  }, [provider]);

  if (token) return <>{children}</>;

  if (provider) {
    const startLogin = async () => {
      setError(null);
      setSigningIn(true);
      try {
        window.location.assign(await beginLogin(provider));
      } catch (cause) {
        setSigningIn(false);
        setError(cause instanceof Error ? cause.message : 'Không bắt đầu được đăng nhập.');
      }
    };
    return (
      <div
        className="flex min-h-screen items-center justify-center px-6"
        style={{ backgroundColor: 'var(--bg)' }}
      >
        <div
          className="w-full max-w-sm rounded-2xl border p-8"
          style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
        >
          <h1 className="mb-1 text-lg font-semibold" style={{ color: 'var(--text)' }}>
            Kết nối ToxAgent
          </h1>
          <p className="mb-6 text-sm" style={{ color: 'var(--text-muted)' }}>
            Đăng nhập bằng tài khoản tổ chức của bạn.
          </p>
          {error && (
            <p className="mb-4 text-sm" style={{ color: 'var(--accent-red)' }}>
              {error}
            </p>
          )}
          <Button type="button" className="w-full" disabled={signingIn} onClick={startLogin}>
            {signingIn ? 'Đang đăng nhập…' : 'Đăng nhập'}
          </Button>
        </div>
      </div>
    );
  }

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
