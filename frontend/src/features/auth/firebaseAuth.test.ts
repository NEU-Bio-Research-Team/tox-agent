import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getToken, setToken } from '../../shared/api/client';
import {
  AUTH_CHANGED_EVENT,
  firebaseAuthConfig,
  refreshSession,
  signInWithPassword,
  signOut,
  startSessionRefresh,
} from './firebaseAuth';

const CONFIG = { apiKey: 'web-key' };

function respond(status: number, body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } }));
}

describe('firebaseAuth', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    localStorage.clear();
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('is off unless the build names the project key', () => {
    expect(firebaseAuthConfig({} as ImportMetaEnv)).toBeNull();
    expect(firebaseAuthConfig({ VITE_FIREBASE_API_KEY: ' k ' } as unknown as ImportMetaEnv)).toEqual({ apiKey: 'k' });
  });

  it('signs in, keeps the ID token where every request reads it, and announces it', async () => {
    fetchMock.mockReturnValue(respond(200, { idToken: 'id-1', refreshToken: 'r-1', expiresIn: '3600' }));
    const heard = vi.fn();
    window.addEventListener(AUTH_CHANGED_EVENT, heard);

    await signInWithPassword(CONFIG, 'a@example.org', 'secret');

    window.removeEventListener(AUTH_CHANGED_EVENT, heard);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain('accounts:signInWithPassword?key=web-key');
    expect(JSON.parse(init.body)).toEqual({ email: 'a@example.org', password: 'secret', returnSecureToken: true });
    expect(getToken()).toBe('id-1');
    expect(heard).toHaveBeenCalledOnce();
  });

  it('turns a provider refusal into a message a person can act on', async () => {
    fetchMock.mockReturnValue(respond(400, { error: { message: 'INVALID_LOGIN_CREDENTIALS' } }));
    await expect(signInWithPassword(CONFIG, 'a@example.org', 'wrong')).rejects.toThrow('Email hoặc mật khẩu không đúng.');
    expect(getToken()).toBeNull();
  });

  it('reads the code out of a message that carries a detail after it', async () => {
    fetchMock.mockReturnValue(respond(400, { error: { message: 'TOO_MANY_ATTEMPTS_TRY_LATER : Access disabled.' } }));
    await expect(signInWithPassword(CONFIG, 'a@example.org', 'x')).rejects.toThrow('Quá nhiều lần thử');
  });

  it('refreshes with the stored refresh token and adopts the rotated pair', async () => {
    fetchMock.mockReturnValueOnce(respond(200, { idToken: 'id-1', refreshToken: 'r-1', expiresIn: '3600' }));
    await signInWithPassword(CONFIG, 'a@example.org', 'secret');
    fetchMock.mockReturnValueOnce(respond(200, { id_token: 'id-2', refresh_token: 'r-2', expires_in: '3600' }));

    expect(await refreshSession(CONFIG)).toBe(true);

    const [url, init] = fetchMock.mock.calls[1];
    expect(url).toContain('securetoken.googleapis.com/v1/token?key=web-key');
    expect(JSON.parse(init.body)).toEqual({ grant_type: 'refresh_token', refresh_token: 'r-1' });
    expect(getToken()).toBe('id-2');
  });

  it('ends the session when the provider refuses the refresh token', async () => {
    fetchMock.mockReturnValueOnce(respond(200, { idToken: 'id-1', refreshToken: 'r-1', expiresIn: '3600' }));
    await signInWithPassword(CONFIG, 'a@example.org', 'secret');
    fetchMock.mockReturnValueOnce(respond(400, { error: { message: 'USER_DISABLED' } }));

    expect(await refreshSession(CONFIG)).toBe(false);
    expect(getToken()).toBeNull();
  });

  it('keeps the session through a network failure, which is not a refusal', async () => {
    fetchMock.mockReturnValueOnce(respond(200, { idToken: 'id-1', refreshToken: 'r-1', expiresIn: '3600' }));
    await signInWithPassword(CONFIG, 'a@example.org', 'secret');
    fetchMock.mockRejectedValueOnce(new TypeError('offline'));

    await expect(refreshSession(CONFIG)).rejects.toThrow();
    expect(getToken()).toBe('id-1');
  });

  it('renews shortly before the token expires', async () => {
    vi.useFakeTimers();
    fetchMock.mockReturnValueOnce(respond(200, { idToken: 'id-1', refreshToken: 'r-1', expiresIn: '3600' }));
    await signInWithPassword(CONFIG, 'a@example.org', 'secret');
    fetchMock.mockReturnValue(respond(200, { id_token: 'id-2', refresh_token: 'r-2', expires_in: '3600' }));

    const stop = startSessionRefresh(CONFIG);
    await vi.advanceTimersByTimeAsync(54 * 60_000);
    expect(getToken()).toBe('id-1');
    await vi.advanceTimersByTimeAsync(2 * 60_000);
    expect(getToken()).toBe('id-2');
    stop();
  });

  it('signs out of a static-token session too', () => {
    setToken('dev-local');
    signOut();
    expect(getToken()).toBeNull();
  });
});
