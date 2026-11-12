import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getToken } from '../../shared/api/client';
import { firebaseAuthConfig } from './firebaseAuth';
import { GoogleSignInError, googleSignInAvailable, signInWithGoogle } from './googleSignIn';

const sdk = vi.hoisted(() => ({
  signInWithPopup: vi.fn(),
  signOut: vi.fn(),
  initializeAuth: vi.fn(),
  getAuth: vi.fn(),
  initializeApp: vi.fn(),
  getApps: vi.fn(),
}));

vi.mock('firebase/app', () => ({ initializeApp: sdk.initializeApp, getApps: sdk.getApps }));
vi.mock('firebase/auth', () => ({
  initializeAuth: sdk.initializeAuth,
  getAuth: sdk.getAuth,
  inMemoryPersistence: { type: 'NONE' },
  browserPopupRedirectResolver: { kind: 'popup' },
  GoogleAuthProvider: class {
    providerId = 'google.com';
  },
  signInWithPopup: sdk.signInWithPopup,
  signOut: sdk.signOut,
}));

const CONFIG = { apiKey: 'web-key', projectId: 'tox-agent' };

describe('signInWithGoogle', () => {
  beforeEach(() => {
    localStorage.clear();
    Object.values(sdk).forEach((fn) => fn.mockReset());
    sdk.getApps.mockReturnValue([]);
    sdk.initializeApp.mockReturnValue({ name: 'toxagent-google' });
    sdk.initializeAuth.mockReturnValue({ app: 'auth' });
    sdk.signOut.mockResolvedValue(undefined);
  });
  afterEach(() => vi.useRealTimers());

  it('is offered only when the build names the project', () => {
    expect(googleSignInAvailable({ apiKey: 'k' })).toBe(false);
    expect(googleSignInAvailable(CONFIG)).toBe(true);
    expect(firebaseAuthConfig({ VITE_FIREBASE_API_KEY: 'k', VITE_FIREBASE_PROJECT_ID: ' tox-agent ' } as unknown as ImportMetaEnv))
      .toEqual({ apiKey: 'k', projectId: 'tox-agent' });
  });

  it('hands the popup\'s tokens to the app session and leaves nothing in the SDK', async () => {
    sdk.signInWithPopup.mockResolvedValue({
      user: {
        refreshToken: 'refresh-1',
        getIdTokenResult: async () => ({ token: 'id-google', expirationTime: new Date(Date.now() + 3_600_000).toUTCString() }),
      },
    });

    await signInWithGoogle(CONFIG);

    expect(getToken()).toBe('id-google');
    expect(localStorage.getItem('toxagent.firebase.refresh_token')).toBe('refresh-1');
    const [, options] = sdk.initializeAuth.mock.calls[0];
    expect(options.persistence).toEqual({ type: 'NONE' });
    expect(sdk.initializeApp.mock.calls[0][0]).toMatchObject({
      apiKey: 'web-key', projectId: 'tox-agent', authDomain: 'tox-agent.firebaseapp.com',
    });
    expect(sdk.signOut).toHaveBeenCalledOnce();
  });

  it('reuses the SDK instance on a second press instead of failing to initialise twice', async () => {
    sdk.initializeAuth.mockImplementationOnce(() => { throw { code: 'auth/already-initialized' }; });
    sdk.getAuth.mockReturnValue({ app: 'existing' });
    sdk.signInWithPopup.mockResolvedValue({
      user: { refreshToken: 'r', getIdTokenResult: async () => ({ token: 'id-2', expirationTime: new Date(Date.now() + 3_600_000).toUTCString() }) },
    });
    await signInWithGoogle(CONFIG);
    expect(sdk.signInWithPopup.mock.calls[0][0]).toEqual({ app: 'existing' });
    expect(getToken()).toBe('id-2');
  });

  it('treats a closed window as a choice, not a failure', async () => {
    sdk.signInWithPopup.mockRejectedValue({ code: 'auth/popup-closed-by-user' });
    const error = await signInWithGoogle(CONFIG).catch((cause) => cause);
    expect(error).toBeInstanceOf(GoogleSignInError);
    expect(error.cancelled).toBe(true);
    expect(getToken()).toBeNull();
    expect(sdk.signOut).toHaveBeenCalledOnce();
  });

  it.each([
    ['auth/popup-blocked', /chặn cửa sổ/],
    ['auth/operation-not-allowed', /chưa được bật/],
    ['auth/unauthorized-domain', /chưa được Firebase cho phép/],
    ['auth/something-new', /thất bại/],
  ])('explains %s in words a person can act on', async (code, message) => {
    sdk.signInWithPopup.mockRejectedValue({ code });
    const error = await signInWithGoogle(CONFIG).catch((cause) => cause);
    expect(error.cancelled).toBe(false);
    expect(error.message).toMatch(message);
  });

  it('refuses without a project instead of opening a popup that cannot work', async () => {
    await expect(signInWithGoogle({ apiKey: 'k' })).rejects.toMatchObject({ code: 'not-configured' });
    expect(sdk.signInWithPopup).not.toHaveBeenCalled();
  });
});
