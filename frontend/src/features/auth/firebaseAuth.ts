/**
 * Sign-in against the Firebase project the production site's accounts live in.
 *
 * The control plane is a resource server: it verifies a bearer token against
 * an identity provider's published keys and never runs a login flow. A
 * Firebase ID token is such a token (RS256, issuer and audience fixed by the
 * project), so the server needs configuration and no code. What is left is
 * the browser's side, and that is this file: exchange an email and password
 * for an ID token, and keep it fresh — an ID token lives an hour.
 *
 * It talks to Firebase's REST endpoints directly. The SDK would add a
 * dependency and its own persistence for two requests' worth of work, and the
 * rest of the app already has one place a token lives: `setToken`.
 */
import { getToken, setToken } from '../../shared/api/client';

export interface FirebaseAuthConfig {
  /** The project's web API key. It identifies the project to Google and is
   * public by design; it authorises nothing by itself. */
  apiKey: string;
}

export function firebaseAuthConfig(env: ImportMetaEnv = import.meta.env): FirebaseAuthConfig | null {
  const apiKey = (env.VITE_FIREBASE_API_KEY ?? '').trim();
  return apiKey ? { apiKey } : null;
}

const REFRESH_TOKEN_KEY = 'toxagent.firebase.refresh_token';
const EXPIRES_AT_KEY = 'toxagent.firebase.expires_at';
/** Fired on sign-in and sign-out so a mounted gate re-reads the token. */
export const AUTH_CHANGED_EVENT = 'toxagent:auth-changed';

/** Renew this long before expiry, so a request in flight never carries a
 * token that dies on the way. */
const RENEW_MARGIN_MS = 5 * 60_000;
const RETRY_MS = 60_000;

const IDENTITY_URL = 'https://identitytoolkit.googleapis.com/v1/accounts';
const TOKEN_URL = 'https://securetoken.googleapis.com/v1/token';

const MESSAGES: Record<string, string> = {
  EMAIL_NOT_FOUND: 'Email hoặc mật khẩu không đúng.',
  INVALID_PASSWORD: 'Email hoặc mật khẩu không đúng.',
  INVALID_LOGIN_CREDENTIALS: 'Email hoặc mật khẩu không đúng.',
  INVALID_EMAIL: 'Email không hợp lệ.',
  MISSING_PASSWORD: 'Nhập mật khẩu.',
  USER_DISABLED: 'Tài khoản này đã bị vô hiệu hoá.',
  EMAIL_EXISTS: 'Email này đã có tài khoản. Hãy đăng nhập.',
  WEAK_PASSWORD: 'Mật khẩu cần ít nhất 6 ký tự.',
  TOO_MANY_ATTEMPTS_TRY_LATER: 'Quá nhiều lần thử. Hãy thử lại sau.',
  OPERATION_NOT_ALLOWED: 'Đăng nhập bằng mật khẩu chưa được bật cho dự án này.',
};

export class FirebaseAuthError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

function store(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // Storage unavailable: the session lasts until this ID token expires.
  }
}

function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function announce(): void {
  window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}

async function post(url: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new FirebaseAuthError('network', 'Không kết nối được dịch vụ đăng nhập. Kiểm tra mạng rồi thử lại.');
  }
  const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  if (!response.ok) {
    // "WEAK_PASSWORD : Password should be at least 6 characters" — the code
    // is the part before the colon.
    const raw = String((payload.error as { message?: string } | undefined)?.message ?? 'UNKNOWN');
    const code = raw.split(/[ :]/, 1)[0];
    throw new FirebaseAuthError(code, MESSAGES[code] ?? 'Đăng nhập thất bại.');
  }
  return payload;
}

function adopt(idToken: unknown, refreshToken: unknown, expiresIn: unknown): void {
  if (typeof idToken !== 'string' || typeof refreshToken !== 'string') {
    throw new FirebaseAuthError('malformed', 'Dịch vụ đăng nhập trả về phản hồi không hợp lệ.');
  }
  const lifetimeS = Number(expiresIn) > 0 ? Number(expiresIn) : 3600;
  setToken(idToken);
  store(REFRESH_TOKEN_KEY, refreshToken);
  store(EXPIRES_AT_KEY, String(Date.now() + lifetimeS * 1000));
  announce();
}

export async function signInWithPassword(config: FirebaseAuthConfig, email: string, password: string): Promise<void> {
  const body = await post(`${IDENTITY_URL}:signInWithPassword?key=${encodeURIComponent(config.apiKey)}`, {
    email,
    password,
    returnSecureToken: true,
  });
  adopt(body.idToken, body.refreshToken, body.expiresIn);
}

export async function signUpWithPassword(config: FirebaseAuthConfig, email: string, password: string): Promise<void> {
  const body = await post(`${IDENTITY_URL}:signUp?key=${encodeURIComponent(config.apiKey)}`, {
    email,
    password,
    returnSecureToken: true,
  });
  adopt(body.idToken, body.refreshToken, body.expiresIn);
}

/** Drop the session everywhere it is kept. Safe to call in a build with no
 * Firebase configured: it is then exactly `setToken(null)`. */
export function signOut(): void {
  setToken(null);
  store(REFRESH_TOKEN_KEY, null);
  store(EXPIRES_AT_KEY, null);
  announce();
}

/**
 * Trade the refresh token for a new ID token.
 *
 * `true` when the session is good afterwards. A refusal from Firebase — the
 * account was disabled or deleted, or its sessions were revoked — ends the
 * session here rather than leaving a token that the server will keep
 * rejecting. A network failure is not a refusal: the current token stays and
 * the caller tries again.
 */
export async function refreshSession(config: FirebaseAuthConfig): Promise<boolean> {
  const refreshToken = read(REFRESH_TOKEN_KEY);
  if (!refreshToken) return false;
  try {
    const body = await post(`${TOKEN_URL}?key=${encodeURIComponent(config.apiKey)}`, {
      grant_type: 'refresh_token',
      refresh_token: refreshToken,
    });
    adopt(body.id_token, body.refresh_token, body.expires_in);
    return true;
  } catch (cause) {
    if (cause instanceof FirebaseAuthError && cause.code === 'network') throw cause;
    signOut();
    return false;
  }
}

/**
 * Keep the ID token fresh for as long as the page is open. Returns a stop
 * function. Checked again whenever the tab becomes visible, because a timer
 * does not run while a laptop sleeps and the token expires regardless.
 */
export function startSessionRefresh(config: FirebaseAuthConfig): () => void {
  let timer: number | undefined;
  let stopped = false;

  const schedule = (delayMs: number) => {
    window.clearTimeout(timer);
    if (!stopped) timer = window.setTimeout(() => void tick(), Math.max(delayMs, 1000));
  };

  const tick = async () => {
    if (stopped || !read(REFRESH_TOKEN_KEY)) return;
    const expiresAt = Number(read(EXPIRES_AT_KEY) ?? 0);
    const dueIn = expiresAt - RENEW_MARGIN_MS - Date.now();
    if (dueIn > 0 && getToken()) return schedule(dueIn);
    try {
      if (await refreshSession(config)) schedule(Number(read(EXPIRES_AT_KEY)) - RENEW_MARGIN_MS - Date.now());
    } catch {
      schedule(RETRY_MS);
    }
  };

  const onVisible = () => {
    if (document.visibilityState === 'visible') void tick();
  };
  // A sign-in in this tab starts the clock for the token it just received.
  const onAuthChanged = () => {
    if (read(REFRESH_TOKEN_KEY)) schedule(Number(read(EXPIRES_AT_KEY)) - RENEW_MARGIN_MS - Date.now());
    else window.clearTimeout(timer);
  };

  document.addEventListener('visibilitychange', onVisible);
  window.addEventListener(AUTH_CHANGED_EVENT, onAuthChanged);
  void tick();

  return () => {
    stopped = true;
    window.clearTimeout(timer);
    document.removeEventListener('visibilitychange', onVisible);
    window.removeEventListener(AUTH_CHANGED_EVENT, onAuthChanged);
  };
}
