import { ApiError, type ErrorBody } from './types';

function resolveBaseUrl(): string {
  const configured = (import.meta.env.VITE_API_BASE_URL ?? '').trim();
  if (!configured) return '';
  const normalized = configured.replace(/\/$/, '');
  if (typeof window === 'undefined') return normalized;

  // Safety net mirrored from the previous frontend: a production bundle that
  // accidentally ships a localhost API URL should fall through to a relative
  // path rather than pointing a real user's browser at the developer's box.
  const hostname = window.location.hostname;
  const isHosted = hostname !== 'localhost' && hostname !== '127.0.0.1';
  const isLocalApiUrl = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(normalized);
  if (isHosted && isLocalApiUrl) return '';
  return normalized;
}

export const API_BASE_URL = resolveBaseUrl();

const TOKEN_STORAGE_KEY = 'toxagent.bearer_token';

/**
 * D-2: static bearer token for internal alpha, entered by the operator and
 * kept in localStorage. `SecuritySettings` already refuses `TOXAGENT_STATIC_TOKENS`
 * in production, so there is no path from this shim into a production deploy;
 * swapping in an OIDC-issued token later needs no change on this side, only a
 * different source for the string.
 */
export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_STORAGE_KEY, token);
    else localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    // localStorage unavailable (private mode, quota) — token just won't persist.
  }
}

function authHeaders(): HeadersInit {
  const token = getToken();
  return token ? { authorization: `Bearer ${token}` } : {};
}

async function parseErrorBody(response: Response): Promise<ErrorBody> {
  try {
    const body = (await response.json()) as ErrorBody;
    if (body?.error?.code) return body;
  } catch {
    // fall through to the synthetic envelope below
  }
  return {
    error: {
      // A non-envelope 404 nearly always means the browser reached an older
      // control-plane (or a proxy pointing at one), not that the predictor
      // failed. Preserve that distinction so an incompatible rollout is
      // diagnosable from the UI.
      code: response.status === 401
        ? 'unauthenticated'
        : response.status === 404
          ? 'api_route_not_found'
          : 'internal_error',
      message: response.status === 404
        ? 'API route was not found (the frontend and control-plane versions may differ)'
        : `unexpected response (HTTP ${response.status})`,
      retryable: false,
      details: {},
    },
  };
}

export interface RequestOptions {
  method?: 'GET' | 'POST';
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined | null>;
  signal?: AbortSignal;
}

function buildUrl(path: string, query?: RequestOptions['query']): string {
  const url = new URL(`${API_BASE_URL}${path}`, window.location.origin);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined || value === null) continue;
      url.searchParams.set(key, String(value));
    }
  }
  // A relative API_BASE_URL ("") plus a relative window.location.origin base
  // still yields the correct path; strip the origin back off in that case so
  // fetch() doesn't force an absolute URL where a relative one was intended.
  return API_BASE_URL ? url.toString() : `${url.pathname}${url.search}`;
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await fetch(buildUrl(path, options.query), {
    method: options.method ?? 'GET',
    headers: {
      ...authHeaders(),
      ...(options.body !== undefined ? { 'content-type': 'application/json' } : {}),
    },
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
    signal: options.signal,
  });

  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorBody(response));
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export { buildUrl };
