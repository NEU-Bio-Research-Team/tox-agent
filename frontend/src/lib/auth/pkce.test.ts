import { describe, expect, it, vi } from 'vitest';
import {
  LoginError,
  type OidcConfig,
  beginLogin,
  challengeFor,
  completeLogin,
  createVerifier,
  isCallback,
} from './pkce';

const config: OidcConfig = {
  authorizationEndpoint: 'https://idp.example.com/authorize',
  tokenEndpoint: 'https://idp.example.com/token',
  clientId: 'toxagent-web',
  redirectUri: 'https://app.example.com/callback',
  scope: 'openid profile',
};

/** sessionStorage stand-in, so a test never depends on the real one's state. */
function storage(): Storage {
  const map = new Map<string, string>();
  return {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    getItem: (k: string) => map.get(k) ?? null,
    key: (i: number) => [...map.keys()][i] ?? null,
    removeItem: (k: string) => void map.delete(k),
    setItem: (k: string, v: string) => void map.set(k, v),
  } as Storage;
}

describe('the verifier', () => {
  it('is 43 base64url characters, the shortest RFC 7636 allows', () => {
    const verifier = createVerifier();
    expect(verifier).toHaveLength(43);
    expect(verifier).toMatch(/^[A-Za-z0-9\-_]+$/);
  });

  it('is different every time', () => {
    const seen = new Set(Array.from({ length: 50 }, createVerifier));
    expect(seen.size).toBe(50);
  });

  it('hashes to a challenge that is not the verifier itself', async () => {
    // The `plain` method sends the verifier and secures nothing; only S256
    // makes an intercepted authorization code useless on its own.
    const verifier = createVerifier();
    const challenge = await challengeFor(verifier);
    expect(challenge).not.toBe(verifier);
    expect(challenge).toMatch(/^[A-Za-z0-9\-_]+$/);
    expect(await challengeFor(verifier)).toBe(challenge);
  });

  it('matches the RFC 7636 appendix B vector', async () => {
    expect(await challengeFor('dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk')).toBe(
      'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM',
    );
  });
});

describe('beginLogin', () => {
  it('asks for a code with an S256 challenge and never sends the verifier', async () => {
    const store = storage();
    const url = new URL(await beginLogin(config, store));
    const parameters = url.searchParams;

    expect(url.origin + url.pathname).toBe(config.authorizationEndpoint);
    expect(parameters.get('response_type')).toBe('code');
    expect(parameters.get('code_challenge_method')).toBe('S256');
    expect(parameters.get('client_id')).toBe('toxagent-web');
    expect(parameters.get('redirect_uri')).toBe(config.redirectUri);
    expect(parameters.get('scope')).toBe('openid profile');

    const verifier = store.getItem('toxagent.pkce_verifier');
    expect(verifier).toBeTruthy();
    expect(url.toString()).not.toContain(verifier as string);
    expect(parameters.get('code_challenge')).toBe(await challengeFor(verifier as string));
  });

  it('sends a state it kept, so a login this tab did not start is detectable', async () => {
    const store = storage();
    const parameters = new URL(await beginLogin(config, store)).searchParams;
    expect(parameters.get('state')).toBe(store.getItem('toxagent.pkce_state'));
    expect(parameters.get('state')).not.toBe(store.getItem('toxagent.pkce_verifier'));
  });
});

describe('completeLogin', () => {
  async function started() {
    const store = storage();
    const parameters = new URL(await beginLogin(config, store)).searchParams;
    return { store, state: parameters.get('state') as string };
  }

  function tokenEndpoint(body: unknown, status = 200) {
    return vi.fn(async (_url: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(body), {
        status,
        headers: { 'content-type': 'application/json' },
      }),
    );
  }

  it('exchanges the code with the verifier and returns the access token', async () => {
    const { store, state } = await started();
    const verifier = store.getItem('toxagent.pkce_verifier');
    const fetchImpl = tokenEndpoint({ access_token: 'at-1', id_token: 'it-1' });

    const token = await completeLogin(
      config,
      `https://app.example.com/callback?code=abc&state=${state}`,
      { storage: store, fetchImpl: fetchImpl as unknown as typeof fetch },
    );

    expect(token).toBe('at-1');
    const sent = new URLSearchParams(fetchImpl.mock.calls[0][1]?.body as string);
    expect(sent.get('grant_type')).toBe('authorization_code');
    expect(sent.get('code')).toBe('abc');
    expect(sent.get('code_verifier')).toBe(verifier);
    expect(sent.get('client_id')).toBe('toxagent-web');
  });

  it('refuses a callback whose state does not match, without exchanging anything', async () => {
    const { store } = await started();
    const fetchImpl = tokenEndpoint({ access_token: 'at-1' });
    await expect(
      completeLogin(config, 'https://app.example.com/callback?code=abc&state=someone-elses', {
        storage: store,
        fetchImpl: fetchImpl as unknown as typeof fetch,
      }),
    ).rejects.toBeInstanceOf(LoginError);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('refuses a callback when no login was started here', async () => {
    const fetchImpl = tokenEndpoint({ access_token: 'at-1' });
    await expect(
      completeLogin(config, 'https://app.example.com/callback?code=abc&state=anything', {
        storage: storage(),
        fetchImpl: fetchImpl as unknown as typeof fetch,
      }),
    ).rejects.toThrow(/did not start here/);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('does not leave a verifier behind for a second exchange', async () => {
    const { store, state } = await started();
    await completeLogin(config, `https://app.example.com/callback?code=abc&state=${state}`, {
      storage: store,
      fetchImpl: tokenEndpoint({ access_token: 'at-1' }) as unknown as typeof fetch,
    });
    expect(store.getItem('toxagent.pkce_verifier')).toBeNull();
    expect(store.getItem('toxagent.pkce_state')).toBeNull();
  });

  it('clears the verifier even when the exchange fails', async () => {
    const { store, state } = await started();
    await expect(
      completeLogin(config, `https://app.example.com/callback?code=abc&state=${state}`, {
        storage: store,
        fetchImpl: tokenEndpoint({ error: 'invalid_grant' }, 400) as unknown as typeof fetch,
      }),
    ).rejects.toThrow(/refused the exchange \(400\)/);
    expect(store.getItem('toxagent.pkce_verifier')).toBeNull();
  });

  it("passes on the provider's own description of a refusal", async () => {
    const { store, state } = await started();
    await expect(
      completeLogin(
        config,
        `https://app.example.com/callback?error=access_denied&error_description=User+cancelled&state=${state}`,
        { storage: store, fetchImpl: tokenEndpoint({}) as unknown as typeof fetch },
      ),
    ).rejects.toThrow('User cancelled');
  });

  it('treats a response with no access token as a failure, not as a login', async () => {
    const { store, state } = await started();
    await expect(
      completeLogin(config, `https://app.example.com/callback?code=abc&state=${state}`, {
        storage: store,
        fetchImpl: tokenEndpoint({ id_token: 'it-1' }) as unknown as typeof fetch,
      }),
    ).rejects.toThrow(/no access token/);
  });
});

describe('isCallback', () => {
  it.each([
    ['https://app.example.com/callback?code=abc&state=s', true],
    ['https://app.example.com/callback?error=access_denied', true],
    ['https://app.example.com/', false],
    ['https://app.example.com/?other=1', false],
  ])('%s -> %s', (url, expected) => {
    expect(isCallback(url)).toBe(expected);
  });
});
