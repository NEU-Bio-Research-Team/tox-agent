/**
 * Authorization code with PKCE (K12).
 *
 * The control plane now verifies an identity provider's token and, in
 * production, will not start without one. This is the browser half: obtaining
 * that token without a client secret, which a browser cannot keep.
 *
 * Why PKCE rather than the implicit flow, or a bare authorization code: a
 * public client has no secret, so an intercepted authorization code could be
 * redeemed by whoever intercepted it. PKCE binds the code to a random verifier
 * that never leaves this origin — only its SHA-256 goes to the provider — so a
 * stolen code is useless without the verifier.
 *
 * What this module does not do: keep the token. Storage stays in
 * `lib/api/client.ts`, so everything reading `getToken()` is unchanged, which
 * is what `RequireToken`'s comment promised when it called the paste-a-token
 * gate an honest MVP substitute.
 */

/** Config an operator supplies; absent means this deployment has no provider. */
export interface OidcConfig {
  authorizationEndpoint: string;
  tokenEndpoint: string;
  clientId: string;
  redirectUri: string;
  /** `openid` at minimum; a deployment may add its own. */
  scope: string;
}

const VERIFIER_KEY = 'toxagent.pkce_verifier';
const STATE_KEY = 'toxagent.pkce_state';

/**
 * Base64url without padding, which is what RFC 7636 specifies. `btoa` produces
 * standard base64, so the three differing characters are translated here.
 */
function base64UrlEncode(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/**
 * 32 bytes of CSPRNG output, base64url-encoded to 43 characters — the shortest
 * length RFC 7636 allows, and the length it recommends. Deliberately not
 * `Math.random`, which is not a CSPRNG and would make the verifier guessable,
 * defeating the whole exchange.
 */
export function createVerifier(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return base64UrlEncode(bytes);
}

/** S256, never `plain`: `plain` sends the verifier itself and secures nothing. */
export async function challengeFor(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier));
  return base64UrlEncode(new Uint8Array(digest));
}

/**
 * The URL to send the browser to, with the verifier kept for the callback.
 *
 * `state` is a second random value, checked on return: without it, a page the
 * user did not initiate could complete a login as somebody else. It is stored
 * in sessionStorage rather than localStorage so it dies with the tab and does
 * not linger to be replayed.
 */
export async function beginLogin(
  config: OidcConfig,
  storage: Storage = sessionStorage,
): Promise<string> {
  const verifier = createVerifier();
  const state = createVerifier();
  storage.setItem(VERIFIER_KEY, verifier);
  storage.setItem(STATE_KEY, state);

  const parameters = new URLSearchParams({
    response_type: 'code',
    client_id: config.clientId,
    redirect_uri: config.redirectUri,
    scope: config.scope,
    state,
    code_challenge: await challengeFor(verifier),
    code_challenge_method: 'S256',
  });
  return `${config.authorizationEndpoint}?${parameters.toString()}`;
}

export class LoginError extends Error {}

/**
 * Redeem the code the provider sent back.
 *
 * The state check comes first and is unconditional. A mismatch means this
 * callback does not belong to a login this tab started, and the only safe
 * response is to refuse — not to try the exchange and see what happens.
 */
export async function completeLogin(
  config: OidcConfig,
  callbackUrl: string,
  options: { storage?: Storage; fetchImpl?: typeof fetch } = {},
): Promise<string> {
  const storage = options.storage ?? sessionStorage;
  const doFetch = options.fetchImpl ?? fetch;
  const parameters = new URL(callbackUrl).searchParams;

  const error = parameters.get('error');
  if (error) {
    // The provider's own description, if it sent one: "access_denied" alone
    // does not tell an operator whether the user cancelled or the client is
    // misconfigured.
    throw new LoginError(parameters.get('error_description') || error);
  }

  const expectedState = storage.getItem(STATE_KEY);
  const verifier = storage.getItem(VERIFIER_KEY);
  // Cleared before anything can fail, so a verifier is never reusable for a
  // second exchange even if this call throws.
  storage.removeItem(STATE_KEY);
  storage.removeItem(VERIFIER_KEY);

  if (!expectedState || parameters.get('state') !== expectedState) {
    throw new LoginError('This sign-in did not start here.');
  }
  const code = parameters.get('code');
  if (!code || !verifier) throw new LoginError('The sign-in response was incomplete.');

  const response = await doFetch(config.tokenEndpoint, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: config.redirectUri,
      client_id: config.clientId,
      code_verifier: verifier,
    }).toString(),
  });

  if (!response.ok) {
    throw new LoginError(`The identity provider refused the exchange (${response.status}).`);
  }
  const payload = (await response.json()) as { access_token?: string; id_token?: string };
  // The access token, not the id token: the id token describes the user to
  // this app, while the control plane's audience check expects a token minted
  // for the API. Sending the wrong one fails at the server as an audience
  // mismatch, which is the check working.
  const token = payload.access_token;
  if (!token) throw new LoginError('The identity provider returned no access token.');
  return token;
}

/** Whether this URL looks like a provider redirecting back to us. */
export function isCallback(url: string): boolean {
  const parameters = new URL(url).searchParams;
  return parameters.has('code') || parameters.has('error');
}
