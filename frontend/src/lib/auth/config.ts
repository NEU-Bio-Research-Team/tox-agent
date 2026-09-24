/**
 * The identity provider this build talks to, if any (K12).
 *
 * Read from build-time environment because the values are public by
 * definition — a client id, a redirect URI and endpoints the browser is about
 * to navigate to. Nothing secret is here; PKCE exists precisely so a browser
 * client needs no secret.
 *
 * All four must be present. Half-configured is refused for the same reason the
 * server refuses it: a partially wired login is worse than an absent one,
 * because it looks like it should work.
 */
import type { OidcConfig } from './pkce';

export function oidcConfig(env: ImportMetaEnv = import.meta.env): OidcConfig | null {
  const authorizationEndpoint = (env.VITE_OIDC_AUTHORIZATION_ENDPOINT ?? '').trim();
  const tokenEndpoint = (env.VITE_OIDC_TOKEN_ENDPOINT ?? '').trim();
  const clientId = (env.VITE_OIDC_CLIENT_ID ?? '').trim();
  if (!authorizationEndpoint || !tokenEndpoint || !clientId) return null;
  return {
    authorizationEndpoint,
    tokenEndpoint,
    clientId,
    // Defaults to this origin, which is what a single-page app almost always
    // wants and what the provider will have been given.
    redirectUri: (env.VITE_OIDC_REDIRECT_URI ?? '').trim() || window.location.origin + '/',
    scope: (env.VITE_OIDC_SCOPE ?? '').trim() || 'openid profile email',
  };
}
