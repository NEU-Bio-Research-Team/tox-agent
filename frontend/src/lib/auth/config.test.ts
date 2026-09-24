import { describe, expect, it } from 'vitest';
import { oidcConfig } from './config';

const full = {
  VITE_OIDC_AUTHORIZATION_ENDPOINT: 'https://idp.example.com/authorize',
  VITE_OIDC_TOKEN_ENDPOINT: 'https://idp.example.com/token',
  VITE_OIDC_CLIENT_ID: 'toxagent-web',
} as unknown as ImportMetaEnv;

describe('oidcConfig', () => {
  it('is null when this build has no provider, so the token gate stays', () => {
    expect(oidcConfig({} as ImportMetaEnv)).toBeNull();
  });

  it.each(['VITE_OIDC_AUTHORIZATION_ENDPOINT', 'VITE_OIDC_TOKEN_ENDPOINT', 'VITE_OIDC_CLIENT_ID'])(
    'is null when %s is missing — half-wired looks like it should work',
    (missing) => {
      const partial = { ...full, [missing]: '' } as unknown as ImportMetaEnv;
      expect(oidcConfig(partial)).toBeNull();
    },
  );

  it('defaults the redirect to this origin and asks for openid', () => {
    const config = oidcConfig(full);
    expect(config?.redirectUri).toBe(window.location.origin + '/');
    expect(config?.scope).toContain('openid');
  });

  it('takes an explicit redirect and scope when given', () => {
    const config = oidcConfig({
      ...full,
      VITE_OIDC_REDIRECT_URI: 'https://app.example.com/cb',
      VITE_OIDC_SCOPE: 'openid api',
    } as unknown as ImportMetaEnv);
    expect(config?.redirectUri).toBe('https://app.example.com/cb');
    expect(config?.scope).toBe('openid api');
  });

  it('ignores whitespace-only values', () => {
    expect(oidcConfig({ ...full, VITE_OIDC_CLIENT_ID: '   ' } as unknown as ImportMetaEnv)).toBeNull();
  });
});
