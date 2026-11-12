import { describe, expect, it } from 'vitest';
import { identityFromToken } from './identity';

function jwt(claims: Record<string, unknown>): string {
  const encode = (value: unknown) =>
    btoa(String.fromCharCode(...new TextEncoder().encode(JSON.stringify(value))))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  return `${encode({ alg: 'RS256' })}.${encode(claims)}.signature`;
}

describe('identityFromToken', () => {
  it('is null without a token', () => {
    expect(identityFromToken(null)).toBeNull();
  });

  it('reads the name and email claims of a JWT, including non-ASCII names', () => {
    expect(identityFromToken(jwt({ name: 'Nguyễn Văn A', email: 'a@example.org' }))).toEqual({
      name: 'Nguyễn Văn A',
      email: 'a@example.org',
    });
  });

  it('names nobody for a static token or an unreadable payload', () => {
    expect(identityFromToken('dev-local')).toEqual({ name: null, email: null });
    expect(identityFromToken('a.b.c')).toEqual({ name: null, email: null });
  });
});
