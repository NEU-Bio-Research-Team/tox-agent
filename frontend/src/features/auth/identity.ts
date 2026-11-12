/** Who the navigation bar says is signed in.
 *
 * Read from the token's own claims when it is a JWT, for display only: the
 * control plane verifies the token on every request, and nothing here grants
 * or denies anything. A static development token has no claims, so the bar
 * shows a neutral label instead of inventing a name. */
export interface DisplayIdentity {
  name: string | null;
  email: string | null;
}

export function identityFromToken(token: string | null): DisplayIdentity | null {
  if (!token) return null;
  const parts = token.split('.');
  if (parts.length !== 3) return { name: null, email: null };
  try {
    const base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4);
    const bytes = Uint8Array.from(atob(padded), (char) => char.charCodeAt(0));
    const claims = JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
    const text = (value: unknown) => (typeof value === 'string' && value.trim() ? value.trim() : null);
    return {
      name: text(claims.name) ?? text(claims.preferred_username),
      email: text(claims.email),
    };
  } catch {
    return { name: null, email: null };
  }
}
