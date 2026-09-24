"""Who is asking.

The control plane authenticates product users. A runtime never authenticates on
a user's behalf: it holds a capability token scoped to one run, which is a
different mechanism handled in ``tools/capability.py`` and deliberately not
interchangeable with this one.

Three providers ship. ``StaticTokenAuth`` is a development shim, refused in
production by ``SecuritySettings``. ``JwksAuth`` verifies an identity
provider's asymmetrically signed token against its published key set, with
issuer and audience required; it is what production uses. ``JwtAuth`` verifies
a symmetrically signed token and is kept for a deployment that runs its own
issuer, but it must be given its own secret and its own issuer and audience —
never the capability signing key, which is what `build_auth` used to hand it.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import jwt
from fastapi import Request

from ..application.policy import Actor
from ..config import SecuritySettings
from ..domain.errors import Unauthenticated


class AuthProvider(Protocol):
    async def authenticate(self, request: Request) -> Actor: ...


def bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise Unauthenticated("a bearer token is required")
    return token.strip()


@dataclass(frozen=True)
class StaticTokenAuth:
    """``TOXAGENT_STATIC_TOKENS=token:subject:role1|role2,...``

    Exists so the stack runs end to end without an identity provider. It is not
    a fallback: a production deployment that sets this fails to start.
    """

    tokens: dict[str, Actor]

    @classmethod
    def from_settings(cls, settings: SecuritySettings) -> "StaticTokenAuth":
        table: dict[str, Actor] = {}
        for entry in settings.static_tokens:
            token, _, rest = entry.partition(":")
            subject, _, roles = rest.partition(":")
            table[token] = Actor(
                subject_id=subject or token,
                roles=frozenset(r for r in roles.split("|") if r),
            )
        return cls(table)

    async def authenticate(self, request: Request) -> Actor:
        actor = self.tokens.get(bearer_token(request))
        if actor is None:
            raise Unauthenticated("unknown token")
        return actor


@dataclass(frozen=True)
class JwtAuth:
    """A symmetric verifier, for a deployment that issues its own tokens.

    `secret` must not be the capability signing secret, and `audience` and
    `issuer` should both be set: without them this accepts any token anything
    signs with that key.
    """

    secret: str
    algorithms: tuple[str, ...] = ("HS256",)
    audience: str | None = None
    issuer: str | None = None
    roles_claim: str = "roles"

    async def authenticate(self, request: Request) -> Actor:
        try:
            claims = jwt.decode(
                bearer_token(request),
                self.secret,
                algorithms=list(self.algorithms),
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["sub", "exp"]},
            )
        except jwt.PyJWTError as exc:
            raise Unauthenticated(f"token rejected: {exc}") from exc
        return Actor(subject_id=str(claims["sub"]), roles=_roles_from(claims, self.roles_claim))


def build_auth(
    settings: SecuritySettings, *, client: httpx.AsyncClient | None = None
) -> AuthProvider:
    """Pick a provider, and never silently pick a weaker one.

    The order matters more than it looks. This used to end in
    `JwtAuth(secret=settings.capability_secret)`: the key a run's capability
    tokens are signed with, verifying no issuer and no audience, deciding who
    a *user* is. Two trust domains on one key, in the deployment shape that
    refuses the development shim — so production got the weakest option
    precisely because it had ruled out the obviously-unsafe one.

    OIDC is now first, and production has nothing to fall back to: `config.py`
    refuses to start without it.
    """
    if settings.oidc_configured:
        return JwksAuth.from_settings(settings, client=client)
    if settings.static_tokens:
        return StaticTokenAuth.from_settings(settings)
    raise ValueError(
        "no authentication configured: set TOXAGENT_OIDC_ISSUER, "
        "TOXAGENT_OIDC_AUDIENCE and TOXAGENT_OIDC_JWKS_URL, or "
        "TOXAGENT_STATIC_TOKENS for local development"
    )


@dataclass
class JwksAuth:
    """Verify a product user token against an identity provider's public keys.

    The control plane is a resource server: it does not run a login flow, it
    checks the access token a browser presents. Three things make that check
    mean something, and the previous `JwtAuth` fallback had none of them.

    *Asymmetric keys.* The provider signs; this server only verifies. There is
    no shared secret that could also mint tokens, which is what let user
    authentication and capability signing collapse onto one key.

    *Issuer and audience, both required.* Without an audience the verifier
    accepts a token the same provider minted for a different service — a valid
    token, for somebody else's API. Without an issuer it accepts one from any
    provider whose key set it happens to fetch.

    *Rotation.* Providers rotate signing keys without announcing it. The key
    set is cached for a bounded time, and an unknown `kid` refetches
    immediately rather than waiting out the TTL, so a rotation is a cache miss
    rather than an outage. A `kid` still unknown after a refetch is rejected:
    fetching per request on an unknown kid would let anyone with a made-up kid
    drive traffic at the provider.
    """

    jwks_url: str
    issuer: str
    audience: str
    #: Asymmetric only. `HS*` here would reintroduce a shared secret, and
    #: `none` is the classic forgery.
    algorithms: tuple[str, ...] = ("RS256", "RS384", "RS512", "ES256", "ES384")
    roles_claim: str = "roles"
    cache_ttl_s: int = 300
    client: httpx.AsyncClient | None = None

    _keys: dict[str, Any] = field(default_factory=dict, repr=False)
    _fetched_at: float = field(default=0.0, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @classmethod
    def from_settings(
        cls, settings: SecuritySettings, *, client: httpx.AsyncClient | None = None
    ) -> "JwksAuth":
        return cls(
            jwks_url=settings.oidc_jwks_url,
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
            roles_claim=settings.oidc_roles_claim,
            cache_ttl_s=settings.oidc_jwks_cache_s,
            client=client,
        )

    async def _fetch(self) -> None:
        client = self.client or httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0))
        owned = self.client is None
        try:
            response = await client.get(self.jwks_url)
            response.raise_for_status()
            document = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise Unauthenticated("the identity provider's key set is unavailable") from exc
        finally:
            if owned:
                await client.aclose()
        keys = {}
        for entry in document.get("keys", []):
            kid = entry.get("kid")
            if not kid:
                continue
            try:
                keys[kid] = jwt.PyJWK(entry).key
            except (jwt.PyJWKError, jwt.InvalidKeyError, KeyError):
                # One unusable entry in a key set is not a reason to reject
                # every token; a set with no usable entry at all is caught by
                # the lookup below.
                continue
        self._keys = keys
        self._fetched_at = time.monotonic()

    async def _key_for(self, kid: str | None):
        if kid is None:
            raise Unauthenticated("token has no key id")
        async with self._lock:
            stale = (time.monotonic() - self._fetched_at) > self.cache_ttl_s
            if kid not in self._keys or stale:
                # An unknown kid refetches immediately: that is what a rotation
                # looks like from here, and waiting out the TTL would be an
                # outage for every token signed with the new key.
                await self._fetch()
            key = self._keys.get(kid)
        if key is None:
            raise Unauthenticated("token is signed by a key this provider does not publish")
        return key

    async def authenticate(self, request: Request) -> Actor:
        token = bearer_token(request)
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except jwt.PyJWTError as exc:
            raise Unauthenticated(f"token rejected: {exc}") from exc
        key = await self._key_for(kid)
        try:
            claims = jwt.decode(
                token, key, algorithms=list(self.algorithms),
                audience=self.audience, issuer=self.issuer,
                options={"require": ["sub", "exp", "iss", "aud"]},
            )
        except jwt.PyJWTError as exc:
            raise Unauthenticated(f"token rejected: {exc}") from exc
        return Actor(subject_id=str(claims["sub"]), roles=_roles_from(claims, self.roles_claim))


def _roles_from(claims: dict[str, Any], roles_claim: str) -> frozenset[str]:
    """Roles as a list or as the space-delimited string OAuth scopes use."""
    roles = claims.get(roles_claim) or []
    if isinstance(roles, str):
        roles = [role for role in roles.split() if role]
    return frozenset(str(role) for role in roles)
