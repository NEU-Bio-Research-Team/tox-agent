"""Who is asking.

The control plane authenticates product users. A runtime never authenticates on
a user's behalf: it holds a capability token scoped to one run, which is a
different mechanism handled in ``tools/capability.py`` and deliberately not
interchangeable with this one.

Two providers ship. ``StaticTokenAuth`` is a development shim, refused in
production by ``SecuritySettings``. ``JwtAuth`` verifies a signed token from an
identity provider and reads roles from a claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
        roles = claims.get(self.roles_claim) or []
        if isinstance(roles, str):
            roles = [r for r in roles.split() if r]
        return Actor(subject_id=str(claims["sub"]), roles=frozenset(roles))


def build_auth(settings: SecuritySettings) -> AuthProvider:
    if settings.static_tokens:
        return StaticTokenAuth.from_settings(settings)
    if not settings.capability_secret:
        raise ValueError(
            "no authentication configured: set TOXAGENT_STATIC_TOKENS for development "
            "or TOXAGENT_CAPABILITY_SECRET for signed tokens"
        )
    return JwtAuth(secret=settings.capability_secret)
