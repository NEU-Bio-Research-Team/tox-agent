"""Short-lived run capability tokens (plan section 8.5).

A runtime is not a principal. It holds a signed token that says: this exact
session, this exact run, this exact tool allowlist, until this exact time. The
tool plane resolves the owner from the token and ignores any session the model
puts in its arguments, so a model that has read a session id out of a document
still cannot reach that session.

Tokens are recorded by ``jti`` so a run's tool access can be revoked
immediately rather than at expiry, and so the audit can say which token
authorised which call.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from ..config import SecuritySettings
from ..domain.errors import Forbidden, Unauthenticated
from ..domain.ids import CAPABILITY, new_id
from .registry import PROFILES

ALGORITHM = "HS256"
AUDIENCE = "toxagent-tools"
ISSUER = "toxagent-control"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CapabilityClaims:
    jti: str
    subject_id: str
    roles: frozenset[str]
    session_id: str
    run_id: str
    profile: str
    allowed_tools: frozenset[str]
    expires_at: datetime
    runtime_binding_id: str | None = None
    language: str = "en"

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


class CapabilityTokenService:
    def __init__(self, settings: SecuritySettings, database) -> None:
        if not settings.capability_secret:
            raise ValueError("a capability signing secret is required to expose the tool plane")
        self._secret = settings.capability_secret
        self._ttl = timedelta(seconds=settings.capability_ttl_s)
        self._grace = timedelta(seconds=settings.capability_grace_s)
        self._db = database

    async def issue(
        self,
        *,
        session_id: str,
        run_id: str,
        profile: str,
        owner_id: str,
        roles: frozenset[str] = frozenset(),
        runtime_binding_id: str | None = None,
        deadline_at: datetime | None = None,
        language: str = "en",
    ) -> str:
        allowed = PROFILES.get(profile)
        if allowed is None:
            raise ValueError(f"unknown capability profile {profile!r}")

        # Never outlive the run: a token valid after its run has reported an
        # outcome is a token that can write into a finished audit trail.
        expires_at = min(
            _now() + self._ttl, (deadline_at or _now() + self._ttl) + self._grace
        )
        jti = new_id(CAPABILITY)
        payload: dict[str, Any] = {
            "jti": jti,
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": runtime_binding_id or f"local:{run_id}",
            "own": owner_id,
            "roles": sorted(roles),
            "sid": session_id,
            "rid": run_id,
            "prof": profile,
            "tools": sorted(allowed),
            "lang": language,
            "iat": int(_now().timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        async with self._db.unit_of_work() as uow:
            await uow.capability_tokens.issue(
                jti=jti, session_id=session_id, run_id=run_id,
                runtime_binding_id=runtime_binding_id, allowed_tools=sorted(allowed),
                issued_at=_now(), expires_at=expires_at,
            )
            await uow.commit()
        return jwt.encode(payload, self._secret, algorithm=ALGORITHM)

    async def verify(self, token: str) -> CapabilityClaims:
        try:
            payload = jwt.decode(
                token, self._secret, algorithms=[ALGORITHM], audience=AUDIENCE, issuer=ISSUER,
                options={"require": ["exp", "jti", "sid", "rid", "prof"]},
            )
        except jwt.PyJWTError as exc:
            raise Unauthenticated(f"capability token rejected: {exc}") from exc

        async with self._db.unit_of_work() as uow:
            if not await uow.capability_tokens.is_valid(payload["jti"], now=_now()):
                # Revoked, expired in the store, or never issued by this control
                # plane. A validly signed token this server did not issue is
                # still refused.
                raise Unauthenticated("capability token is not active")

        return CapabilityClaims(
            jti=payload["jti"],
            subject_id=payload.get("own", ""),
            roles=frozenset(payload.get("roles", ())),
            session_id=payload["sid"],
            run_id=payload["rid"],
            profile=payload["prof"],
            allowed_tools=frozenset(payload.get("tools", ())),
            expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            runtime_binding_id=(
                payload["sub"] if not str(payload["sub"]).startswith("local:") else None
            ),
            language=payload.get("lang", "en"),
        )

    async def revoke(self, jti: str) -> None:
        async with self._db.unit_of_work() as uow:
            await uow.capability_tokens.revoke(jti, now=_now())
            await uow.commit()

    @staticmethod
    def require_tool(claims: CapabilityClaims, tool_name: str) -> None:
        if not claims.allows(tool_name):
            raise Forbidden(f"{tool_name} is not in this run's capability")
