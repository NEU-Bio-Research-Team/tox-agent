"""Capability token issuance and verification (plan section 8.5)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from toxagent.config import SecuritySettings
from toxagent.domain.errors import Unauthenticated
from toxagent.domain.ids import new_id
from toxagent.tools.capability import ALGORITHM, AUDIENCE, ISSUER, CapabilityTokenService

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def a_service(db, **overrides) -> CapabilityTokenService:
    overrides.setdefault("capability_ttl_s", 900)
    return CapabilityTokenService(
        SecuritySettings(capability_secret="test-secret-at-least-32-bytes-long", **overrides), db
    )


async def test_a_token_carries_exactly_the_allowlist_of_its_profile(db):
    service = a_service(db)
    session_id, run_id = new_id("ses"), new_id("run")
    token = await service.issue(
        session_id=session_id, run_id=run_id, profile="report_qa", owner_id="user-1",
    )
    claims = await service.verify(token)
    assert claims.session_id == session_id
    assert claims.run_id == run_id
    assert claims.allowed_tools == {"get_analysis_slice", "get_attribution", "submit_grounded_answer"}
    assert claims.allows("get_attribution")
    assert not claims.allows("search_toxicology_evidence")


async def test_an_unknown_profile_cannot_be_issued(db):
    service = a_service(db)
    with pytest.raises(ValueError, match="unknown capability profile"):
        await service.issue(
            session_id=new_id("ses"), run_id=new_id("run"), profile="root", owner_id="user-1",
        )


async def test_a_revoked_token_is_refused_even_though_the_signature_is_valid(db):
    service = a_service(db)
    token = await service.issue(
        session_id=new_id("ses"), run_id=new_id("run"), profile="analysis", owner_id="user-1",
    )
    claims = await service.verify(token)
    await service.revoke(claims.jti)
    with pytest.raises(Unauthenticated, match="not active"):
        await service.verify(token)


async def test_an_expired_token_is_refused(db):
    service = a_service(db, capability_ttl_s=0, capability_grace_s=0)
    token = await service.issue(
        session_id=new_id("ses"), run_id=new_id("run"), profile="analysis", owner_id="user-1",
        deadline_at=NOW,
    )
    with pytest.raises(Unauthenticated, match="rejected"):
        await service.verify(token)


async def test_a_token_never_outlives_its_run_deadline_by_more_than_the_grace(db):
    service = a_service(db, capability_ttl_s=3600, capability_grace_s=30)
    deadline = datetime.now(timezone.utc) + timedelta(seconds=10)
    token = await service.issue(
        session_id=new_id("ses"), run_id=new_id("run"), profile="analysis", owner_id="user-1",
        deadline_at=deadline,
    )
    claims = await service.verify(token)
    assert claims.expires_at <= deadline + timedelta(seconds=31)


async def test_a_token_this_server_never_issued_is_refused_even_if_correctly_signed(db):
    service = a_service(db)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    forged = jwt.encode(
        {
            "jti": new_id("cap"), "iss": ISSUER, "aud": AUDIENCE, "sub": "forged",
            "own": "mallory", "roles": [], "sid": new_id("ses"), "rid": new_id("run"),
            "prof": "analysis", "tools": ["create_analysis_snapshot"],
            "iat": int(datetime.now(timezone.utc).timestamp()), "exp": int(future.timestamp()),
        },
        "test-secret-at-least-32-bytes-long", algorithm=ALGORITHM,
    )
    with pytest.raises(Unauthenticated, match="not active"):
        await service.verify(forged)


async def test_a_token_signed_with_the_wrong_secret_is_refused(db):
    service = a_service(db)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    token = jwt.encode(
        {
            "jti": new_id("cap"), "iss": ISSUER, "aud": AUDIENCE, "sub": "x", "own": "user-1",
            "roles": [], "sid": new_id("ses"), "rid": new_id("run"), "prof": "analysis",
            "tools": ["create_analysis_snapshot"],
            "iat": int(datetime.now(timezone.utc).timestamp()), "exp": int(future.timestamp()),
        },
        "wrong-secret-also-at-least-32-bytes", algorithm=ALGORITHM,
    )
    with pytest.raises(Unauthenticated, match="rejected"):
        await service.verify(token)


async def test_require_tool_denies_what_the_claims_do_not_allow(db):
    from toxagent.domain.errors import Forbidden

    service = a_service(db)
    token = await service.issue(
        session_id=new_id("ses"), run_id=new_id("run"), profile="analysis", owner_id="user-1",
    )
    claims = await service.verify(token)
    with pytest.raises(Forbidden):
        CapabilityTokenService.require_tool(claims, "search_toxicology_evidence")
