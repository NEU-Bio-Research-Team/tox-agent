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


async def test_a_token_preserves_run_intent_as_presentation_context(db):
    service = a_service(db)
    token = await service.issue(
        session_id=new_id("ses"), run_id=new_id("run"), profile="report_qa",
        owner_id="user-1", intent="attribution",
    )
    assert (await service.verify(token)).intent == "attribution"


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


# --------------------------------------------------- Wave 3 abuse-case matrix


async def test_a_token_for_another_audience_is_refused_even_when_issued_here(db):
    """MCP authorization: a resource server validates the audience. A token
    this service issued, re-signed for a different audience with the same
    secret (a confused deputy), must not be accepted."""
    service = a_service(db)
    token = await service.issue(
        session_id=new_id("ses"), run_id=new_id("run"), profile="analysis", owner_id="user-1",
    )
    claims = jwt.decode(token, options={"verify_signature": False})
    retargeted = jwt.encode(
        {**claims, "aud": "some-other-resource"}, "test-secret-at-least-32-bytes-long",
        algorithm=ALGORITHM,
    )
    with pytest.raises(Unauthenticated, match="rejected"):
        await service.verify(retargeted)


async def test_a_token_whose_tool_list_was_widened_is_refused(db):
    """The allowlist in a token is signed; editing it without the secret
    breaks the signature, and a token re-signed with a guessed secret is not
    one the store issued."""
    service = a_service(db)
    token = await service.issue(
        session_id=new_id("ses"), run_id=new_id("run"), profile="analysis", owner_id="user-1",
    )
    header, payload, signature = token.split(".")
    import base64
    import json

    decoded = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    decoded["tools"] = decoded["tools"] + ["search_toxicology_evidence"]
    widened_payload = base64.urlsafe_b64encode(json.dumps(decoded).encode()).decode().rstrip("=")
    with pytest.raises(Unauthenticated, match="rejected"):
        await service.verify(f"{header}.{widened_payload}.{signature}")


async def test_a_run_scoped_token_is_dead_once_revoked_at_run_end(db):
    """Cross-run reuse: the gateway revokes a run's token when the run ends;
    presenting it for any later run is refused by the store, not by expiry."""
    service = a_service(db, capability_ttl_s=3600)
    token = await service.issue(
        session_id=new_id("ses"), run_id=new_id("run"), profile="decision_support",
        owner_id="user-1",
    )
    await service.revoke((await service.verify(token)).jti)
    with pytest.raises(Unauthenticated, match="not active"):
        await service.verify(token)
