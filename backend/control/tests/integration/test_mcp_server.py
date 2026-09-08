"""The MCP transport (plan sections 8, 10.1, 11.3; Phase 2 exit gate).

Two things are asserted that the plan makes non-negotiable: a standard MCP
client can call exactly the allowlist a token carries, and a denied tool is
refused identically whether a model tries to enumerate it or to call it
directly (PROD-06) — there is no probing signal that distinguishes "hidden"
from "exists but refused".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.support.mcp import connected_session, http_session
from tests.support.predictor import ASPIRIN, StubPredictor
from toxagent.application.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.config import PolicySettings, SecuritySettings
from toxagent.domain.events import EventType
from toxagent.domain.ids import new_id
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.tools.bootstrap import build_registry
from toxagent.tools.capability import CapabilityTokenService
from toxagent.tools.mcp_server import build_server, mcp_asgi_app
from toxagent.tools.runner import ToolRunner

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)
SECRET = "mcp-test-secret-at-least-32-bytes-long"


async def rig(db, stub: StubPredictor, *, profile: str = "analysis"):
    session = Session.create("user-1", now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.REPORT_QA, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        uow.emit(
            session_id=session.id, type=EventType.SESSION_CREATED,
            entity_type="session", entity_id=session.id,
        )
        await uow.commit()

    client = stub.client()
    analysis = CreateAnalysis(db, client, PolicySettings())
    registry = build_registry(db, client, analysis)
    runner = ToolRunner(registry, db)
    tokens = CapabilityTokenService(SecuritySettings(capability_secret=SECRET), db)
    token = await tokens.issue(
        session_id=session.id, run_id=run.id, profile=profile, owner_id="user-1",
        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    claims = await tokens.verify(token)
    return registry, runner, tokens, token, claims, session, run


# --- in-memory protocol conformance -----------------------------------------

async def test_a_standard_client_sees_exactly_the_profiles_allowlist(db):
    registry, runner, _, _, claims, _, _ = await rig(db, StubPredictor(), profile="report_qa")
    server = build_server(registry, runner, claims)
    async with connected_session(server) as client:
        tools = await client.list_tools()
        visible = {t.name for t in tools.tools}
        assert visible == {"get_analysis_slice", "get_attribution", "submit_grounded_answer"}


async def test_calling_a_tool_outside_the_profile_is_refused_not_silently_ignored(db):
    registry, runner, _, _, claims, _, _ = await rig(db, StubPredictor(), profile="analysis")
    server = build_server(registry, runner, claims)
    async with connected_session(server) as client:
        result = await client.call_tool("search_toxicology_evidence", {"analysis_id": "x", "query": "q"})
        assert result.isError
        assert "not available" in result.content[0].text


async def test_a_visible_tool_call_round_trips_through_the_real_protocol(db):
    registry, runner, _, _, claims, session, _ = await rig(db, StubPredictor(), profile="analysis")
    server = build_server(registry, runner, claims)
    async with connected_session(server) as client:
        result = await client.call_tool(
            "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN}
        )
        assert not result.isError
        assert result.structuredContent["status"] == "completed"
        assert result.structuredContent["observation_ids"]
        # The model never receives the raw predictor payload through this path.
        assert "0.73064" not in result.content[0].text


async def test_a_malformed_argument_set_is_refused_before_the_handler_runs(db):
    registry, runner, _, _, claims, session, _ = await rig(db, StubPredictor(), profile="analysis")
    server = build_server(registry, runner, claims)
    async with connected_session(server) as client:
        result = await client.call_tool(
            "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN, "bogus": 1}
        )
        assert result.isError


# --- HTTP auth boundary ------------------------------------------------------

async def test_the_endpoint_requires_a_bearer_token(db):
    registry, runner, tokens, _, _, _, _ = await rig(db, StubPredictor())
    app = mcp_asgi_app(tokens, registry, runner)
    with pytest.raises(Exception):
        async with http_session(app, token=None):
            pass


async def test_an_expired_or_forged_token_is_refused_at_the_transport(db):
    registry, runner, tokens, _, _, _, _ = await rig(db, StubPredictor())
    app = mcp_asgi_app(tokens, registry, runner)
    with pytest.raises(Exception):
        async with http_session(app, token="not-a-real-token"):
            pass


async def test_a_real_run_capability_token_authenticates_over_http(db):
    registry, runner, tokens, token, claims, session, _ = await rig(
        db, StubPredictor(), profile="analysis"
    )
    app = mcp_asgi_app(tokens, registry, runner)
    async with http_session(app, token=token) as client:
        listed = await client.list_tools()
        assert {t.name for t in listed.tools} == {
            "create_analysis_snapshot", "get_analysis_slice", "submit_grounded_answer",
        }
        result = await client.call_tool(
            "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN}
        )
        assert not result.isError


async def test_a_revoked_token_stops_working_over_http_immediately(db):
    registry, runner, tokens, token, claims, _, _ = await rig(db, StubPredictor())
    await tokens.revoke(claims.jti)
    app = mcp_asgi_app(tokens, registry, runner)
    with pytest.raises(Exception):
        async with http_session(app, token=token):
            pass


async def test_a_token_scoped_to_one_run_cannot_be_reused_for_another_sessions_data(db):
    """Plan section 8.5: a model reading a session id from a document still
    cannot reach it — the token, not the argument, decides scope."""
    registry, runner, tokens, token, claims, own_session, _ = await rig(
        db, StubPredictor(), profile="analysis"
    )
    _, _, _, _, _, foreign_session, _ = await rig(db, StubPredictor(), profile="analysis")
    app = mcp_asgi_app(tokens, registry, runner)
    async with http_session(app, token=token) as client:
        result = await client.call_tool(
            "create_analysis_snapshot",
            {"session_id": foreign_session.id, "smiles": ASPIRIN},
        )
        assert result.isError
        assert "not the session this run belongs to" in result.content[0].text
