"""Pinned OpenCode V1 wire-contract tests.

These use an HTTP transport double rather than a model provider.  They prove
the exact management/SSE calls and, importantly, that the run capability stays
out of the model prompt while being sent only as a remote-MCP secret header.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from toxagent.config import RuntimeSettings
from toxagent.harness.adapters.opencode_v1 import (
    MCP_NAME,
    MCP_TOOL_PREFIXES,
    OpenCodeV1Provider,
)
from toxagent.harness.provider import RuntimeEventType, RuntimeSessionSpec, RuntimeTurn

pytestmark = pytest.mark.anyio


def _settings() -> RuntimeSettings:
    return RuntimeSettings(
        kind="opencode",
        opencode_base_url="http://opencode.test",
        opencode_directory="/srv/toxagent/runtime/run-1",
        provider_id="provider-a",
        model_id="model-a",
    )


def _spec() -> RuntimeSessionSpec:
    now = datetime.now(timezone.utc)
    return RuntimeSessionSpec(
        session_id="ses_" + "1" * 32,
        run_id="run_" + "2" * 32,
        provider_id="provider-a",
        model_id="model-a",
        profile="report_qa",
        system_prompt="system prompt with no secret",
        system_prompt_hash="a" * 64,
        tool_schema=(),
        tool_schema_hash="b" * 64,
        mcp_url="http://control.private/internal/mcp",
        max_steps=4,
        deadline_at=now + timedelta(minutes=2),
    )


async def test_v1_dispatch_configures_mcp_after_binding_and_keeps_token_out_of_prompt():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/agent":
            return httpx.Response(200, json=[{"name": "toxagent"}])
        if request.url.path == "/session" and request.method == "POST":
            return httpx.Response(200, json={"id": "opencode-session-1"})
        if request.url.path == "/mcp" and request.method == "POST":
            return httpx.Response(200, json={MCP_NAME: {"status": "connected"}})
        if request.url.path == f"/mcp/{MCP_NAME}/connect":
            return httpx.Response(200, json=True)
        if request.url.path == "/session/opencode-session-1/prompt_async":
            return httpx.Response(204)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = httpx.AsyncClient(base_url="http://opencode.test", transport=httpx.MockTransport(handler))
    provider = OpenCodeV1Provider(_settings(), client=client)
    assert (await provider.health()).healthy
    session = await provider.create_session(_spec())
    await provider.send(
        session,
        RuntimeTurn(
            turn_id="run_" + "2" * 32,
            user_message="What is the hERG result?",
            deadline_at=datetime.now(timezone.utc) + timedelta(minutes=2),
            capability_token="run-capability-secret",
        ),
    )

    assert [request.url.path for request in requests] == [
        "/agent",
        "/session",
        "/mcp",
        f"/mcp/{MCP_NAME}/connect",
        "/session/opencode-session-1/prompt_async",
    ]
    mcp_body = json.loads(requests[2].content)
    assert mcp_body["name"] == MCP_NAME
    assert mcp_body["config"]["url"] == "http://control.private/internal/mcp"
    assert mcp_body["config"]["headers"] == {"Authorization": "Bearer run-capability-secret"}
    # The remote-MCP timeout OpenCode enforces per request must clear the
    # longest tool's hard timeout (get_attribution, 180 s) or a slow-but-valid
    # call is aborted with -32001 (progress log §4.3).
    assert mcp_body["config"]["timeout"] >= 180_000

    prompt_body = json.loads(requests[-1].content)
    assert prompt_body["agent"] == "toxagent"
    assert prompt_body["system"] == "system prompt with no secret"
    assert prompt_body["parts"] == [{"type": "text", "text": "What is the hERG result?"}]
    assert "run-capability-secret" not in json.dumps(prompt_body)
    expected_directory = "/srv/toxagent/runtime/run-1/run_" + "2" * 32
    assert dict(requests[-1].url.params) == {"directory": expected_directory}
    await client.aclose()


async def test_v1_events_normalize_only_the_bound_session_and_preserve_usage():
    session_id = "opencode-session-1"
    foreign = "opencode-session-foreign"
    events = [
        {
            "directory": "/srv/toxagent/runtime/run-1",
            "payload": {
                "type": "message.part.updated",
                "properties": {
                    "part": {"type": "text", "sessionID": foreign, "text": "foreign"},
                    "delta": "foreign",
                },
            },
        },
        {
            "directory": "/srv/toxagent/runtime/run-1",
            "payload": {
                "type": "session.status",
                "properties": {"sessionID": session_id, "status": {"type": "busy"}},
            },
        },
        {
            "directory": "/srv/toxagent/runtime/run-1",
            "payload": {
                "type": "message.part.updated",
                "properties": {
                    "part": {
                        "type": "tool",
                        "sessionID": session_id,
                        "callID": "tool-call-1",
                        "tool": "mcp_toxagent_get_analysis_slice",
                        "state": {"status": "completed"},
                    }
                },
            },
        },
        {
            "directory": "/srv/toxagent/runtime/run-1",
            "payload": {
                "type": "message.part.updated",
                "properties": {
                    "part": {"type": "text", "sessionID": session_id, "text": "0.731"},
                    "delta": "0.731",
                },
            },
        },
        {
            "directory": "/srv/toxagent/runtime/run-1",
            "payload": {
                "type": "message.part.updated",
                "properties": {
                    "part": {
                        "type": "step-finish",
                        "sessionID": session_id,
                        "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0}},
                    }
                },
            },
        },
        {
            "directory": "/srv/toxagent/runtime/run-1",
            "payload": {"type": "session.idle", "properties": {"sessionID": session_id}},
        },
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/session/status" or request.url.path == "/global/event"
        if request.url.path == "/session/status":
            return httpx.Response(200, json={session_id: {"type": "busy"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = httpx.AsyncClient(base_url="http://opencode.test", transport=httpx.MockTransport(handler))
    provider = OpenCodeV1Provider(_settings(), client=client)
    provider._directories[session_id] = "/srv/toxagent/runtime/run-1"  # V1 session was created earlier.
    received = [
        event
        async for event in provider.events(
            type("Session", (), {"runtime_session_id": session_id})(),  # protocol-level shape only
            after=None,
        )
    ]
    assert [event.type for event in received] == [
        RuntimeEventType.TURN_STARTED,
        RuntimeEventType.TOOL_COMPLETED,
        RuntimeEventType.MESSAGE_DELTA,
        RuntimeEventType.USAGE_REPORTED,
        RuntimeEventType.TURN_IDLE,
    ]
    assert received[1].payload["tool_name"] == "get_analysis_slice"
    assert received[2].payload["text"] == "0.731"
    assert received[3].payload["tokens"]["output"] == 5
    await client.aclose()


async def test_v1_events_do_not_treat_queued_turns_stale_idle_status_as_terminal():
    """``prompt_async`` can return before V1 flips a queued turn to busy.

    The adapter must subscribe to the event feed instead of accepting that
    transient idle status as a completed run.  The user text replay is also
    intentionally suppressed until the bound session has entered ``busy``.
    """
    session_id = "opencode-session-1"
    events = [
        {
            "payload": {
                "type": "message.part.updated",
                "properties": {
                    "part": {"type": "text", "sessionID": session_id, "text": "user prompt"},
                    "delta": "user prompt",
                },
            },
        },
        {
            "payload": {
                "type": "session.status",
                "properties": {"sessionID": session_id, "status": {"type": "busy"}},
            },
        },
        {"payload": {"type": "session.idle", "properties": {"sessionID": session_id}}},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/global/event"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = httpx.AsyncClient(base_url="http://opencode.test", transport=httpx.MockTransport(handler))
    provider = OpenCodeV1Provider(_settings(), client=client)
    provider._directories[session_id] = "/srv/toxagent/runtime/run-1"
    received = [
        event
        async for event in provider.events(
            type("Session", (), {"runtime_session_id": session_id})(), after=None
        )
    ]

    assert [event.type for event in received] == [
        RuntimeEventType.TURN_STARTED,
        RuntimeEventType.TURN_IDLE,
    ]
    await client.aclose()


@pytest.mark.parametrize("prefix", MCP_TOOL_PREFIXES)
async def test_v1_tool_events_normalize_either_mcp_tool_naming(prefix: str):
    """OpenCode has used both ``mcp_toxagent_<tool>`` and ``toxagent_<tool>``
    for a remote MCP tool (progress log §3.2). Either must normalize to the
    bare tool name."""
    session_id = "opencode-session-1"
    events = [
        {
            "payload": {
                "type": "message.part.updated",
                "properties": {
                    "part": {
                        "type": "tool",
                        "sessionID": session_id,
                        "callID": "tool-call-1",
                        "tool": f"{prefix}get_analysis_slice",
                        "state": {"status": "completed"},
                    }
                },
            },
        },
        {"payload": {"type": "session.idle", "properties": {"sessionID": session_id}}},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session/status":
            return httpx.Response(200, json={session_id: {"type": "busy"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = httpx.AsyncClient(base_url="http://opencode.test", transport=httpx.MockTransport(handler))
    provider = OpenCodeV1Provider(_settings(), client=client)
    provider._directories[session_id] = "/srv/toxagent/runtime/run-1"
    received = [
        event
        async for event in provider.events(
            type("Session", (), {"runtime_session_id": session_id})(), after=None
        )
    ]
    tool_events = [e for e in received if e.type is RuntimeEventType.TOOL_COMPLETED]
    assert [e.payload["tool_name"] for e in tool_events] == ["get_analysis_slice"]
    await client.aclose()


async def test_v1_cancel_and_close_report_only_real_transport_outcomes():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/abort"):
            return httpx.Response(200, json=True)
        if request.url.path.endswith("/disconnect"):
            return httpx.Response(200, json=True)
        if request.method == "DELETE":
            return httpx.Response(200, json=True)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = httpx.AsyncClient(base_url="http://opencode.test", transport=httpx.MockTransport(handler))
    provider = OpenCodeV1Provider(_settings(), client=client)
    provider._directories["opencode-session-1"] = "/srv/toxagent/runtime/run-1"
    session = type(
        "Session", (), {"runtime_session_id": "opencode-session-1", "provider_id": "p", "model_id": "m"}
    )()
    receipt = type("Receipt", (), {"turn_id": "run_" + "2" * 32, "accepted": True})()
    cancelled = await provider.cancel(session, receipt)
    closed = await provider.close(session)
    assert cancelled.runtime_cancel_supported and cancelled.action == "runtime_turn_aborted"
    assert closed.closed
    assert [request.method for request in requests] == ["POST", "POST", "DELETE"]
    await client.aclose()


async def test_v1_local_mode_creates_and_reaps_only_its_run_workspace(tmp_path: Path):
    settings = RuntimeSettings(
        kind="opencode",
        opencode_base_url="http://opencode.test",
        opencode_directory=str(tmp_path / "opencode-runs"),
        opencode_create_run_directories=True,
        provider_id="provider-a",
        model_id="model-a",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session" and request.method == "POST":
            return httpx.Response(200, json={"id": "opencode-session-1"})
        if request.url.path.endswith("/disconnect"):
            return httpx.Response(200, json=True)
        if request.method == "DELETE":
            return httpx.Response(200, json=True)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = httpx.AsyncClient(base_url="http://opencode.test", transport=httpx.MockTransport(handler))
    provider = OpenCodeV1Provider(settings, client=client)
    session = await provider.create_session(_spec())
    workspace = tmp_path / "opencode-runs" / _spec().run_id
    assert workspace.is_dir()

    closed = await provider.close(session)

    assert closed.closed
    assert not workspace.exists()
    await client.aclose()


def test_v1_refuses_a_floating_or_wrong_version_pin():
    with pytest.raises(ValueError, match="pinned V1 contract"):
        OpenCodeV1Provider(RuntimeSettings(kind="opencode", opencode_version="latest"))
