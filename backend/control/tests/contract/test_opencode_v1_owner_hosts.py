"""The V1 adapter on private per-connection hosts (I12/K05 closed for OpenCode).

A run pinned to a user AI connection must happen entirely on that
connection's host: session, MCP install, prompt, events, close. Nothing about
it may reach the shared host, whose account belongs to the deployment. The
prompt names OpenCode's provider id (a custom endpoint is not called
``openai_compatible`` inside OpenCode), while the product binding keeps the
product's own provider id.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from toxagent.connections import providers
from toxagent.domain.errors import RuntimeUnavailable
from toxagent.harness.adapters.opencode_v1 import MCP_NAME, OpenCodeV1Provider
from toxagent.harness.owner_hosts import OpenCodeHostPool
from toxagent.harness.provider import RuntimeSessionSpec, RuntimeTurn
from toxagent.platform.config import RuntimeSettings

pytestmark = pytest.mark.anyio


class FakeProcess:
    pid = None

    def __init__(self):
        self.returncode = None

    async def wait(self):
        return 0

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


def _spec(**overrides) -> RuntimeSessionSpec:
    values = dict(
        session_id="ses_" + "1" * 32, run_id="run_" + "2" * 32,
        provider_id="openai_compatible", model_id="my-model", profile="decision_support",
        system_prompt="sys", system_prompt_hash="a" * 64, tool_schema=(),
        tool_schema_hash="b" * 64, mcp_url="http://control.private/internal/mcp",
        max_steps=8, deadline_at=datetime.now(timezone.utc) + timedelta(minutes=2),
        connection_id="con_" + "c" * 32, provider_credential="sk-user-key",
        provider_base_url="https://llm.example/v1", auth_mode="api_key",
        owner_id="owner-a", provider_credential_ref="secret_" + "1" * 32,
    )
    values.update(overrides)
    return RuntimeSessionSpec(**values)


@pytest.fixture
def world(tmp_path: Path):
    shared_requests: list[httpx.Request] = []
    host_requests: list[httpx.Request] = []
    spawned: list[dict] = []

    def shared(request: httpx.Request) -> httpx.Response:
        shared_requests.append(request)
        return httpx.Response(500)

    def host(request: httpx.Request) -> httpx.Response:
        host_requests.append(request)
        path = request.url.path
        if path == "/agent":
            return httpx.Response(200, json=[{"name": "toxagent"}, {"name": "toxagent-report"}])
        if path == "/session" and request.method == "POST":
            return httpx.Response(200, json={"id": "private-session"})
        if path.startswith("/auth/") and request.method == "PUT":
            return httpx.Response(200, json=True)
        if path == "/mcp":
            return httpx.Response(200, json={})
        if path in {f"/mcp/{MCP_NAME}/connect", f"/mcp/{MCP_NAME}/disconnect"}:
            return httpx.Response(200, json=True)
        if path == "/session/private-session/prompt_async":
            return httpx.Response(204)
        if path == "/session/private-session" and request.method == "DELETE":
            return httpx.Response(200, json=True)
        if path == "/global/event":
            event = {"payload": {"type": "session.idle", "properties": {"sessionID": "private-session"}}}
            return httpx.Response(200, content=b"data: " + json.dumps(event).encode() + b"\n\n")
        raise AssertionError(f"unexpected {request.method} {path}")

    async def spawn(argv, env, cwd, log_path):
        process = FakeProcess()
        spawned.append({"env": env, "process": process})
        return process

    profile = tmp_path / "toxagent.json"
    profile.write_text("{}")
    pool = OpenCodeHostPool(
        binary="/opt/opencode", profile_path=profile, root=tmp_path / "hosts",
        spawner=spawn, startup_timeout_s=2, port_picker=lambda: 41234,
        client_factory=lambda base, pw: httpx.AsyncClient(
            base_url=base, auth=("opencode", pw), transport=httpx.MockTransport(host)
        ),
    )
    settings = RuntimeSettings(
        kind="opencode", opencode_base_url="http://shared.test",
        opencode_directory="/srv/shared", provider_id="openai", model_id="deployment-model",
    )
    client = httpx.AsyncClient(base_url="http://shared.test", transport=httpx.MockTransport(shared))
    provider = OpenCodeV1Provider(settings, client=client, hosts=pool)
    return provider, pool, shared_requests, host_requests, spawned


async def test_a_profile_run_happens_entirely_on_its_own_host(world):
    provider, pool, shared_requests, host_requests, spawned = world
    session = await provider.create_session(_spec())
    assert session.provider_id == "openai_compatible"  # the product's name, for the binding
    await provider.send(session, RuntimeTurn(
        turn_id="run_" + "2" * 32, user_message="question",
        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        capability_token="cap-secret",
    ))
    events = [event async for event in provider.events(session, None)]
    assert events[-1].type.value == "runtime.turn.idle"
    outcome = await provider.close(session)
    assert outcome.closed is True

    assert shared_requests == []
    paths = [r.url.path for r in host_requests]
    assert paths[0] == "/agent"
    assert "/session/private-session/prompt_async" in paths
    prompt = json.loads(next(r for r in host_requests if r.url.path.endswith("prompt_async")).content)
    # OpenCode's own name for the custom endpoint, not the product's.
    assert prompt["model"] == {"providerID": providers.CUSTOM_RUNTIME_PROVIDER, "modelID": "my-model"}
    assert "sk-user-key" not in json.dumps(prompt)
    # The key reached the host's environment, nowhere in its config.
    env = spawned[0]["env"]
    assert env["TOXAGENT_PROVIDER_KEY"] == "sk-user-key"
    assert "sk-user-key" not in env["OPENCODE_CONFIG_CONTENT"]
    # The run's workspace sits inside the host and is removed at close.
    directory = Path(dict(host_requests[1].url.params)["directory"])
    assert directory.parent.name == "workspace" and not directory.exists()
    assert pool.size == 1 and next(iter(pool._hosts.values())).leases == 0


async def test_a_chatgpt_profile_prompts_openai_on_its_host(world):
    provider, pool, shared_requests, host_requests, _ = world
    oauth = {"type": "oauth", "refresh": "r", "access": "a", "expires": 1}
    session = await provider.create_session(_spec(
        provider_id="openai", model_id="gpt-5.5", auth_mode="chatgpt_subscription",
        provider_base_url=None, provider_credential=json.dumps(oauth),
    ))
    await provider.send(session, RuntimeTurn(
        turn_id="t", user_message="q",
        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=1), capability_token="c",
    ))
    put = next(r for r in host_requests if r.method == "PUT")
    assert put.url.path == "/auth/openai" and json.loads(put.content) == oauth
    prompt = json.loads(next(r for r in host_requests if r.url.path.endswith("prompt_async")).content)
    assert prompt["model"] == {"providerID": "openai", "modelID": "gpt-5.5"}
    assert shared_requests == []
    await provider.close(session)


async def test_without_private_hosts_a_credential_is_still_refused():
    provider = OpenCodeV1Provider(RuntimeSettings(kind="opencode", opencode_directory="/srv/x"))
    with pytest.raises(RuntimeUnavailable, match="will not fall back"):
        await provider.create_session(_spec())
    with pytest.raises(RuntimeUnavailable):
        await provider.create_session(_spec(provider_credential=None, auth_mode="chatgpt_subscription"))
    await provider.aclose()


async def test_closing_the_adapter_stops_the_private_hosts(world):
    provider, pool, *_ , spawned = world
    session = await provider.create_session(_spec())
    await provider.close(session)
    await provider.aclose()
    assert pool.size == 0 and spawned[0]["process"].returncode is not None


async def test_a_session_never_takes_or_removes_another_sessions_workspace(world):
    provider, pool, *_ = world
    first = await provider.create_session(_spec(profile="decision_support"))
    host = next(iter(pool._hosts.values()))
    run_dir = host.workspace / ("run_" + "2" * 32)
    taken = host.workspace / ("run_" + "2" * 32 + "-decision-support")
    taken.mkdir()
    (taken / "keep").write_text("another session's file")
    third = await provider.create_session(_spec(profile="decision_support"))
    assert (taken / "keep").exists() and run_dir.exists()
    third_dir = Path(provider._directories[third.runtime_session_id])
    assert third_dir not in {run_dir, taken} and third_dir.parent == host.workspace
    await provider.close(first)
