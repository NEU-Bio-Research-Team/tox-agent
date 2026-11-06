"""Private OpenCode hosts, one per user AI connection (harness/owner_hosts.py).

What must hold, whatever the binary does:

- the credential reaches only that connection's host, installed the way
  OpenCode reads it, and the host is isolated (own HOME/XDG, the checked-in
  agent profile, loopback, a password on its management API);
- a host is reused while the connection is unchanged and replaced when it
  changes, never while a run is still on it;
- the pool is bounded and refuses when full — it never falls back to a shared
  host's account;
- a ChatGPT token pair the runtime refreshed is written back under the same
  reference, and the credential file leaves disk with the host.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from toxagent.connections.runtime_install import HostCredential
from toxagent.domain.errors import RuntimeUnavailable
from toxagent.harness.owner_hosts import OpenCodeHostPool

pytestmark = pytest.mark.anyio

OAUTH = {"type": "oauth", "refresh": "r-1", "access": "a-1", "expires": 1_900_000_000_000}


class FakeProcess:
    pid = None  # never a real process group

    def __init__(self) -> None:
        self.returncode: int | None = None

    async def wait(self) -> int:
        return self.returncode or 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


class Harness:
    def __init__(self, tmp_path: Path, *, agents=("toxagent", "toxagent-report"), auth_ok=True):
        self.spawned: list[dict] = []
        self.requests: list[tuple[str, str, str, object]] = []
        self.agents = agents
        self.auth_ok = auth_ok
        self.profile = tmp_path / "toxagent.json"
        self.profile.write_text("{}")
        self.root = tmp_path / "hosts"
        self.ports = iter(range(41000, 41100))

    async def spawn(self, argv, env, cwd, log_path):
        process = FakeProcess()
        self.spawned.append({"argv": argv, "env": env, "cwd": cwd, "process": process})
        return process

    def client(self, base_url: str, password: str) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else None
            self.requests.append((base_url, request.method, request.url.path, body))
            assert request.headers["authorization"].startswith("Basic ")
            if request.url.path == "/agent":
                return httpx.Response(200, json=[{"name": n} for n in self.agents])
            if request.url.path.startswith("/auth/"):
                return httpx.Response(200, json=self.auth_ok)
            return httpx.Response(404)

        return httpx.AsyncClient(
            base_url=base_url, auth=("opencode", password), transport=httpx.MockTransport(handler)
        )

    def pool(self, **overrides) -> OpenCodeHostPool:
        options = dict(
            binary="/opt/opencode", profile_path=self.profile, root=self.root,
            required_agents=("toxagent", "toxagent-report"), max_hosts=2, idle_ttl_s=600,
            startup_timeout_s=2, spawner=self.spawn, client_factory=self.client,
            port_picker=lambda: next(self.ports),
        )
        options.update(overrides)
        return OpenCodeHostPool(**options)


def credential(connection="a", **overrides) -> HostCredential:
    values = dict(
        connection_id="con_" + connection * 32, owner_id="owner-" + connection,
        provider_id="openai", model_id="gpt-x", auth_mode="api_key", secret="sk-" + connection,
        secret_ref="secret_" + "1" * 32,
    )
    values.update(overrides)
    return HostCredential(**values)


async def test_a_host_is_isolated_and_receives_only_its_own_credential(tmp_path):
    h = Harness(tmp_path)
    pool = h.pool()
    host = await pool.acquire(credential("a"))
    spawned = h.spawned[0]
    env = spawned["env"]
    assert spawned["argv"][1:] == ["serve", "--pure", "--hostname", "127.0.0.1", "--port", "41000"]
    for name in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        assert Path(env[name]).is_relative_to(host.root), name
    assert env["OPENCODE_CONFIG"] == str(h.profile)
    assert len(env["OPENCODE_SERVER_PASSWORD"]) >= 32
    assert "sk-a" not in json.dumps(env)  # catalog providers take it via /auth, not env
    assert oct(host.root.stat().st_mode & 0o777) == "0o700"
    puts = [r for r in h.requests if r[1] == "PUT"]
    assert puts == [("http://127.0.0.1:41000", "PUT", "/auth/openai", {"type": "api", "key": "sk-a"})]
    assert host.runtime_provider_id == "openai"


async def test_an_unchanged_connection_reuses_its_host_and_a_changed_one_replaces_it(tmp_path):
    h = Harness(tmp_path)
    pool = h.pool()
    first = await pool.acquire(credential("a"))
    await pool.release(first)
    again = await pool.acquire(credential("a"))
    assert again is first and len(h.spawned) == 1
    await pool.release(again)
    changed = await pool.acquire(credential("a", secret="sk-rotated"))
    assert changed is not first and len(h.spawned) == 2
    assert h.spawned[0]["process"].returncode is not None  # old one stopped
    assert not first.root.exists() or first.root == changed.root


async def test_a_connection_is_not_replaced_under_a_running_turn(tmp_path):
    h = Harness(tmp_path)
    pool = h.pool()
    await pool.acquire(credential("a"))
    with pytest.raises(RuntimeUnavailable, match="still active"):
        await pool.acquire(credential("a", model_id="gpt-y"))


async def test_a_full_pool_evicts_an_idle_host_and_refuses_when_all_are_busy(tmp_path):
    h = Harness(tmp_path)
    pool = h.pool(max_hosts=2)
    a = await pool.acquire(credential("a"))
    b = await pool.acquire(credential("b"))
    with pytest.raises(RuntimeUnavailable, match="busy"):
        await pool.acquire(credential("c"))
    await pool.release(a)
    c = await pool.acquire(credential("c"))
    assert pool.size == 2 and c is not a and b.leases == 1
    assert a.process.returncode is not None


async def test_an_idle_host_is_reaped_and_its_files_go_with_it(tmp_path):
    h = Harness(tmp_path)
    pool = h.pool(idle_ttl_s=0)
    host = await pool.acquire(credential("a"))
    host.auth_file.parent.mkdir(parents=True, exist_ok=True)
    host.auth_file.write_text(json.dumps({"openai": {"type": "api", "key": "sk-a"}}))
    await pool.release(host)  # release reaps
    assert pool.size == 0
    assert not host.root.exists()


async def test_a_host_without_the_product_agent_is_refused(tmp_path):
    h = Harness(tmp_path, agents=("build",))
    pool = h.pool()
    with pytest.raises(RuntimeUnavailable, match="product agent"):
        await pool.acquire(credential("a"))
    assert pool.size == 0
    assert h.spawned[0]["process"].returncode is not None


async def test_a_refused_credential_stops_the_host(tmp_path):
    h = Harness(tmp_path, auth_ok=False)
    pool = h.pool()
    with pytest.raises(RuntimeUnavailable, match="refused the credential"):
        await pool.acquire(credential("a"))
    assert pool.size == 0


async def test_a_refreshed_chatgpt_pair_is_written_back_under_the_same_reference(tmp_path):
    h = Harness(tmp_path)
    pool = h.pool()
    written: list[tuple[str, str, str]] = []
    pool.credential_sink = lambda ref, owner, secret: written.append((ref, owner, secret))
    host = await pool.acquire(
        credential("a", auth_mode="chatgpt_subscription", secret=json.dumps(OAUTH))
    )
    assert ("http://127.0.0.1:41000", "PUT", "/auth/openai", OAUTH) in h.requests
    refreshed = {**OAUTH, "access": "a-2", "refresh": "r-2", "expires": 1_900_000_100_000}
    host.auth_file.parent.mkdir(parents=True, exist_ok=True)
    host.auth_file.write_text(json.dumps({"openai": refreshed}))
    await pool.release(host)
    assert len(written) == 1
    ref, owner, secret = written[0]
    assert (ref, owner) == ("secret_" + "1" * 32, "owner-a")
    assert json.loads(secret) == refreshed
    # The host now counts the refreshed pair as current: no restart, no rewrite.
    again = await pool.acquire(
        credential("a", auth_mode="chatgpt_subscription", secret=secret)
    )
    assert again is host
    await pool.release(again)
    assert len(written) == 1


async def test_an_api_key_host_never_writes_a_credential_back(tmp_path):
    h = Harness(tmp_path)
    pool = h.pool()
    written: list = []
    pool.credential_sink = lambda *args: written.append(args)
    host = await pool.acquire(credential("a"))
    host.auth_file.parent.mkdir(parents=True, exist_ok=True)
    host.auth_file.write_text(json.dumps({"openai": {"type": "api", "key": "sk-other"}}))
    await pool.release(host)
    assert written == []


async def test_closing_the_pool_stops_every_host(tmp_path):
    h = Harness(tmp_path)
    pool = h.pool()
    await pool.acquire(credential("a"))
    await pool.acquire(credential("b"))
    await pool.aclose()
    assert pool.size == 0
    assert all(item["process"].returncode is not None for item in h.spawned)


# -- a deployment with no AI of its own ----------------------------------------


def _fake_binary(tmp_path: Path, version: str) -> str:
    binary = tmp_path / "opencode"
    binary.write_text(f"#!/bin/sh\necho {version}\n")
    binary.chmod(0o755)
    return str(binary)


async def test_preflight_passes_with_the_pinned_binary_and_both_agents(tmp_path):
    h = Harness(tmp_path)
    h.profile.write_text(json.dumps({"agent": {"toxagent": {}, "toxagent-report": {}}}))
    pool = h.pool(binary=_fake_binary(tmp_path, "1.17.11"))
    ok, detail, missing = await pool.preflight("1.17.11")
    assert ok and missing == () and "1.17.11" in detail
    assert h.spawned == []  # nothing started just to check


async def test_preflight_refuses_another_version_and_a_missing_binary(tmp_path):
    h = Harness(tmp_path)
    h.profile.write_text(json.dumps({"agent": {"toxagent": {}}}))
    wrong = h.pool(binary=_fake_binary(tmp_path, "1.18.0"))
    ok, detail, missing = await wrong.preflight("1.17.11")
    assert not ok and "1.18.0" in detail and missing == ("toxagent-report",)
    absent = h.pool(binary=str(tmp_path / "nope"))
    assert (await absent.preflight("1.17.11"))[0] is False


async def test_with_no_shared_host_health_is_the_pools_preflight(tmp_path):
    from toxagent.harness.adapters.opencode_v1 import OpenCodeV1Provider
    from toxagent.platform.config import RuntimeSettings

    h = Harness(tmp_path)
    h.profile.write_text(json.dumps({"agent": {"toxagent": {}, "toxagent-report": {}}}))
    pool = h.pool(binary=_fake_binary(tmp_path, "1.17.11"))

    def refuse(request):
        raise AssertionError("a deployment with no shared host must not be probed")

    shared = httpx.AsyncClient(base_url="http://nowhere", transport=httpx.MockTransport(refuse))
    provider = OpenCodeV1Provider(
        RuntimeSettings(kind="opencode", opencode_directory=str(tmp_path / "ws"),
                        opencode_shared_host=False),
        client=shared, hosts=pool,
    )
    health = await provider.health()
    assert health.healthy and health.missing_agents == ()
    await shared.aclose()
