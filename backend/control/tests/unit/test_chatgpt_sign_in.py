"""ChatGPT Plus/Pro sign-in through OpenCode's device-code OAuth (connections/chatgpt.py).

The flow G1_Web_Design runs, checked here without a browser or an OpenAI
account: the server picks the headless method by label, accepts only an
``https://auth.openai.com`` link, waits for the approval, and keeps the token
pair in memory (never on a host's disk) until the user picks a model.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from toxagent.connections.chatgpt import (
    ChatGPTEnrollment,
    ChatGPTUnavailable,
    EnrollmentError,
    availability,
    headless_method,
    validate_authorization,
)

pytestmark = pytest.mark.anyio

OAUTH = {"type": "oauth", "refresh": "r-1", "access": "a-1", "expires": 1_900_000_000_000, "accountId": "acc"}
METHODS = {"openai": [
    {"type": "oauth", "label": "ChatGPT Pro/Plus (browser)"},
    {"type": "oauth", "label": "ChatGPT Pro/Plus (headless)"},
    {"type": "api", "label": "Manually enter API Key"},
]}
AUTHORIZE = {"url": "https://auth.openai.com/codex/device", "method": "auto",
             "instructions": "Enter code: ABCD-12345"}


class FakeHost:
    def __init__(self, root: Path, *, authorize=AUTHORIZE, callback_ok=True, gate=None):
        self.root = root
        self.workspace = root / "workspace"
        self.calls: list[tuple[str, str, object]] = []
        self.gate = gate or asyncio.Event()
        self.callback_ok = callback_ok
        self.authorize = authorize
        self.client = httpx.AsyncClient(base_url="http://host", transport=httpx.MockTransport(self._handle))

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        if request.url.path == "/provider/auth":
            return httpx.Response(200, json=METHODS)
        if request.url.path == "/provider/openai/oauth/authorize":
            return httpx.Response(200, json=self.authorize)
        if request.url.path == "/provider/openai/oauth/callback":
            await self.gate.wait()
            if self.callback_ok:
                self.auth_file.parent.mkdir(parents=True, exist_ok=True)
                self.auth_file.write_text(json.dumps({"openai": OAUTH}))
            return httpx.Response(200, json=self.callback_ok)
        if request.url.path == "/config/providers":
            return httpx.Response(200, json={"providers": [
                {"id": "openai", "models": {"gpt-5.5": {"name": "GPT-5.5"}, "gpt-5.5-mini": {}}},
                {"id": "opencode", "models": {"big-pickle": {}}},
            ]})
        return httpx.Response(404)

    @property
    def auth_file(self) -> Path:
        return self.root / "data" / "opencode" / "auth.json"

    def read_auth(self, provider_id):
        try:
            return json.loads(self.auth_file.read_text()).get(provider_id)
        except OSError:
            return None


class FakePool:
    def __init__(self, host: FakeHost):
        self.host = host
        self.discarded: list[FakeHost] = []

    async def enrollment_host(self, owner_id):
        return self.host

    async def discard(self, host):
        self.discarded.append(host)
        # The host's files go with it, as the real pool deletes its root.
        if host.auth_file.exists():
            host.auth_file.unlink()


class FakeConnections:
    def __init__(self):
        self.created: list[dict] = []

    async def create_subscription(self, **kwargs):
        self.created.append(kwargs)
        return kwargs


AVAILABLE = availability(mode="local", hosts_enabled=True, egress_policy="local")


async def _settle(enrollment, owner="owner-a", want="connected"):
    for _ in range(100):
        status = await enrollment.status(owner)
        if status["state"] == want:
            return status
        await asyncio.sleep(0.01)
    return status


async def test_the_full_sign_in_stores_the_pair_only_when_a_model_is_chosen(tmp_path):
    host = FakeHost(tmp_path)
    pool, connections = FakePool(host), FakeConnections()
    enrollment = ChatGPTEnrollment(pool, connections, AVAILABLE)
    started = await enrollment.start("owner-a")
    assert started["state"] == "pending"
    assert started["verification_url"] == "https://auth.openai.com/codex/device"
    assert started["instructions"] == "Enter code: ABCD-12345"
    # The headless method, found by label: index 1, not a hard-coded guess.
    assert ("POST", "/provider/openai/oauth/authorize", {"method": 1}) in host.calls
    host.gate.set()
    status = await _settle(enrollment)
    assert status["state"] == "connected"
    assert [m["id"] for m in status["models"]] == ["gpt-5.5", "gpt-5.5-mini"]
    assert "verification_url" not in status
    assert "r-1" not in json.dumps(status)
    # Host stopped and its auth file gone as soon as the pair was read.
    assert pool.discarded == [host] and not host.auth_file.exists()
    assert connections.created == []
    await enrollment.complete("owner-a", model_id="gpt-5.5")
    created = connections.created[0]
    assert created["owner_id"] == "owner-a" and created["model_id"] == "gpt-5.5"
    assert json.loads(created["credential"]) == OAUTH
    assert (await enrollment.status("owner-a"))["state"] == "none"


async def test_completing_before_approval_is_refused(tmp_path):
    enrollment = ChatGPTEnrollment(FakePool(FakeHost(tmp_path)), FakeConnections(), AVAILABLE)
    await enrollment.start("owner-a")
    with pytest.raises(EnrollmentError) as excinfo:
        await enrollment.complete("owner-a", model_id="gpt-5.5")
    assert excinfo.value.kind == "not_connected"
    await enrollment.cancel("owner-a")


async def test_one_owner_cannot_complete_anothers_sign_in(tmp_path):
    host = FakeHost(tmp_path)
    enrollment = ChatGPTEnrollment(FakePool(host), FakeConnections(), AVAILABLE)
    await enrollment.start("owner-a")
    host.gate.set()
    await _settle(enrollment)
    with pytest.raises(EnrollmentError):
        await enrollment.complete("owner-b", model_id="gpt-5.5")


async def test_a_rejected_approval_is_reported_and_nothing_is_stored(tmp_path):
    host = FakeHost(tmp_path, callback_ok=False)
    connections = FakeConnections()
    enrollment = ChatGPTEnrollment(FakePool(host), connections, AVAILABLE)
    await enrollment.start("owner-a")
    host.gate.set()
    status = await _settle(enrollment, want="failed")
    assert status["state"] == "failed" and "not completed" in status["error"]
    assert connections.created == []


async def test_cancel_stops_the_wait_and_the_host(tmp_path):
    host = FakeHost(tmp_path)
    pool = FakePool(host)
    enrollment = ChatGPTEnrollment(pool, FakeConnections(), AVAILABLE)
    await enrollment.start("owner-a")
    await enrollment.cancel("owner-a")
    assert (await enrollment.status("owner-a"))["state"] == "none"
    assert pool.discarded == [host]


async def test_an_expired_sign_in_must_be_started_again(tmp_path):
    now = [1000.0]
    host = FakeHost(tmp_path)
    enrollment = ChatGPTEnrollment(
        FakePool(host), FakeConnections(), AVAILABLE, timeout_s=60, clock=lambda: now[0]
    )
    await enrollment.start("owner-a")
    now[0] += 61
    assert (await enrollment.status("owner-a"))["state"] == "expired"


@pytest.mark.parametrize(
    "payload",
    [
        {**AUTHORIZE, "url": "https://evil.example/codex/device"},
        {**AUTHORIZE, "url": "http://auth.openai.com/codex/device"},
        {**AUTHORIZE, "url": "https://user:pw@auth.openai.com/x"},
        {**AUTHORIZE, "url": "https://auth.openai.com:8443/x"},
        {**AUTHORIZE, "instructions": "x" * 2001},
        {**AUTHORIZE, "method": "code"},
        "not an object",
    ],
)
def test_only_an_openai_sign_in_link_is_ever_shown(payload):
    with pytest.raises(EnrollmentError):
        validate_authorization(payload)


async def test_a_bad_link_stops_the_host_before_anything_is_shown(tmp_path):
    host = FakeHost(tmp_path, authorize={**AUTHORIZE, "url": "https://phish.example/"})
    pool = FakePool(host)
    enrollment = ChatGPTEnrollment(pool, FakeConnections(), AVAILABLE)
    with pytest.raises(EnrollmentError):
        await enrollment.start("owner-a")
    assert pool.discarded == [host]


def test_the_headless_method_is_found_by_label_not_position():
    assert headless_method(METHODS["openai"]) == 1
    assert headless_method(list(reversed(METHODS["openai"]))) == 1
    with pytest.raises(EnrollmentError):
        headless_method([{"type": "api", "label": "key"}])


@pytest.mark.parametrize(
    ("mode", "hosts", "egress", "available"),
    [
        ("local", True, "local", True),
        ("local", True, "hosted", False),   # hosted needs an explicit opt-in
        ("hosted", True, "hosted", True),
        ("local", False, "local", False),   # nowhere to run it
        ("off", True, "local", False),
    ],
)
def test_availability_is_a_deployment_decision(mode, hosts, egress, available):
    result = availability(mode=mode, hosts_enabled=hosts, egress_policy=egress)
    assert result.available is available
    assert result.reason


async def test_an_unavailable_deployment_refuses_with_its_reason(tmp_path):
    refused = availability(mode="off", hosts_enabled=True, egress_policy="local")
    enrollment = ChatGPTEnrollment(FakePool(FakeHost(tmp_path)), FakeConnections(), refused)
    with pytest.raises(ChatGPTUnavailable, match="turned off"):
        await enrollment.start("owner-a")
