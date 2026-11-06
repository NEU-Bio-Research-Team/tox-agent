"""Private hosts against the real, pinned OpenCode binary (live_runtime).

Skipped unless ``TOXAGENT_LIVE_OPENCODE=1`` and OpenCode 1.17.11 is installed.
No account or paid model is used: the "provider" is a local OpenAI-compatible
server that records the Authorization header it receives, and the ChatGPT
check stops at the sign-in link (nothing is approved).

What the doubles elsewhere cannot prove, and this does:

- OpenCode really forwards *this connection's* key to the endpoint, for both
  a custom endpoint (key via environment) and a catalog provider (key via
  ``PUT /auth``) pointed at the same local server;
- the host runs the product agent from the checked-in profile;
- the host's management API refuses requests without its password;
- the credential file is gone once the host stops;
- the device-code sign-in hands back an ``auth.openai.com`` link.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from toxagent.connections.chatgpt import ChatGPTEnrollment, availability
from toxagent.connections.runtime_install import HostCredential
from toxagent.harness.owner_hosts import OpenCodeHostPool
from toxagent.platform.config import PACKAGE_ROOT

BINARY = os.environ.get("TOXAGENT_OPENCODE_BINARY") or shutil.which("opencode") or str(
    Path.home() / ".opencode" / "bin" / "opencode"
)


def _binary_ok() -> bool:
    try:
        out = subprocess.run([BINARY, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.stdout.strip() == "1.17.11"


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.live_runtime,
    pytest.mark.skipif(
        os.environ.get("TOXAGENT_LIVE_OPENCODE") != "1" or not _binary_ok(),
        reason="set TOXAGENT_LIVE_OPENCODE=1 with OpenCode 1.17.11 installed",
    ),
]


@pytest.fixture
def model_server():
    seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            seen.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "pong"},
                              "finish_reason": None}]},
                {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
            ):
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen
    server.shutdown()


def _pool(tmp_path: Path) -> OpenCodeHostPool:
    return OpenCodeHostPool(
        binary=BINARY,
        profile_path=PACKAGE_ROOT / "agent_profiles" / "opencode" / "toxagent.json",
        root=tmp_path / "hosts",
        required_agents=("toxagent", "toxagent-report"),
        max_hosts=2,
        startup_timeout_s=60,
        request_timeout_s=30,
    )


async def _one_turn(host, model_id: str) -> dict:
    directory = host.workspace / "run_live"
    directory.mkdir(exist_ok=True)
    params = {"directory": str(directory)}
    session = (await host.client.post("/session", params=params, json={"title": "live"})).json()["id"]
    reply = await host.client.post(
        f"/session/{session}/message", params=params, timeout=120,
        json={"model": {"providerID": host.runtime_provider_id, "modelID": model_id},
              "agent": "toxagent", "parts": [{"type": "text", "text": "ping"}]},
    )
    reply.raise_for_status()
    return reply.json()


def _text(data: dict) -> str:
    assert data["info"].get("error") is None, data["info"]
    return "".join(p.get("text", "") for p in data["parts"] if p.get("type") == "text")


@pytest.mark.parametrize(
    ("provider_id", "secret", "path"),
    [
        ("openai_compatible", "sk-live-custom", "/v1/chat/completions"),
        # OpenCode's own OpenAI provider speaks the Responses API, so a gateway
        # behind it must too; the reply of this chat/completions double is not
        # one it accepts, and only the request is checked.
        ("openai", "sk-live-catalog", "/v1/responses"),
    ],
)
async def test_the_users_key_reaches_the_users_endpoint(
    tmp_path, model_server, provider_id, secret, path
):
    base_url, seen = model_server
    pool = _pool(tmp_path)
    try:
        host = await pool.acquire(HostCredential(
            connection_id="con_" + "1" * 32, owner_id="owner-live", provider_id=provider_id,
            model_id="live-model", auth_mode="api_key", base_url=base_url, secret=secret,
        ))
        # No password, no management API.
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{host.port}", trust_env=False) as anon:
            assert (await anon.get("/agent")).status_code == 401
        data = await _one_turn(host, "live-model")
        if path.endswith("chat/completions"):
            assert _text(data) == "pong"
        assert seen and set(seen) == {(path, f"Bearer {secret}")}
        root = host.root
        await pool.release(host)
    finally:
        await pool.aclose()
    assert not root.exists()


async def test_the_chatgpt_sign_in_hands_back_an_openai_device_link(tmp_path):
    pool = _pool(tmp_path)
    enrollment = ChatGPTEnrollment(
        pool, connections=None,
        availability=availability(mode="local", hosts_enabled=True, egress_policy="local"),
    )
    try:
        started = await enrollment.start("owner-live")
        assert started["state"] == "pending"
        assert started["verification_url"].startswith("https://auth.openai.com/")
        assert started["instructions"]
    finally:
        await enrollment.cancel("owner-live")
        await pool.aclose()
    assert pool.size == 0
