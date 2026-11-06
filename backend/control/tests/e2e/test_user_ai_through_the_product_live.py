"""A user's own AI key, through the whole product, on the real runtime (live_runtime).

Skipped unless ``TOXAGENT_LIVE_OPENCODE=1`` and OpenCode 1.17.11 is installed.

The control plane serves on a real port (OpenCode must reach its MCP
endpoint), the deployment's shared OpenCode host has no credential at all,
and the user's "provider" is a local OpenAI-compatible server that records
what it receives. The user creates a connection through the API, tests it,
pins it to a session and asks a question. The run must reach the user's
endpoint with the user's key — which only the private host has — and its
runtime binding must name the connection and the auth mode.

The double answers in plain text and never calls ``submit_grounded_answer``,
so the run does not publish an answer; what this proves is the path, not a
model's quality.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from cryptography.fernet import Fernet

from tests.contract.test_owner_hosts_live import BINARY, _binary_ok
from tests.support.api import AUTH, USER_TOKEN, settings
from tests.support.predictor import ASPIRIN, StubPredictor
from toxagent.platform.config import PACKAGE_ROOT, RuntimeSettings, SecuritySettings

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.live_runtime,
    pytest.mark.skipif(
        os.environ.get("TOXAGENT_LIVE_OPENCODE") != "1" or not _binary_ok(),
        reason="set TOXAGENT_LIVE_OPENCODE=1 with OpenCode 1.17.11 installed",
    ),
]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def model_server():
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            seen.append({
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "tools": [t.get("function", {}).get("name") for t in body.get("tools") or []],
            })
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": '{"ok": true}'},
                              "finish_reason": None}]},
                {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
            ):
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen
    server.shutdown()


@pytest.fixture
def shared_host(tmp_path: Path):
    """The deployment's own runtime host, deliberately without any credential."""
    root = tmp_path / "shared"
    for name in ("home", "data", "config", "state", "cache", "ws"):
        (root / name).mkdir(parents=True)
    port = _free_port()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(root / "home"),
        "XDG_DATA_HOME": str(root / "data"), "XDG_CONFIG_HOME": str(root / "config"),
        "XDG_STATE_HOME": str(root / "state"), "XDG_CACHE_HOME": str(root / "cache"),
        "OPENCODE_CONFIG": str(PACKAGE_ROOT / "agent_profiles" / "opencode" / "toxagent.json"),
    }
    process = subprocess.Popen(
        [BINARY, "serve", "--pure", "--hostname", "127.0.0.1", "--port", str(port)],
        env=env, cwd=root / "ws", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/agent", params={"directory": str(root / "ws")}, timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        import time

        time.sleep(0.25)
    yield f"http://127.0.0.1:{port}", root / "ws"
    process.terminate()
    process.wait(timeout=10)
    assert not (root / "data" / "opencode" / "auth.json").exists(), "the shared host was given a credential"


async def test_a_users_key_runs_the_agent_on_the_users_endpoint(db, tmp_path, model_server, shared_host):
    import uvicorn

    from toxagent.api.app import create_app

    base_url, seen = model_server
    shared_url, shared_ws = shared_host
    port = _free_port()
    config = settings(
        secrets_dir=tmp_path / "secrets",
        runtime=RuntimeSettings(
            kind="opencode", opencode_base_url=shared_url, opencode_directory=str(shared_ws),
            opencode_create_run_directories=True, provider_id="openai", model_id="deployment-model",
            owner_hosts=True, opencode_binary=BINARY, owner_hosts_root=str(tmp_path / "hosts"),
            turn_deadline_s=90,
        ),
        security=SecuritySettings(
            capability_secret="live-test-secret-not-for-production-0001",
            static_tokens=(f"{USER_TOKEN}:user-1",),
            mcp_runtime_url=f"http://127.0.0.1:{port}/internal/mcp",
            secret_encryption_key=Fernet.generate_key().decode(),
        ),
    )
    app = create_app(config, database=db, predictor=StubPredictor().client())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", headers=AUTH, timeout=30) as client:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(0.05)
            created = (await client.post("/v1/model-connections", json={
                "provider_id": "openai_compatible", "model_id": "live-model", "auth_mode": "api_key",
                "base_url": base_url, "credential": "sk-user-live",
            })).json()
            tested = await client.post(f"/v1/model-connections/{created['connection_id']}:test")
            assert tested.json()["status"] == "ready", tested.text

            session_id = (await client.post("/v1/sessions", json={})).json()["session_id"]
            settings_now = (await client.get(f"/v1/sessions/{session_id}/settings")).json()
            patched = await client.patch(f"/v1/sessions/{session_id}/settings", json={
                **settings_now, "ai_profile_id": created["connection_id"],
            })
            assert patched.status_code == 200, patched.text
            analysis = (await client.post(f"/v1/sessions/{session_id}/messages",
                                          json={"molecule": {"smiles": ASPIRIN}})).json()
            for _ in range(200):
                state = (await client.get(f"/v1/sessions/{session_id}/runs/{analysis['run_id']}")).json()
                if state["status"] in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.1)
            assert state["status"] == "completed", state
            submitted = await client.post(f"/v1/sessions/{session_id}/messages", json={
                "intent_hint": "auto",
                "content": [{"type": "text", "text": "Does this compound block hERG?"}],
            })
            assert submitted.status_code == 202, submitted.text
            run_id = submitted.json()["run_id"]
            for _ in range(600):
                run = (await client.get(f"/v1/sessions/{session_id}/runs/{run_id}")).json()
                if run["status"] in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.25)
            assert run["status"] in {"completed", "failed"}, run["status"]

        # The probe made one call; the agent turn made the rest, with the
        # product's tools and the user's key — from the private host, since
        # the shared host has no key to send.
        turns = [call for call in seen if call["tools"] and call["tools"] != ["probe_echo"]]
        assert turns, seen
        assert {call["authorization"] for call in seen} == {"Bearer sk-user-live"}
        assert any(name.startswith("toxagent_") for name in turns[0]["tools"]), turns[0]["tools"]

        async with db.unit_of_work() as uow:
            bindings = await uow.runtime_bindings.list_for_run(session_id, run_id)
        assert bindings, "the run never bound a runtime"
        manifest = bindings[0].manifest()
        assert manifest["connection_id"] == created["connection_id"]
        assert manifest["auth_mode"] == "api_key"
        assert manifest["provider_id"] == "openai_compatible"
        stored = list((tmp_path / "secrets").iterdir())
        assert len(stored) == 1 and "sk-user-live" not in stored[0].read_text()
    finally:
        server.should_exit = True
        await serving
