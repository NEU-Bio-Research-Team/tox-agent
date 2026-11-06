"""Users connect their own AI, through the real API (routes, service, store).

Covers the surface G1_Web_Design taught: list the models a key can use before
saving it, switch a saved connection's model, keep every credential out of
every response, and offer ChatGPT Plus/Pro sign-in only where the deployment
can run it — through the sign-in flow, never by pasting a token.
"""
from __future__ import annotations

import json
import socket

import httpx
import pytest
from cryptography.fernet import Fernet

from tests.support.api import AUTH, OTHER_AUTH, USER_TOKEN, OTHER_TOKEN, api_client, settings
from tests.support.predictor import StubPredictor
from tests.unit.test_chatgpt_sign_in import AVAILABLE, OAUTH, FakeHost, FakePool
from toxagent.connections.chatgpt import ChatGPTEnrollment
from toxagent.platform.config import SecuritySettings

pytestmark = pytest.mark.anyio


def _security(**overrides) -> SecuritySettings:
    return SecuritySettings(
        capability_secret="test-secret-not-for-production",
        static_tokens=(f"{USER_TOKEN}:user-1", f"{OTHER_TOKEN}:user-2"),
        **overrides,
    )


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))],
    )


def _models_transport(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("authorization") != "Bearer sk-good":
            return httpx.Response(401)
        return httpx.Response(200, json={"data": [{"id": "gpt-a", "created": 2}, {"id": "gpt-b", "created": 1}]})

    return httpx.MockTransport(handler)


async def test_the_provider_list_says_why_chatgpt_sign_in_is_not_offered(db, tmp_path):
    async with api_client(db, StubPredictor(), config=settings(secrets_dir=tmp_path / "secrets")) as client:
        catalogue = (await client.get("/v1/model-connections:providers", headers=AUTH)).json()
        openai = next(p for p in catalogue["providers"] if p["provider_id"] == "openai")
        assert "chatgpt_subscription" not in openai["auth_modes"]
        assert openai["chatgpt_oauth"]["available"] is False
        assert "TOXAGENT_OWNER_RUNTIME_HOSTS" in openai["chatgpt_oauth"]["reason"]
        assert {p["provider_id"] for p in catalogue["providers"]} >= {"anthropic", "deepseek"}

        started = await client.post("/v1/model-connections:chatgpt-start", headers=AUTH)
        assert started.status_code == 503
        assert started.json()["error"]["code"] == "capability_unavailable"

        pasted = await client.post("/v1/model-connections", headers=AUTH, json={
            "provider_id": "openai", "model_id": "gpt-x", "auth_mode": "chatgpt_subscription",
            "credential": json.dumps(OAUTH),
        })
        assert pasted.status_code == 400
        assert "sign-in flow" in pasted.json()["error"]["message"]
        assert not any((tmp_path / "secrets").iterdir())


async def test_a_key_lists_its_models_without_being_stored(db, tmp_path):
    seen: list[httpx.Request] = []
    async with api_client(db, StubPredictor(), config=settings(secrets_dir=tmp_path / "secrets")) as client:
        client.app.state.connections._catalog._transport = _models_transport(seen)
        listed = await client.post("/v1/model-connections:models", headers=AUTH, json={
            "provider_id": "openai", "credential": "sk-good",
        })
        assert listed.status_code == 200
        assert [m["id"] for m in listed.json()["models"]] == ["gpt-a", "gpt-b"]
        assert not any((tmp_path / "secrets").iterdir())

        refused = await client.post("/v1/model-connections:models", headers=AUTH, json={
            "provider_id": "openai", "credential": "sk-bad",
        })
        assert refused.status_code == 400
        assert refused.json()["error"]["details"]["model_list_failure"] == "unauthorized"
        assert "sk-bad" not in refused.text


async def test_a_saved_connection_lists_its_models_and_switches_model(db, tmp_path):
    seen: list[httpx.Request] = []
    async with api_client(db, StubPredictor(), config=settings(secrets_dir=tmp_path / "secrets")) as client:
        client.app.state.connections._catalog._transport = _models_transport(seen)
        created = (await client.post("/v1/model-connections", headers=AUTH, json={
            "provider_id": "openai", "model_id": "gpt-a", "auth_mode": "api_key",
            "credential": "sk-good",
        })).json()
        assert "sk-good" not in json.dumps(created)
        cid = created["connection_id"]

        models = await client.get(f"/v1/model-connections/{cid}/models", headers=AUTH)
        assert [m["id"] for m in models.json()["models"]] == ["gpt-a", "gpt-b"]
        assert seen[-1].headers["authorization"] == "Bearer sk-good"

        changed = await client.patch(f"/v1/model-connections/{cid}", headers=AUTH,
                                     json={"model_id": "gpt-b"})
        assert changed.status_code == 200
        body = changed.json()
        assert body["model_id"] == "gpt-b" and body["status"] == "untested"
        assert body["display_name"] == "openai · gpt-b" and body["has_credential"] is True

        # Another user can neither see, list nor change it.
        assert (await client.get(f"/v1/model-connections/{cid}/models", headers=OTHER_AUTH)).status_code == 404
        assert (await client.patch(f"/v1/model-connections/{cid}", headers=OTHER_AUTH,
                                   json={"model_id": "gpt-a"})).status_code == 404


async def test_with_an_encryption_key_a_stored_key_is_not_readable_from_disk(db, tmp_path):
    key = Fernet.generate_key().decode()
    config = settings(secrets_dir=tmp_path / "secrets", security=_security(secret_encryption_key=key))
    async with api_client(db, StubPredictor(), config=config) as client:
        created = await client.post("/v1/model-connections", headers=AUTH, json={
            "provider_id": "anthropic", "model_id": "claude-x", "auth_mode": "api_key",
            "credential": "sk-ant-private",
        })
        assert created.status_code == 201
        files = list((tmp_path / "secrets").iterdir())
        assert len(files) == 1
        assert "sk-ant-private" not in files[0].read_text()


async def test_a_finished_chatgpt_sign_in_becomes_a_ready_connection(db, tmp_path):
    key = Fernet.generate_key().decode()
    config = settings(secrets_dir=tmp_path / "secrets", security=_security(secret_encryption_key=key))
    host = FakeHost(tmp_path / "host")
    async with api_client(db, StubPredictor(), config=config) as client:
        app = client.app
        app.state.chatgpt = ChatGPTEnrollment(FakePool(host), app.state.connections, AVAILABLE)

        started = (await client.post("/v1/model-connections:chatgpt-start", headers=AUTH)).json()
        assert started["state"] == "pending"
        assert started["verification_url"].startswith("https://auth.openai.com/")
        early = await client.post("/v1/model-connections:chatgpt-complete", headers=AUTH,
                                  json={"model_id": "gpt-5.5"})
        assert early.status_code == 409

        host.gate.set()
        for _ in range(100):
            status = (await client.get("/v1/model-connections:chatgpt", headers=AUTH)).json()
            if status["state"] == "connected":
                break
            import asyncio

            await asyncio.sleep(0.01)
        assert status["state"] == "connected"
        assert "a-1" not in json.dumps(status) and "r-1" not in json.dumps(status)
        # The other user has no sign-in in progress.
        assert (await client.get("/v1/model-connections:chatgpt", headers=OTHER_AUTH)).json() == {"state": "none"}

        done = await client.post("/v1/model-connections:chatgpt-complete", headers=AUTH,
                                 json={"model_id": "gpt-5.5"})
        assert done.status_code == 201
        connection = done.json()
        assert connection["auth_mode"] == "chatgpt_subscription"
        assert connection["status"] == "ready" and connection["has_credential"] is True
        assert connection["display_name"] == "ChatGPT · gpt-5.5"
        assert "r-1" not in done.text

        listed = (await client.get("/v1/model-connections", headers=AUTH)).json()["connections"]
        assert [c["connection_id"] for c in listed] == [connection["connection_id"]]
        stored = list((tmp_path / "secrets").iterdir())
        assert len(stored) == 1 and "r-1" not in stored[0].read_text()

        deleted = await client.delete(f"/v1/model-connections/{connection['connection_id']}", headers=AUTH)
        assert deleted.status_code == 204
        assert not any((tmp_path / "secrets").iterdir())
