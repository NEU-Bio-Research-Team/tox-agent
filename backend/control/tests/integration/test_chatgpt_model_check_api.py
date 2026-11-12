"""Pick a model, and the model is the one that runs (routes, service, host).

The picker used to save whatever the runtime listed. These drive the real API:
a model is checked with a real turn before a pick stands, a good answer is
remembered (it costs the user's own tokens), a refusal says why, and the list
includes models newer than the runtime's own allowance.
"""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from tests.integration.test_user_ai_connections_api import _security
from tests.support.api import AUTH, OTHER_AUTH, api_client, settings
from tests.support.predictor import StubPredictor
from tests.unit.test_chatgpt_model_check import OAUTH, Host, Pool, _answered

pytestmark = pytest.mark.anyio

REFUSAL = "The 'gpt-5.4-mini' model is not supported when using Codex with a ChatGPT account."


async def _client_with_sign_in(db, tmp_path, host: Host):
    key = Fernet.generate_key().decode()
    config = settings(secrets_dir=tmp_path / "secrets", security=_security(secret_encryption_key=key))
    return api_client(db, StubPredictor(), config=config), host


async def _connect(client, host: Host) -> tuple[str, Pool]:
    service = client.app.state.connections
    pool = Pool(host)
    service._hosts = pool
    connection = await service.create_subscription(
        owner_id="user-1", model_id="gpt-5.3-codex-spark", credential=OAUTH,
        display_name="ChatGPT · gpt-5.3-codex-spark",
    )
    return connection.id, pool


async def test_a_usable_model_is_confirmed_and_not_asked_twice(db, tmp_path):
    host = Host(tmp_path, reply=_answered())
    context, _ = await _client_with_sign_in(db, tmp_path, host)
    async with context as client:
        connection_id, pool = await _connect(client, host)
        url = f"/v1/model-connections/{connection_id}:check-model"
        first = await client.post(url, headers=AUTH, json={"model_id": "gpt-5.6"})
        assert first.status_code == 200, first.text
        assert first.json() == {"model_id": "gpt-5.6", "status": "usable", "message": "the model answered"}
        again = await client.post(url, headers=AUTH, json={"model_id": "gpt-5.6"})
        assert again.json()["status"] == "usable"
        # A working model costs the user's tokens: the second pick is free.
        assert len(pool.acquired) == 1


async def test_a_refused_model_is_an_answer_not_an_error(db, tmp_path):
    host = Host(tmp_path, reply=_answered(error={"name": "APIError", "data": {"message": REFUSAL, "statusCode": 400}}))
    context, _ = await _client_with_sign_in(db, tmp_path, host)
    async with context as client:
        connection_id, pool = await _connect(client, host)
        response = await client.post(
            f"/v1/model-connections/{connection_id}:check-model", headers=AUTH,
            json={"model_id": "gpt-5.4-mini"},
        )
        assert response.status_code == 200
        assert response.json() == {"model_id": "gpt-5.4-mini", "status": "unsupported", "message": REFUSAL}
        # Remembered briefly, so a user hammering the model does not hammer the provider.
        await client.post(f"/v1/model-connections/{connection_id}:check-model", headers=AUTH,
                          json={"model_id": "gpt-5.4-mini"})
        assert len(pool.acquired) == 1


async def test_a_busy_or_unreachable_answer_is_never_remembered(db, tmp_path):
    host = Host(tmp_path, timeout=True)
    context, _ = await _client_with_sign_in(db, tmp_path, host)
    async with context as client:
        connection_id, pool = await _connect(client, host)
        url = f"/v1/model-connections/{connection_id}:check-model"
        assert (await client.post(url, headers=AUTH, json={"model_id": "gpt-5.6"})).json()["status"] == "unavailable"
        host.timeout, host.reply = False, _answered()
        assert (await client.post(url, headers=AUTH, json={"model_id": "gpt-5.6"})).json()["status"] == "usable"
        assert len(pool.acquired) == 2


async def test_another_owner_cannot_check_a_connection_and_a_bad_id_is_a_400(db, tmp_path):
    host = Host(tmp_path, reply=_answered())
    context, _ = await _client_with_sign_in(db, tmp_path, host)
    async with context as client:
        connection_id, pool = await _connect(client, host)
        url = f"/v1/model-connections/{connection_id}:check-model"
        assert (await client.post(url, headers=OTHER_AUTH, json={"model_id": "gpt-5.6"})).status_code == 404
        assert (await client.post(url, headers=AUTH, json={"model_id": "bad model"})).status_code == 400
        assert (await client.post(url, headers=AUTH, json={})).status_code == 400
        assert pool.acquired == []


async def test_deleting_a_connection_forgets_what_was_checked(db, tmp_path):
    host = Host(tmp_path, reply=_answered())
    context, _ = await _client_with_sign_in(db, tmp_path, host)
    async with context as client:
        connection_id, _ = await _connect(client, host)
        await client.post(f"/v1/model-connections/{connection_id}:check-model", headers=AUTH,
                          json={"model_id": "gpt-5.6"})
        assert client.app.state.connections._checks
        assert (await client.delete(f"/v1/model-connections/{connection_id}", headers=AUTH)).status_code == 204
        assert not client.app.state.connections._checks


async def test_the_picker_list_includes_models_newer_than_the_runtimes_own_allowance(db, tmp_path):
    host = Host(tmp_path, reply=_answered())
    context, _ = await _client_with_sign_in(db, tmp_path, host)
    async with context as client:
        connection_id, _ = await _connect(client, host)
        listed = await client.get(f"/v1/model-connections/{connection_id}/models", headers=AUTH)
        assert listed.status_code == 200, listed.text
        by_id = {item["id"]: item for item in listed.json()["models"]}
        assert by_id["gpt-5.5"]["availability"] == "runtime"
        assert by_id["gpt-5.6"]["availability"] == "catalogue"
        assert by_id["gpt-6.1-sol"]["availability"] == "catalogue"
        assert "gpt-image-2" not in by_id and "gpt-realtime-2.1" not in by_id
        assert OAUTH not in listed.text and "refresh" not in listed.text


async def test_a_key_connection_is_not_checked_this_way(db, tmp_path):
    key = Fernet.generate_key().decode()
    config = settings(secrets_dir=tmp_path / "secrets", security=_security(secret_encryption_key=key))
    async with api_client(db, StubPredictor(), config=config) as client:
        created = await client.post(
            "/v1/model-connections", headers=AUTH,
            json={"provider_id": "openai", "model_id": "gpt-4.1", "auth_mode": "api_key",
                  "credential": "sk-private-key"},
        )
        assert created.status_code == 201, created.text
        response = await client.post(
            f"/v1/model-connections/{created.json()['connection_id']}:check-model", headers=AUTH,
            json={"model_id": "gpt-4.1"},
        )
        assert response.status_code == 400 and "ChatGPT" in response.text
        assert "sk-private-key" not in response.text
