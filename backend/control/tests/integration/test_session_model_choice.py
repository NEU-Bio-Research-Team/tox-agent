"""The model a session runs on is picked in the chat, per session.

A connection is a credential for one provider; the composer's model picker
chooses which of that account's models the session uses. The choice is saved
with the session, copied into each run's snapshot at admission, and is what
the gateway hands the runtime — while the provider, endpoint and credential
still come only from the connection.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from toxagent.connections.model import ConnectionStatus, ModelConnection
from toxagent.connections.secrets import FilesystemSecretStore
from toxagent.domain.runtime import AuthMode
from toxagent.harness.gateway import AgentRuntimeGateway
from toxagent.platform.config import RuntimeSettings
from tests.support.api import AUTH, api_client, wait_for_run
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 11, 7, tzinfo=timezone.utc)


async def _ready_connection(db, owner_id: str, *, credential_ref: str | None = None) -> ModelConnection:
    connection = ModelConnection.create(
        owner_id=owner_id, provider_id="openai_compatible", model_id="model-tested",
        auth_mode=AuthMode.API_KEY if credential_ref else AuthMode.NONE,
        credential_ref=credential_ref, base_url="https://a.example.com/v1", now=NOW,
    )
    connection = replace(connection, status=ConnectionStatus.READY)
    async with db.unit_of_work() as uow:
        await uow.model_connections.add(connection)
        await uow.commit()
    return connection


async def _new_session(client) -> str:
    response = await client.post("/v1/sessions", json={}, headers=AUTH)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


async def test_the_chosen_model_is_saved_and_pinned_into_the_run(db):
    connection = await _ready_connection(db, "user-1")
    predictor = StubPredictor()
    async with api_client(db, predictor) as client:
        session_id = await _new_session(client)
        saved = await client.patch(
            f"/v1/sessions/{session_id}/settings",
            json={"ai_profile_id": connection.id, "ai_model_id": "model-picked-in-chat"},
            headers=AUTH,
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["ai_model_id"] == "model-picked-in-chat"
        read = (await client.get(f"/v1/sessions/{session_id}/settings", headers=AUTH)).json()
        assert read == {**read, "ai_profile_id": connection.id, "ai_model_id": "model-picked-in-chat"}

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"molecule": {"smiles": ASPIRIN}, "analysis_options": {"endpoints": ["herg"]}},
            headers=AUTH,
        )
        run_id = submitted.json()["run_id"]
        await wait_for_run(client, session_id, run_id)

        # A later pick changes the session, not a run already admitted.
        await client.patch(
            f"/v1/sessions/{session_id}/settings",
            json={"ai_profile_id": connection.id, "ai_model_id": "another-model"},
            headers=AUTH,
        )
        run = (await client.get(f"/v1/sessions/{session_id}/runs/{run_id}", headers=AUTH)).json()
        assert run["configuration_snapshot"]["ai_model_id"] == "model-picked-in-chat", run


async def test_a_model_without_a_connection_is_refused(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await _new_session(client)
        response = await client.patch(
            f"/v1/sessions/{session_id}/settings",
            json={"ai_profile_id": None, "ai_model_id": "gpt-x"},
            headers=AUTH,
        )
        assert response.status_code == 400, response.text


async def test_a_model_id_is_an_identifier_not_free_text(db):
    connection = await _ready_connection(db, "user-1")
    async with api_client(db, StubPredictor()) as client:
        session_id = await _new_session(client)
        response = await client.patch(
            f"/v1/sessions/{session_id}/settings",
            json={"ai_profile_id": connection.id, "ai_model_id": "gpt x; rm -rf /"},
            headers=AUTH,
        )
        assert response.status_code in {400, 422}, response.text


class _Context:
    def __init__(self, owner_id, ai_profile_id, ai_model_id=None):
        self.actor = type("Actor", (), {"subject_id": owner_id})()
        self.ai_profile_id = ai_profile_id
        self.ai_model_id = ai_model_id


async def test_the_gateway_runs_the_chosen_model_on_the_connections_own_account(db, tmp_path):
    secrets = FilesystemSecretStore(tmp_path / "secrets")
    ref = secrets.put("user-1", "sk-user-1")
    connection = await _ready_connection(db, "user-1", credential_ref=ref)
    gateway = AgentRuntimeGateway(
        db, registry=None, capability_tokens=None, provider=object(),
        settings=RuntimeSettings(kind="scripted"), secrets=secrets,
    )

    picked = await gateway._resolve_ai_profile(_Context("user-1", connection.id, "model-picked"))
    assert picked.model_id == "model-picked"
    assert (picked.provider_id, picked.base_url, picked.credential) == (
        "openai_compatible", "https://a.example.com/v1", "sk-user-1",
    )

    default = await gateway._resolve_ai_profile(_Context("user-1", connection.id))
    assert default.model_id == "model-tested"
