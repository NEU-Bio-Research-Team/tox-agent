"""A chosen AI profile decides which endpoint and key a run actually uses.

I12: `_resolve_ai_profile` returned the provider/model pair and stopped. The
base URL and the credential stayed in the database, so the runtime dispatched
under whatever authentication its host happened to have. A profile that passed
"Test connection" therefore proved nothing about which account a run would
bill, whether it would reach the server the user chose, or — with two owners
on one deployment — whose key it would spend.

Two owners, two different endpoints and keys. Each owner's turn must carry
that owner's values, and a runtime that cannot be given them must refuse
rather than fall back.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from toxagent.config import RuntimeSettings
from toxagent.connections.model import ConnectionStatus, ModelConnection
from toxagent.connections.secrets import FilesystemSecretStore
from toxagent.domain.errors import RuntimeUnavailable
from toxagent.domain.runtime import AuthMode
from toxagent.harness.gateway import AgentRuntimeGateway

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)

OWNERS = {
    "owner-a": ("https://a.example.com/v1", "sk-owner-a-key"),
    "owner-b": ("https://b.example.com/v1", "sk-owner-b-key"),
}


class Recorder:
    """Captures the spec a create_session was called with."""

    kind = "scripted"
    specs: list = []

    def __init__(self) -> None:
        self.specs = []


async def _store(db, secrets, owner_id: str) -> ModelConnection:
    base_url, key = OWNERS[owner_id]
    ref = secrets.put(owner_id, key)
    connection = ModelConnection.create(
        owner_id=owner_id, provider_id="openai_compatible", model_id="model-x",
        auth_mode=AuthMode.API_KEY, credential_ref=ref, base_url=base_url, now=NOW,
    )
    connection = replace(connection, status=ConnectionStatus.READY)
    async with db.unit_of_work() as uow:
        await uow.model_connections.add(connection)
        await uow.commit()
    return connection


def _gateway(db, secrets, provider=None):
    return AgentRuntimeGateway(
        db, registry=None, capability_tokens=None,
        provider=provider or Recorder(), settings=RuntimeSettings(kind="scripted"),
        secrets=secrets,
    )


class _Context:
    """The two fields _resolve_ai_profile reads."""

    def __init__(self, actor, ai_profile_id):
        self.actor = actor
        self.ai_profile_id = ai_profile_id


class _Actor:
    def __init__(self, subject_id):
        self.subject_id = subject_id


async def test_each_owner_gets_their_own_endpoint_and_key(db, tmp_path):
    secrets = FilesystemSecretStore(tmp_path / "secrets")
    gateway = _gateway(db, secrets)

    for owner_id in OWNERS:
        connection = await _store(db, secrets, owner_id)
        resolved = await gateway._resolve_ai_profile(
            _Context(_Actor(owner_id), connection.id)
        )
        expected_url, expected_key = OWNERS[owner_id]
        assert resolved.base_url == expected_url
        assert resolved.credential == expected_key
        assert resolved.connection_id == connection.id


async def test_one_owner_cannot_resolve_another_owners_profile(db, tmp_path):
    secrets = FilesystemSecretStore(tmp_path / "secrets")
    gateway = _gateway(db, secrets)
    a = await _store(db, secrets, "owner-a")

    with pytest.raises(RuntimeUnavailable):
        await gateway._resolve_ai_profile(_Context(_Actor("owner-b"), a.id))


async def test_an_untested_profile_is_refused(db, tmp_path):
    secrets = FilesystemSecretStore(tmp_path / "secrets")
    gateway = _gateway(db, secrets)
    ref = secrets.put("owner-a", "sk-untested")
    connection = ModelConnection.create(
        owner_id="owner-a", provider_id="openai_compatible", model_id="model-x",
        auth_mode=AuthMode.API_KEY, credential_ref=ref,
        base_url="https://a.example.com/v1", now=NOW,
    )
    async with db.unit_of_work() as uow:
        await uow.model_connections.add(connection)
        await uow.commit()

    with pytest.raises(RuntimeUnavailable, match="connection test"):
        await gateway._resolve_ai_profile(_Context(_Actor("owner-a"), connection.id))


async def test_a_gateway_with_no_secret_store_refuses_rather_than_using_ambient_auth(db, tmp_path):
    """The exact failure mode I12 describes, made impossible.

    Dispatching here would run the turn under the runtime host's own
    credentials while the audit row named this owner's profile.
    """
    secrets = FilesystemSecretStore(tmp_path / "secrets")
    connection = await _store(db, secrets, "owner-a")
    blind = AgentRuntimeGateway(
        db, registry=None, capability_tokens=None,
        provider=Recorder(), settings=RuntimeSettings(kind="scripted"),
        secrets=None,
    )
    with pytest.raises(RuntimeUnavailable, match="runtime's own authentication"):
        await blind._resolve_ai_profile(_Context(_Actor("owner-a"), connection.id))


async def test_no_profile_falls_back_to_the_deployment_default_with_no_credential(db, tmp_path):
    gateway = _gateway(db, FilesystemSecretStore(tmp_path / "secrets"))
    resolved = await gateway._resolve_ai_profile(_Context(_Actor("owner-a"), None))
    assert resolved.connection_id is None
    assert resolved.credential is None
    assert resolved.auth_mode is AuthMode.NONE


# --- the credential is identified in the audit trail, never stored -----------

def test_the_fingerprint_identifies_a_credential_without_revealing_it():
    from toxagent.harness.provider import RuntimeSessionSpec

    def spec(credential):
        return RuntimeSessionSpec(
            session_id="ses_x", run_id="run_x", provider_id="p", model_id="m",
            profile="report_qa", system_prompt="", system_prompt_hash="",
            tool_schema=(), tool_schema_hash="", mcp_url="", max_steps=8,
            deadline_at=NOW, provider_credential=credential,
        )

    key = "sk-owner-a-key"
    fingerprint = spec(key).credential_fingerprint()
    assert key not in fingerprint
    assert fingerprint.startswith("sha256:")
    # Same key, same fingerprint; different key, different fingerprint.
    assert fingerprint == spec(key).credential_fingerprint()
    assert fingerprint != spec("sk-owner-b-key").credential_fingerprint()
    assert spec(None).credential_fingerprint() == ""


# --- the runtime that cannot use it says so ----------------------------------

async def test_the_opencode_adapter_refuses_a_credential_it_cannot_inject():
    """OpenCode V1 takes {providerID, modelID} and reads its key from the
    host's auth store. Running anyway would spend the wrong account's key."""
    from toxagent.harness.adapters.opencode_v1 import OpenCodeV1Provider
    from toxagent.harness.provider import RuntimeSessionSpec

    provider = OpenCodeV1Provider(RuntimeSettings(kind="opencode"))
    spec = RuntimeSessionSpec(
        session_id="ses_x", run_id="run_x", provider_id="openai_compatible",
        model_id="model-x", profile="report_qa", system_prompt="",
        system_prompt_hash="", tool_schema=(), tool_schema_hash="", mcp_url="",
        max_steps=8, deadline_at=NOW,
        connection_id="con_x", provider_credential="sk-owner-a-key",
        provider_base_url="https://a.example.com/v1", auth_mode="api_key",
    )
    with pytest.raises(RuntimeUnavailable) as excinfo:
        await provider.create_session(spec)
    assert "will not fall back" in str(excinfo.value)
    assert "sk-owner-a-key" not in str(excinfo.value)
