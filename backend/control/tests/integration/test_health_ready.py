"""`/health/ready` answers three separate questions (I05).

Core readiness decides the status code. Capability availability is reported
per intent with a reason. `mode` says which product this deployment is. The
endpoint previously merged the first two and probed no database at all, so a
control plane with an unreachable database answered `ready: true`.
"""
from __future__ import annotations

import pytest

from toxagent.domain.run import Intent
from toxagent.harness.adapters.scripted import ScriptedRuntimeProvider
from toxagent.harness.gateway import AgentRuntimeGateway
from tests.support.api import api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

CONVERSATIONAL = ("report_qa", "attribution", "evidence_research")


def _install(app, script):
    provider = ScriptedRuntimeProvider(app.state.tool_registry, app.state.tool_runner, script)
    gateway = AgentRuntimeGateway(
        app.state.database, app.state.tool_registry, app.state.capability_tokens,
        provider, app.state.settings.runtime, create_analysis=app.state.create_analysis,
    )
    for intent in (Intent.REPORT_QA, Intent.ATTRIBUTION, Intent.EVIDENCE_RESEARCH):
        app.state.scheduler.register(intent, gateway.execute)
    app.state.runtime_gateway = gateway
    return gateway


async def test_a_predictor_only_deployment_is_ready_and_says_what_it_cannot_do(db):
    """A deliberately absent agent is a mode, not an outage.

    Reporting 503 here would page an operator about a feature nobody
    installed, and a readiness probe that cries wolf stops being read.
    """
    async with api_client(db, StubPredictor()) as client:
        response = await client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["mode"] == "predictor_only"

        assert body["capabilities"]["analysis"]["available"] is True
        assert body["capabilities"]["analysis_batch"]["available"] is True
        for name in CONVERSATIONAL:
            capability = body["capabilities"][name]
            assert capability["available"] is False, name
            assert capability["configured"] is False, name
            assert capability["reason"], name
        assert body["runtime"]["bound"] is False
        assert "healthy" not in body["runtime"]


async def test_the_reported_kind_is_labelled_as_configured_not_as_bound(db):
    """I01: Compose set a kind the app never builds a provider for.

    Reporting a bare `kind` read as "this runtime is running". Both facts are
    now named, and `mode` is derived from the binding rather than the string.
    """
    async with api_client(db, StubPredictor()) as client:
        body = (await client.get("/health/ready")).json()
        assert body["runtime"]["configured_kind"] == "scripted"
        assert body["runtime"]["bound"] is False
        assert body["mode"] == "predictor_only"


async def test_an_agent_enabled_deployment_probes_the_runtime(db):
    async def script(turn) -> None:
        turn.say("noop")

    async with api_client(db, StubPredictor()) as client:
        _install(client.app, script)
        body = (await client.get("/health/ready")).json()
        assert body["ready"] is True
        assert body["mode"] == "agent_enabled"
        for name in CONVERSATIONAL:
            assert body["capabilities"][name]["available"] is True, name
        assert body["runtime"]["healthy"] is True


async def test_a_registered_handler_with_no_runtime_behind_it_is_not_ready(db):
    """Declared but unservable is incoherent wiring, not a mode.

    This is the one capability condition that fails readiness: something
    registered a handler, so this deployment intended to serve the intent.
    """
    async with api_client(db, StubPredictor()) as client:
        app = client.app
        app.state.scheduler.register(Intent.REPORT_QA, lambda context: None)
        app.state.runtime_gateway = None

        response = await client.get("/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["ready"] is False
        assert body["runtime"]["healthy"] is False
        assert body["runtime"]["misconfigured"] == ["report_qa"]
        assert body["capabilities"]["report_qa"]["configured"] is True
        assert body["capabilities"]["report_qa"]["available"] is False


async def test_the_database_is_probed(db):
    """It was not, so an unreachable database still answered ready: true."""
    async with api_client(db, StubPredictor()) as client:
        body = (await client.get("/health/ready")).json()
        assert body["database"]["ready"] is True
        assert body["database"]["checked_at"]


async def test_an_unreachable_database_fails_readiness(db, monkeypatch):
    async with api_client(db, StubPredictor()) as client:
        async def broken(**_kwargs):
            raise ConnectionRefusedError("database is gone")

        monkeypatch.setattr(client.app.state.database, "check", broken)
        response = await client.get("/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["ready"] is False
        assert body["database"]["ready"] is False
        assert body["database"]["reason"] == "ConnectionRefusedError"


async def test_every_capability_carries_a_checked_at_timestamp(db):
    async with api_client(db, StubPredictor()) as client:
        body = (await client.get("/health/ready")).json()
        for name, capability in body["capabilities"].items():
            assert capability["checked_at"], name
