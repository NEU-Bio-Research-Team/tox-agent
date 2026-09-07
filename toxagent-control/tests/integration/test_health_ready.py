"""`/health/ready` reports real capabilities, not just a configured kind
(audit_5_9.md A16)."""
from __future__ import annotations

import pytest

from toxagent.domain.run import Intent
from toxagent.harness.adapters.scripted import ScriptedRuntimeProvider
from toxagent.harness.gateway import AgentRuntimeGateway
from tests.support.api import api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio


async def test_a_deployment_with_no_conversational_handler_reports_it_honestly(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["capabilities"] == {
            "analysis": True, "report_qa": False, "attribution": False,
            "evidence_research": False, "structure_recognition": False,
        }
        assert "healthy" not in body["runtime"]


async def test_a_registered_conversational_handler_reports_runtime_health(db):
    async def script(turn) -> None:
        turn.say("noop")

    async with api_client(db, StubPredictor()) as client:
        app = client.app
        provider = ScriptedRuntimeProvider(app.state.tool_registry, app.state.tool_runner, script)
        gateway = AgentRuntimeGateway(
            app.state.database, app.state.tool_registry, app.state.capability_tokens,
            provider, app.state.settings.runtime, create_analysis=app.state.create_analysis,
        )
        for intent in (Intent.REPORT_QA, Intent.ATTRIBUTION, Intent.EVIDENCE_RESEARCH):
            app.state.scheduler.register(intent, gateway.execute)
        app.state.runtime_gateway = gateway

        response = await client.get("/health/ready")
        body = response.json()
        assert body["ready"] is True
        assert body["capabilities"]["report_qa"] is True
        assert body["runtime"]["healthy"] is True


async def test_a_registered_handler_with_no_gateway_is_not_ready(db):
    """Should not happen given how create_app composes the object graph, but
    readiness must not report `ready=true` if it ever did."""
    async with api_client(db, StubPredictor()) as client:
        app = client.app
        app.state.scheduler.register(Intent.REPORT_QA, lambda context: None)
        app.state.runtime_gateway = None

        response = await client.get("/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["ready"] is False
        assert body["runtime"]["healthy"] is False
