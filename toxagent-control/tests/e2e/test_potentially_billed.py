"""``potentially_billed`` boundary semantics (plan section 6.6, remaining-plan
W2-12/15).

The flag exists to distinguish "we know nothing was spent" from "the runtime
confirmed receiving this turn and we do not know what happened after that" —
never inferred by guessing at an adapter's own billing internals, since we
cannot see them. ``AgentRuntimeGateway.execute`` treats a provider receipt's
``accepted=True`` as the line: before it, a failure costs nothing this
product could have caused a provider request for; after it, an undetermined
outcome must say so rather than default to "no".
"""
from __future__ import annotations

import pytest

from toxagent.domain.run import Intent
from toxagent.harness.adapters.scripted import ScriptedRuntimeProvider
from toxagent.harness.gateway import AgentRuntimeGateway
from toxagent.harness.provider import RuntimeHealth
from tests.support.api import AUTH, api_client, wait_for_run
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio


async def _install_scripted_runtime(client, script) -> None:
    app = client.app
    provider = ScriptedRuntimeProvider(app.state.tool_registry, app.state.tool_runner, script)
    gateway = AgentRuntimeGateway(
        app.state.database,
        app.state.tool_registry,
        app.state.capability_tokens,
        provider,
        app.state.settings.runtime,
        create_analysis=app.state.create_analysis,
    )

    async def run_agentic(context) -> None:
        await gateway.execute(context)

    for intent in (Intent.REPORT_QA, Intent.ATTRIBUTION, Intent.EVIDENCE_RESEARCH):
        app.state.scheduler.register(intent, run_agentic)


async def _install_unhealthy_runtime(client) -> None:
    """A provider that never accepts a turn — health fails first, so
    execute() never reaches create_session/send at all. Only health() is
    implemented; anything else being called is the bug this guards against,
    and would raise AttributeError rather than silently pass.
    """

    class _UnhealthyProvider:
        kind = "scripted"

        async def health(self) -> RuntimeHealth:
            return RuntimeHealth(healthy=False, detail="deliberately unhealthy for this test")

    app = client.app
    gateway = AgentRuntimeGateway(
        app.state.database,
        app.state.tool_registry,
        app.state.capability_tokens,
        _UnhealthyProvider(),
        app.state.settings.runtime,
        create_analysis=app.state.create_analysis,
    )

    async def run_agentic(context) -> None:
        await gateway.execute(context)

    for intent in (Intent.REPORT_QA, Intent.ATTRIBUTION, Intent.EVIDENCE_RESEARCH):
        app.state.scheduler.register(intent, run_agentic)


async def _new_session(client) -> str:
    response = await client.post("/v1/sessions", json={}, headers=AUTH)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


async def _analyse_aspirin(client, session_id: str) -> None:
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"molecule": {"smiles": ASPIRIN}},
        headers=AUTH,
    )
    assert submitted.status_code == 202, submitted.text
    await wait_for_run(client, session_id, submitted.json()["run_id"])


async def test_a_turn_the_runtime_never_accepted_is_not_potentially_billed(db):
    """Fails at the pre-flight health probe, before create_session/send are
    ever reached — no provider request this product could have caused."""
    async with api_client(db, StubPredictor()) as client:
        await _install_unhealthy_runtime(client)
        session_id = await _new_session(client)
        await _analyse_aspirin(client, session_id)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "ask_report",
                "content": [{"type": "text", "text": "Explain the report."}],
            },
            headers=AUTH,
        )
        run = await wait_for_run(client, session_id, submitted.json()["run_id"])
        assert run["status"] == "failed"
        assert run["failure_code"] == "runtime_unavailable"
        assert run["potentially_billed"] is False


async def test_a_runtime_accepted_turn_that_never_committed_an_answer_is_potentially_billed(db):
    """The runtime confirms accepting the turn (ScriptedRuntimeProvider.send
    always returns accepted=True), then ends idle without ever calling
    submit_grounded_answer — a real accepted-then-undetermined outcome, not a
    pre-flight refusal."""

    async def script(turn) -> None:
        turn.say("Thinking about the report, but never actually submitting an answer.")

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse_aspirin(client, session_id)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "ask_report",
                "content": [{"type": "text", "text": "Explain the report."}],
            },
            headers=AUTH,
        )
        run = await wait_for_run(client, session_id, submitted.json()["run_id"])
        assert run["status"] == "failed"
        assert run["failure_code"] == "runtime_protocol_error"
        assert run["potentially_billed"] is True


async def test_a_completed_run_with_a_committed_answer_is_not_potentially_billed(db):
    """The ordinary, successful case: an accepted turn that did reach a
    committed answer is not "undetermined" — nothing about its outcome is
    unknown, so the flag stays false, same as before this feature existed."""
    analysis_id = ""

    async def script(turn) -> None:
        slice_result = await turn.call_tool(
            "get_analysis_slice",
            {"analysis_id": analysis_id, "section": "herg", "fields": ["probability_blocker"]},
        )
        value = slice_result["model_view"]["values"]["probability_blocker"]
        await turn.call_tool(
            "submit_grounded_answer",
            {
                "schema_version": "grounded-answer-v1",
                "answer_markdown": "The predicted hERG blocker probability is 0.731.",
                "claims": [
                    {
                        "claim_id": "clm_" + "9" * 32,
                        "kind": "numeric",
                        "text": "The predicted hERG blocker probability is 0.731.",
                        "observation_id": value["observation_id"],
                        "field_path": value["field_path"],
                        "source_value": value["value"],
                        "rendered_value": "0.731",
                        "transform": "round:3",
                    }
                ],
                "limitations": [{"code": "uncalibrated_probability", "text": ""}],
                "recommended_next_steps": [],
            },
        )

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        submitted_analysis = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"molecule": {"smiles": ASPIRIN}},
            headers=AUTH,
        )
        run1 = await wait_for_run(client, session_id, submitted_analysis.json()["run_id"])
        analysis_id = (
            await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
        ).json()["active_analysis"]["analysis_id"]
        assert run1["status"] == "completed"

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "ask_report",
                "content": [{"type": "text", "text": "Explain the report."}],
            },
            headers=AUTH,
        )
        run = await wait_for_run(client, session_id, submitted.json()["run_id"])
        assert run["status"] == "completed"
        assert run["potentially_billed"] is False
