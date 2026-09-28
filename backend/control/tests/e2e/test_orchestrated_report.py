"""An orchestrated report build, end to end, through the real app (PR-12).

The gate for PR-12 is that ``report_orchestrator_v2`` produces a report someone
can open: the server runs every assembly stage itself, the runtime is invoked
exactly once with one tool, and what comes back is a v3 artifact with
renderings. The scripted runtime stands in for the model and calls the same
registry and runner MCP would.

What these tests pin, beyond "it completes":

* the runtime sees only ``submit_report_synthesis`` and a prompt that carries
  the fact bundle, not the report-builder skills;
* every stage event corresponds to work — a stage that ran emits ``running``
  then its outcome, a stage switched off emits only ``skipped``;
* a synthesis that types a number is refused twice and the build fails with no
  artifact — there is no fallback report;
* the accepted synthesis survives the orchestrator's own saves.
"""
from __future__ import annotations

import json
import re

import pytest

from toxagent.application.report.dispatch import DatabaseBuildStore, OrchestratedReportBuild
from toxagent.application.report.synthesis import SYNTHESIS_KEY
from toxagent.domain.run import Intent
from toxagent.harness.adapters.scripted import ScriptedRuntimeProvider
from toxagent.harness.gateway import AgentRuntimeGateway
from toxagent.persistence.object_store import InMemoryObjectStore
from tests.support.api import AUTH, api_client, settings, wait_for_run
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

NARRATIVE = (
    "executive_summary",
    "substance_profile",
    "explanation_and_visuals",
    "external_evidence",
    "integrated_interpretation",
    "conclusions",
    "recommendations",
)

_BUNDLE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


def _fact_view(system_prompt: str) -> dict:
    match = _BUNDLE.search(system_prompt)
    assert match, "the synthesis prompt must carry the fact bundle"
    return json.loads(match.group(1))


def _synthesis(view: dict, *, summary: str | None = None) -> dict:
    probability = next(
        fact["fact_id"] for fact in view["facts"] if fact["label"] == "hERG blocker probability"
    )
    return {
        "report_build_id": view["report_build_id"],
        "title": "hERG screening report",
        "sections": [
            {
                "section_id": section_id,
                "heading": section_id.replace("_", " ").title(),
                "prose_markdown": (
                    summary
                    if summary is not None and section_id == "executive_summary"
                    else (
                        f"The hERG blocker probability is {{{{{probability}}}}}."
                        if section_id == "executive_summary"
                        else "No further narrative for this section."
                    )
                ),
                "basis_fact_ids": [probability] if section_id == "executive_summary" else [],
            }
            for section_id in NARRATIVE
        ],
        "conclusions": [
            {
                "local_ref": "c1",
                "text": "This model predicts low hERG blocker liability.",
                "basis_fact_ids": [probability],
                "endpoint": "herg",
            }
        ],
        "recommendations": [
            {
                "local_ref": "r1",
                "text": "Confirm with a patch-clamp hERG assay before prioritising.",
                "basis_fact_ids": [probability],
                "action_category": "in_vitro_assay",
                "priority": "medium",
                "rationale": "A model probability is a screening signal, not a measurement.",
            }
        ],
    }


async def _install(client, script) -> OrchestratedReportBuild:
    app = client.app
    provider = ScriptedRuntimeProvider(app.state.tool_registry, app.state.tool_runner, script)
    gateway = AgentRuntimeGateway(
        app.state.database,
        app.state.tool_registry,
        app.state.capability_tokens,
        provider,
        app.state.settings.runtime,
        create_analysis=app.state.create_analysis,
        profiles_dir=app.state.settings.profiles_dir,
    )
    orchestrated = OrchestratedReportBuild(
        app.state.database,
        gateway=gateway,
        runner=app.state.tool_runner,
        registry=app.state.tool_registry,
        predictor=app.state.predictor,
        object_store=app.state.object_store,
    )
    app.state.scheduler.register(Intent.BUILD_REPORT, orchestrated.execute)
    app.state.runtime_gateway = gateway
    return orchestrated


async def _analysis(client) -> tuple[str, str]:
    session = await client.post("/v1/sessions", json={}, headers=AUTH)
    session_id = session.json()["session_id"]
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"molecule": {"smiles": "CCO"}},
        headers=AUTH,
    )
    assert submitted.status_code == 202, submitted.text
    await wait_for_run(client, session_id, submitted.json()["run_id"])
    state = await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
    return session_id, state.json()["active_analysis"]["analysis_id"]


async def _build_report(client, session_id: str, analysis_id: str) -> dict:
    accepted = await client.post(
        f"/v1/sessions/{session_id}/reports",
        json={
            "analysis_id": analysis_id,
            "selected_endpoints": ["herg"],
            "include_explanations": False,
            "include_external_evidence": False,
            "output_formats": ["markdown", "html"],
        },
        headers=AUTH,
    )
    assert accepted.status_code == 202, accepted.text
    return await wait_for_run(client, session_id, accepted.json()["run_id"])


async def _events(client, session_id: str) -> list[dict]:
    response = await client.get(f"/v1/sessions/{session_id}/events:list?limit=500", headers=AUTH)
    response.raise_for_status()
    return response.json()["events"]


async def test_an_orchestrated_build_publishes_a_v3_report_from_one_runtime_turn(
    db, monkeypatch
):
    monkeypatch.setenv("TOXAGENT_FLAG_REPORT_ORCHESTRATOR_V2", "1")
    turns: list[dict] = []

    async def script(turn):
        view = _fact_view(turn.system_prompt)
        turns.append(
            {
                "profile": turn.tool_context.profile,
                "prompt": turn.system_prompt,
                "view": view,
            }
        )
        result = await turn.call_tool("submit_report_synthesis", _synthesis(view))
        assert result["status"] == "completed", result

    async with api_client(
        db, StubPredictor(), config=settings(), object_store=InMemoryObjectStore()
    ) as client:
        await _install(client, script)
        session_id, analysis_id = await _analysis(client)
        run = await _build_report(client, session_id, analysis_id)
        assert run["status"] == "completed", run

        assert len(turns) == 1, "the runtime is invoked once, at synthesis"
        assert turns[0]["profile"] == "report_synthesis"
        # The report-builder skills describe work this turn cannot do.
        assert "save_report_draft" not in turns[0]["prompt"]
        denied = await client.app.state.tool_runner.call(
            _context_like(client, session_id, run["run_id"]), "get_report_context", {}
        )
        assert denied["status"] == "error"
        assert denied["error"]["code"] == "tool_denied"

        messages = (
            await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        ).json()["messages"]
        ref = next(
            part["content"]
            for message in messages
            for part in message["parts"]
            if part["type"] == "report_ref"
        )
        report = (
            await client.get(
                f"/v1/sessions/{session_id}/reports/{ref['report_id']}", headers=AUTH
            )
        ).json()
        assert report["schema_version"] == "toxagent-report-v3"
        assert report["status"] == "completed_with_gaps"
        assert {r["format"] for r in report["renderings"]} == {"markdown", "html"}
        summary = next(s for s in report["sections"] if s["section_id"] == "executive_summary")
        rendered = next(
            fact["rendered"]
            for fact in turns[0]["view"]["facts"]
            if fact["label"] == "hERG blocker probability"
        )
        assert f"The hERG blocker probability is {rendered}." in summary["body_markdown"]
        assert "{{fct_" not in json.dumps(report)
        reasons = {gap["reason"] for gap in report["gaps"]}
        assert "external_evidence_not_requested" in reasons

        markdown = await client.get(
            f"/v1/sessions/{session_id}/reports/{ref['report_id']}/renderings/markdown",
            headers=AUTH,
        )
        assert markdown.status_code == 200
        assert rendered in markdown.text

        stage_events = [
            event["payload"]
            for event in await _events(client, session_id)
            if event["type"] == "report.stage_changed"
        ]
        by_stage: dict[str, list[str]] = {}
        for payload in stage_events:
            by_stage.setdefault(payload["stage"], []).append(payload["status"])
        assert by_stage["preparing_analysis"] == ["running", "completed"]
        assert by_stage["synthesizing"] == ["running", "completed"]
        assert by_stage["rendering"] == ["running", "completed"]
        # Switched off: reached and recorded, never "started".
        assert by_stage["generating_explanations"] == ["skipped"]
        assert by_stage["researching_evidence"] == ["skipped"]
        completed_counts = [payload["completed"] for payload in stage_events]
        assert completed_counts == sorted(completed_counts), "progress never goes backwards"


async def test_a_synthesis_that_types_a_number_is_refused_and_no_report_is_published(
    db, monkeypatch
):
    monkeypatch.setenv("TOXAGENT_FLAG_REPORT_ORCHESTRATOR_V2", "1")
    refusals: list[dict] = []

    async def script(turn):
        view = _fact_view(turn.system_prompt)
        for _ in range(3):
            result = await turn.call_tool(
                "submit_report_synthesis",
                _synthesis(view, summary="The hERG blocker probability is 0.99."),
            )
            refusals.append(result)

    async with api_client(
        db, StubPredictor(), config=settings(), object_store=InMemoryObjectStore()
    ) as client:
        await _install(client, script)
        session_id, analysis_id = await _analysis(client)
        run = await _build_report(client, session_id, analysis_id)

        assert run["status"] == "failed"
        assert run["failure_code"] == "report_validation_failed"
        assert [r["status"] for r in refusals] == ["error", "error", "error"]
        # Two judged refusals, then a conflict: the budget is the build's.
        assert refusals[0]["error"]["details"]["attempts_remaining"] == 1
        assert refusals[1]["error"]["details"]["attempts_remaining"] == 0
        listed = await client.get(f"/v1/sessions/{session_id}/reports", headers=AUTH)
        assert listed.json()["count"] == 0
        types = [event["type"] for event in await _events(client, session_id)]
        assert "report.failed" in types
        assert "report.completed" not in types and "report.completed_with_gaps" not in types


async def test_the_orchestrator_store_never_erases_an_accepted_synthesis(db):
    from datetime import datetime, timezone

    from toxagent.domain.message import Message, Role
    from toxagent.domain.report import ReportBuild, ReportBuildRequest
    from toxagent.domain.run import Lane, Run
    from toxagent.domain.session import Session
    from toxagent.domain.ids import ANALYSIS, new_id

    now = datetime.now(timezone.utc)
    session = Session.create("user-1", now=now)
    message = Message.create(session.id, Role.USER, 1, now=now)
    run = Run.create(session.id, message.id, Lane.MIXED, Intent.BUILD_REPORT, now=now)
    build = ReportBuild.start(
        session_id=session.id, run_id=run.id, now=now,
        request=ReportBuildRequest(
            session_id=session.id, analysis_id=new_id(ANALYSIS), selected_endpoints=("herg",)
        ),
    )
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.reports.add_build(build)
        await uow.commit()

    store = DatabaseBuildStore(db, session_id=session.id)
    stale = await store.load(build.id)
    # The synthesis turn writes while the orchestrator still holds `stale`.
    async with db.unit_of_work() as uow:
        await uow.reports.save_build(
            __import__("dataclasses").replace(
                stale, stage_state={**stale.stage_state, SYNTHESIS_KEY: {"title": "kept"}}
            )
        )
        await uow.commit()
    await store.save(
        __import__("dataclasses").replace(stale, stage_state={"stage_checkpoints": {"x": {}}})
    )
    stored = await store.load(build.id)
    assert stored.stage_state[SYNTHESIS_KEY] == {"title": "kept"}
    assert stored.stage_state["stage_checkpoints"] == {"x": {}}


def _context_like(client, session_id: str, run_id: str):
    """A synthesis-profile context, to prove the profile's surface is one tool."""
    from datetime import datetime, timedelta, timezone

    from toxagent.application.policy import Actor
    from toxagent.tools.registry import ToolContext

    return ToolContext(
        session_id=session_id,
        run_id=run_id,
        actor=Actor(subject_id="user-1"),
        profile="report_synthesis",
        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
