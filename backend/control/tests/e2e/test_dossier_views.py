"""W9-05: the dossier is the source of the chat answer's case view and of the report.

RETHINK §4.4 step 3 and §4.9: chat, report and board are views of one record.
Before this the dossier was stored and served by the API, and nothing else read
it: the chat message carried only the answer, and a report built on the same
compound knew nothing of the investigation.
"""
from __future__ import annotations

import pytest

from toxagent.application.report_dispatch import DatabaseBuildStore
from toxagent.persistence.object_store import InMemoryObjectStore
from tests.e2e.test_orchestrated_report import _build_report, _fact_view, _install, _synthesis
from tests.e2e.test_scientific_case_e2e import _answer, _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client, settings
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

H1 = "The compound blocks hERG at relevant exposure"


@pytest.fixture
def flags_on(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_CASE_V1", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_REPORT_ORCHESTRATOR_V2", "1")


async def _decision_turn(turn) -> None:
    await turn.call_tool("update_scientific_case", {"operations": [
        {"op": "add_hypothesis", "statement": H1, "hypothesis_kind": "mechanism",
         "refutation_condition": "An IC50 far above free Cmax"},
        {"op": "set_conclusion", "cannot_say": ["whether it blocks hERG at its exposure"],
         "what_would_change": ["an in-house patch-clamp IC50"]},
    ]})
    await turn.call_tool("submit_grounded_answer", _answer(None, "It depends on exposure."))


async def test_the_chat_answer_and_the_report_read_the_runs_dossier(db, flags_on):
    views: list[dict] = []

    async def synthesis(turn) -> None:
        view = _fact_view(turn.system_prompt)
        views.append(view)
        result = await turn.call_tool("submit_report_synthesis", _synthesis(view))
        assert result["status"] == "completed", result

    async with api_client(
        db, StubPredictor(), config=settings(), object_store=InMemoryObjectStore()
    ) as client:
        await _install(client, synthesis)
        await _install_scripted_runtime(client, _decision_turn)
        session_id = await _new_session(client)
        analysis_id = await _analyse(client, session_id)

        run = await _post(client, session_id, "Should hERG stop us developing this compound?")
        assert run["status"] == "completed", run
        messages = (await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)).json()
        reply = messages["messages"][-1]
        refs = [part["content"] for part in reply["parts"] if part["type"] == "dossier_ref"]
        assert len(refs) == 1 and refs[0]["run_id"] == run["run_id"]
        dossier = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/dossier", headers=AUTH
        )).json()
        assert refs[0]["case_id"] == dossier["case_id"]

        report_run = await _build_report(client, session_id, analysis_id)
        assert report_run["status"] == "completed", report_run
        case_view = views[0]["case_dossier"]
        assert case_view["case_id"] == dossier["case_id"]
        assert case_view["run_id"] == run["run_id"]
        assert case_view["hypotheses"][0]["statement"] == H1
        assert case_view["conclusion"]["cannot_say"] == ["whether it blocks hERG at its exposure"]
        build = await DatabaseBuildStore(db, session_id=session_id).load(views[0]["report_build_id"])
        assert build.stage_state["case_dossier_ref"] == {
            "case_id": dossier["case_id"], "run_id": run["run_id"],
            "case_revision": dossier["case_revision"],
        }

        # The runtime-driven report path reads it through get_report_context.
        from datetime import datetime, timedelta, timezone

        from toxagent.application.policy import Actor
        from toxagent.tools.definitions.report import ReportContextInput
        from toxagent.tools.registry import ToolContext

        tool = client.app.state.tool_registry.get("get_report_context")
        output = await tool.handler(
            ToolContext(session_id=session_id, run_id=report_run["run_id"],
                        actor=Actor(subject_id="user-1"), profile="report_build",
                        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=1)),
            ReportContextInput(report_build_id=build.id),
        )
        assert output.model_view["case_dossier"]["run_id"] == run["run_id"]


async def test_without_a_case_the_report_prompt_is_unchanged(db, flags_on):
    views: list[dict] = []

    async def synthesis(turn) -> None:
        view = _fact_view(turn.system_prompt)
        views.append(view)
        await turn.call_tool("submit_report_synthesis", _synthesis(view))

    async with api_client(
        db, StubPredictor(), config=settings(), object_store=InMemoryObjectStore()
    ) as client:
        await _install(client, synthesis)
        session_id = await _new_session(client)
        analysis_id = await _analyse(client, session_id)
        await _build_report(client, session_id, analysis_id)
        assert "case_dossier" not in views[0]


async def test_closing_a_restricted_case_does_not_lift_the_restriction(db, flags_on):
    """W9-07 follow-up: a confidential compound stays confidential in its next case."""
    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, _decision_turn)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        await _post(client, session_id, "Should hERG stop us?")
        first = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()["cases"][0]
        await client.post(
            f"/v1/sessions/{session_id}/cases/{first['case_id']}/scope",
            json={"external_search": False, "reason": "unpublished structure"}, headers=AUTH,
        )
        closed = await client.post(
            f"/v1/sessions/{session_id}/cases/{first['case_id']}:close", headers=AUTH
        )
        assert closed.status_code == 200, closed.text
        await _post(client, session_id, "Start again: is hERG a concern?")
        cases = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()["cases"]
        assert len(cases) == 2
        newest = next(c for c in cases if c["case_id"] != first["case_id"])
        assert newest["external_search"] is False
