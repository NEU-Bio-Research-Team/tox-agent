"""TAB-Suite Wave 2 end to end: a decision_support run keeps a
DecisionSupportStateV1 — plan, usage, resolution, stop reason — records its
EffectiveRunBudgetV1, refuses synthesis without lineage, and scopes the next
turn's memory to the same subject.

Driven through the product API with the scripted runtime, the same seam an
OpenCode adapter uses.
"""
from __future__ import annotations

import pytest

from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client, wait_for_run
from tests.support.predictor import ASPIRIN, StubPredictor
from tests.support.research import ACCEPTED_HIT, StubResearchProvider

pytestmark = pytest.mark.anyio

BENZENE = "c1ccccc1"
HERG_Q = "Does the compound block hERG?"


@pytest.fixture
def flags_on(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_DECISION_STATE_PLAN_TOOL", "1")


def _draft(evidence_id: str, *, relations: list[dict]) -> dict:
    return {
        "schema_version": "grounded-answer-v2",
        "answer_markdown": "One retrieved study reports hERG blockade in a related series.",
        "claims": [{
            "local_ref": "herg_lit", "kind": "scientific",
            "text": "A retrieved study reports hERG blockade in a related series.",
            "citation_ids": [evidence_id],
        }],
        "limitations": [{"code": "evidence_scope_limited", "text": ""}],
        "evidence_relations": relations,
    }


async def _post(client, session_id: str, text: str) -> dict:
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"intent_hint": "auto", "content": [{"type": "text", "text": text}]},
        headers=AUTH,
    )
    assert submitted.status_code == 202, submitted.text
    return await wait_for_run(client, session_id, submitted.json()["run_id"])


async def test_a_decision_support_run_records_plan_resolution_and_stop(db, flags_on):
    state: dict = {"analysis_id": "", "prompts": [], "results": []}

    async def script(turn) -> None:
        state["prompts"].append(turn.system_prompt)
        if "hERG" not in turn.user_message:
            turn.say("noop")
            return
        plan = await turn.call_tool("record_decision_plan", {"propositions": [
            {"question": HERG_Q, "required_sources": ["evidence"]},
        ]})
        state["results"].append(plan)
        found = await turn.call_tool("search_toxicology_evidence", {
            "analysis_id": state["analysis_id"], "query": "hERG blockade", "limit": 5,
        })
        evidence_id = found["model_view"]["results"][0]["evidence_id"]
        await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        submitted = await turn.call_tool("submit_grounded_answer", _draft(evidence_id, relations=[
            {"proposition": HERG_Q, "source_class": "external_experimental",
             "source_id": evidence_id, "relation": "supports", "reason_codes": ["same_endpoint"]},
            {"proposition": HERG_Q, "source_class": "agent_synthesis", "source_id": evidence_id,
             "relation": "supports", "reason_codes": ["synthesis"],
             "input_source_refs": [f"evidence:{evidence_id}"]},
        ]))
        state["results"].append(submitted)

    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)

        run = await _post(client, session_id, "Should we worry about hERG for development?")
        assert run["status"] == "completed", run
        assert run["intent"] == "decision_support"
        budget = run["configuration_snapshot"]["effective_budget"]
        assert budget["schema_version"] == "effective-run-budget-v1"
        assert budget["max_searches"] is not None

        response = await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/decision-state", headers=AUTH
        )
        assert response.status_code == 200, response.text
        decision = response.json()
        assert state["results"][-1]["status"] == "completed", state["results"][-1]
        assert decision["propositions"][0]["id"] == "p1"
        assert decision["propositions"][0]["status"] == "supported"
        assert decision["coverage"] == {"required": 1, "resolved": 1}
        assert decision["stop_reason"] == "sufficient"
        assert decision["answer_outcome"] == "first_pass"
        assert decision["usage"]["searches"] == 1 and decision["usage"]["evidence_reads"] == 1
        assert decision["subject_refs"] == [f"analysis:{state['analysis_id']}"]

        # Same subject: the next turn is told what was established.
        await _post(client, session_id, "Summarise what we know so far.")
        assert "Previous decision-support goal" in state["prompts"][-1]
        assert ACCEPTED_HIT.title in state["prompts"][-1]

        # A different compound: no checkpoint, no evidence from the old subject.
        await _analyse(client, session_id, BENZENE)
        await _post(client, session_id, "Summarise what we know so far.")
        assert "Previous decision-support goal" not in state["prompts"][-1]
        assert ACCEPTED_HIT.title not in state["prompts"][-1]

        # A non-decision_support run keeps no state.
        analysis_runs = await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
        assert analysis_runs.status_code == 200


async def test_synthesis_without_resolvable_lineage_is_refused(db, flags_on):
    outcomes: list[dict] = []
    analysis: dict = {"id": ""}

    async def script(turn) -> None:
        found = await turn.call_tool("search_toxicology_evidence", {
            "analysis_id": analysis["id"], "query": "hERG blockade", "limit": 5,
        })
        evidence_id = found["model_view"]["results"][0]["evidence_id"]
        await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        # Well-formed, but the input names an observation this session never had.
        outcomes.append(await turn.call_tool("submit_grounded_answer", _draft(evidence_id, relations=[
            {"proposition": HERG_Q, "source_class": "agent_synthesis", "source_id": evidence_id,
             "relation": "supports", "reason_codes": ["synthesis"],
             "input_source_refs": ["observation:obs_" + "0" * 32]},
        ])))
        # No lineage at all: refused at the wire.
        outcomes.append(await turn.call_tool("submit_grounded_answer", _draft(evidence_id, relations=[
            {"proposition": HERG_Q, "source_class": "agent_synthesis", "source_id": evidence_id,
             "relation": "supports", "reason_codes": ["synthesis"]},
        ])))

    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        analysis["id"] = await _analyse(client, session_id, ASPIRIN)
        run = await _post(client, session_id, "Is hERG a development concern?")
        decision = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/decision-state", headers=AUTH
        )).json()

    first, second = outcomes
    assert first["status"] == "error"
    assert "agent_synthesis_lineage_unresolved" in str(first)
    assert second["status"] == "error"
    assert "input_source_refs" in str(second)
    # The run did not get a model answer through; the state says so.
    assert decision["stop_reason"] in ("blocked", "insufficient_evidence")
