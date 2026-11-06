"""The gap-driven loop through the product (#70/#71): a real run, its ledger, its stop."""
from __future__ import annotations

import pytest

from tests.e2e.test_scientific_case_e2e import _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio


@pytest.fixture
def flags_on(monkeypatch):
    for flag in ("ANSWER_DRAFT_V2", "SCIENTIFIC_CASE_V1", "INVESTIGATOR_CONTRACT_V1",
                 "DECISION_LOOP_V1", "TYPED_STOP_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{flag}", "1")


async def test_a_run_chooses_by_gap_refuses_a_repeat_and_records_why(db, flags_on):
    state: dict = {}

    async def script(turn) -> None:
        state["prompt"] = turn.system_prompt
        first = await turn.call_tool("plan_next_action", {"candidates": [
            {"action": "get_analysis_bundle", "target": "prediction", "requirement_id": "q1",
             "discriminates": ["prediction_mismatch_in_scope"], "source": "observation",
             "reuses": "analysis", "reason": "The prediction is already held; read it first."},
            {"action": "make_coffee", "target": "x", "requirement_id": "q1",
             "discriminates": ["no_relevant_evidence"], "reason": "Not a real action."},
        ]})
        state["first"] = first
        await turn.call_tool("get_artifact_inventory", {})
        again = await turn.call_tool("plan_next_action", {"candidates": [
            {"action": "answer", "requirement_id": "q1", "reason": "The prediction answers it."},
        ]})
        state["again"] = again
        await turn.call_tool("submit_grounded_answer", {
            "answer_markdown": "Not established.", "claims": [], "limitations": []})

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        run = await _post(client, session_id, "Does this compound block hERG?")
        decision = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/decision-state", headers=AUTH,
        )).json()

    assert "call plan_next_action" in state["prompt"]
    first = state["first"]["model_view"]
    assert first["chosen"]["action"] == "get_analysis_bundle"
    assert first["refused"][0]["action"] == "make_coffee"
    assert "not an action" in first["refused"][0]["why"]
    assert state["again"]["model_view"]["chosen"]["action"] == "answer"
    ledger = decision["decision_loop"]
    assert [a["action"] for a in ledger["actions"]] == ["get_analysis_bundle", "answer"]
    assert ledger["actions"][0]["reason"] == "The prediction is already held; read it first."
    assert ledger["actions"][0]["discriminates"] == ["prediction_mismatch_in_scope"]
    assert ledger["gaps"] and all(g["gap"] for g in ledger["gaps"] if g["status"] == "open")


async def test_a_procedure_run_is_counted_and_the_contract_names_it(db, flags_on, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_PROCEDURES_V1", "1")
    state: dict = {}

    async def script(turn) -> None:
        state["prompt"] = turn.system_prompt
        state["plan"] = await turn.call_tool("plan_next_action", {
            "gap_updates": [{"requirement_id": "q1", "status": "open", "gap": "identity_or_scope"}],
            "candidates": [
            {"action": "run_scientific_procedure", "target": "resolve_subject_context",
             "requirement_id": "q1", "discriminates": ["identity_or_scope"],
             "reason": "Settle what the question is about first."}]})
        state["result"] = await turn.call_tool("run_scientific_procedure", {
            "procedure": "resolve_subject_context", "endpoint": "herg"})
        await turn.call_tool("submit_grounded_answer", {
            "answer_markdown": "Not established.", "claims": [], "limitations": []})

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        run = await _post(client, session_id, "Does this compound block hERG?")
        decision = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/decision-state", headers=AUTH,
        )).json()

    result = state["result"]["model_view"]
    assert result["procedure"] == "resolve_subject_context"
    assert result["facts"]["established"]  # the bound analysis is the subject
    assert decision["usage"]["procedure_sub_operations"] == result["sub_operation_count"] > 0
    (gap,) = state["plan"]["model_view"]["open_gaps"]
    assert gap["procedure"] == "run_scientific_procedure resolve_subject_context"
