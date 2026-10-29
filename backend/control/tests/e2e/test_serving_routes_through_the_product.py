"""§4.1 through the product: the route decides how a turn runs, not only its label.

* **read/explain** runs under the read profile: the artifact reads and the
  answer, no search, no case writes, no skills;
* **update** is decided by what the previous run of the subject did not see --
  a researcher value filed since -- and the contract names that value; the turn
  after it, with nothing new, is not an update;
* **compare/batch** needs the wording *and* more than one candidate; it reads
  each candidate from its own artifact through compare_candidates, and one
  candidate is a question back, not a comparison of one.
"""
from __future__ import annotations

import pytest

from tests.e2e.test_scientific_case_e2e import _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
_NOTED = {"answer_markdown": "Noted.", "claims": [], "limitations": []}


@pytest.fixture
def flags_on(monkeypatch):
    for flag in ("ANSWER_DRAFT_V2", "SCIENTIFIC_CASE_V1", "SCIENTIFIC_PROCEDURES_V1",
                 "INVESTIGATOR_CONTRACT_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{flag}", "1")


async def _contract(client, session_id: str, run_id: str) -> dict:
    state = (await client.get(f"/v1/sessions/{session_id}/runs/{run_id}/decision-state",
                              headers=AUTH)).json()
    return state["goal_contract"]


async def test_a_lookup_runs_under_the_read_profile_and_cannot_search(db, flags_on):
    seen: dict = {}

    async def script(turn) -> None:
        seen["profile"] = turn.tool_context.profile
        seen["search"] = await turn.call_tool("search_toxicology_evidence", {
            "analysis_id": seen["analysis_id"], "query": "hERG", "limit": 3})
        await turn.call_tool("submit_grounded_answer", _NOTED)

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        seen["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id, "What is the hERG label?")
        contract = await _contract(client, session_id, run["run_id"])

    assert run["status"] == "completed", run
    assert contract["route"] == "read_explain"
    assert seen["profile"] == "decision_support_read"
    assert "search_toxicology_evidence" not in contract["admissible_actions"]
    assert seen["search"]["status"] == "error"


async def test_a_value_filed_since_the_last_turn_makes_an_update_and_only_once(db, flags_on):
    async def script(turn) -> None:
        await turn.call_tool("submit_grounded_answer", _NOTED)

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        first = await _post(client, session_id, "Is hERG a concern here?")
        case_id = (await client.get(f"/v1/sessions/{session_id}/cases",
                                    headers=AUTH)).json()["cases"][0]["case_id"]
        added = await client.post(f"/v1/sessions/{session_id}/cases/{case_id}/context",
                                  json={"key": "free_cmax", "value": "0.01", "unit": "uM"},
                                  headers=AUTH)
        assert added.status_code == 200, added.text
        second = await _post(client, session_id, "Is hERG a concern here?")
        third = await _post(client, session_id, "Is hERG a concern here?")
        contracts = [await _contract(client, session_id, r["run_id"])
                     for r in (first, second, third)]

    assert [c["route"] for c in contracts] == ["investigate", "update", "investigate"]
    update = contracts[1]
    assert update["route_decision"]["reason_codes"] == ["researcher_supplied_input"]
    assert any("free_cmax = 0.01 uM (case_value:free_cmax@v1)" in q["question"]
               for q in update["must_answer"])
    assert contracts[1]["researcher_values_seen"] == ["case_value:free_cmax@v1"]
    assert contracts[2]["researcher_values_seen"] == ["case_value:free_cmax@v1"]


async def test_a_comparison_reads_each_candidate_from_its_own_artifact(db, flags_on):
    seen: dict = {}

    async def script(turn) -> None:
        seen["compare"] = await turn.call_tool("run_scientific_procedure", {
            "procedure": "compare_candidates", "criterion": "herg_probability_blocker"})
        seen["no_criterion"] = await turn.call_tool("run_scientific_procedure", {
            "procedure": "compare_candidates"})
        await turn.call_tool("submit_grounded_answer", _NOTED)

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        first = await _analyse(client, session_id)
        second = await _analyse(client, session_id, smiles=CAFFEINE)
        run = await _post(client, session_id, "Compare the candidates by hERG probability")
        contract = await _contract(client, session_id, run["run_id"])

    assert contract["route"] == "compare_batch"
    assert {c["subject_key"] for c in contract["candidates"]} == {
        f"analysis:{first}", f"analysis:{second}"}
    facts = seen["compare"]["model_view"]["facts"]
    (group,) = facts["groups"]
    assert {r["subject_key"] for r in group["rows"]} == {f"analysis:{first}",
                                                        f"analysis:{second}"}
    assert all(r["observation_id"].startswith("obs_") for r in group["rows"])
    assert all(r["field_path"] == "predictions.herg.probability_blocker" for r in group["rows"])
    assert seen["no_criterion"]["status"] == "error"
    assert "ask the researcher" in seen["no_criterion"]["error"]["message"]


async def test_asking_to_compare_one_candidate_asks_which_others(db, flags_on):
    seen: dict = {}

    async def script(turn) -> None:
        seen["profile"] = turn.tool_context.profile
        await turn.call_tool("submit_grounded_answer", _NOTED)

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        run = await _post(client, session_id, "Compare it with the alternatives")
        contract = await _contract(client, session_id, run["run_id"])

    assert contract["route"] == "undetermined"
    assert seen["profile"] == "decision_support_read"
    assert contract["must_answer"][0]["gap"] == "only_researcher_has_it"
    assert "Which others" in contract["must_answer"][0]["question"]
