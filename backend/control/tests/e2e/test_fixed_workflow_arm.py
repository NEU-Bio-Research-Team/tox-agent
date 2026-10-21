"""#50 arm B through the product: the fixed workflow on the agent's tools.

B is the comparison D has to beat, so it is checked for being what it
claims: the server reads by rule, through the same tool runner and budget;
the model's only move is one synthesis turn with one tool; and the same
validator and release gate decide what is released.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from tests.e2e.test_activity_replay_through_the_product import _attach, _config
from tests.e2e.test_scientific_case_e2e import _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.activity import frozen_screening
from tests.support.api import AUTH, api_client
from tests.support.predictor import ASPIRIN, StubPredictor
from tests.support.research import ACCEPTED_HIT, POTENCY_JATS, FullTextStubProvider

pytestmark = pytest.mark.anyio

_MATERIAL = re.compile(r"Material:\n```json\n(.*?)\n```", re.DOTALL)


@pytest.fixture
def arm_b(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_ACTIVITY_REPLAY_V1", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_EVIDENCE_FULL_TEXT_V1", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_FIXED_WORKFLOW_V1", "1")


def _synthesis(state: dict, *, answer):
    async def script(turn) -> None:
        state.setdefault("profiles", []).append(turn.tool_context.profile)
        if turn.tool_context.profile == "claim_review":
            bundle = json.loads(re.search(r"```json\n(.*?)\n```", turn.system_prompt,
                                          re.DOTALL).group(1))
            await turn.call_tool("submit_claim_review", {"reviews": [
                {"claim_id": c["claim_id"], "verdict": state.get("verdict", "supported"),
                 "reason": "judged"} for c in bundle["claims"]
            ]})
            state.setdefault("review_rounds", 0)
            state["review_rounds"] += 1
            state["verdict"] = "supported"
            return
        material = json.loads(_MATERIAL.search(turn.system_prompt).group(1))
        state.setdefault("materials", []).append(material)
        state["submitted"] = await turn.call_tool("submit_grounded_answer", answer(material))

    return script


def _screening_answer(material: dict) -> dict:
    (screening,) = material["screenings"].values()
    estimate = screening["estimate"]
    return {
        "schema_version": "grounded-answer-v2",
        "answer_markdown": "The estimate is {{pic50}}.",
        "claims": [{"local_ref": "pic50", "kind": "numeric", "text": "Estimated pIC50.",
                    "observation_id": estimate["observation_id"],
                    "field_path": "estimate.predicted_pic50", "transform": "round:1"}],
        "limitations": [{"code": c, "text": ""} for c in estimate["required_limitations"]],
    }


async def _decision_state(client, session_id, run_id) -> dict:
    return (await client.get(f"/v1/sessions/{session_id}/runs/{run_id}/decision-state",
                             headers=AUTH)).json()


async def test_a_lookup_is_answered_from_the_artifacts_without_a_search(db, arm_b, tmp_path):
    record = frozen_screening()
    state: dict = {}
    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, _synthesis(state, answer=_screening_answer))
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        run = await _post(client, session_id, "What is the estimated pIC50 here?")
        decision = await _decision_state(client, session_id, run["run_id"])

    assert run["status"] == "completed", {k: v for k, v in run.items() if "fail" in k or "error" in k}
    # The model got one turn, under the one-tool synthesis profile.
    assert state["profiles"] == ["workflow_synthesis"]
    assert state["submitted"]["status"] == "completed", state["submitted"]
    steps = decision["workflow"]["steps"]
    tools = [s["tool"] for s in steps if s["tool"]]
    assert tools == ["get_artifact_inventory", "get_screening_state", "get_activity_snapshot",
                     "get_prioritisation_decision"]
    (route,) = [s for s in steps if s.get("decision") in ("search", "no_search")]
    assert route["decision"] == "no_search" and "lookup" in route["reason"]
    assert all(s.get("reason") for s in steps)
    # the reads went through the runner like any tool call, and are counted
    assert decision["usage"]["tool_calls"] >= 4


#: A record about the analysed compound itself: B searches with the endpoint,
#: so relevance is assessed, and a paper about another series is rejected --
#: correctly -- before B could cite it.
ABOUT_THE_COMPOUND = replace(
    ACCEPTED_HIT,
    title=f"hERG channel blockade by {ASPIRIN} in HEK293 cells",
    abstract_or_excerpt=f"hERG channel blockade by {ASPIRIN} was measured by patch clamp.",
)


async def test_a_decision_question_searches_by_rule_and_reads_the_tables(db, arm_b):
    state: dict = {}
    provider = FullTextStubProvider(hits=[ABOUT_THE_COMPOUND],
                                    documents={ACCEPTED_HIT.provider_record_id: POTENCY_JATS})

    def answer(material: dict) -> dict:
        record = material["evidence"][0]
        return {
            "schema_version": "grounded-answer-v2",
            "answer_markdown": "A retrieved study reports hERG blockade in a screened series.",
            "claims": [{"local_ref": "lit", "kind": "scientific",
                        "text": "A retrieved study reports hERG blockade in a screened series.",
                        "citation_ids": [record["evidence_id"]]}],
            "limitations": [{"code": "evidence_scope_limited", "text": ""}],
        }

    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, _synthesis(state, answer=answer))
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        run = await _post(client, session_id, "Should hERG stop us developing this compound?")
        decision = await _decision_state(client, session_id, run["run_id"])

    assert run["status"] == "completed", {k: v for k, v in run.items() if "fail" in k or "error" in k}
    steps = decision["workflow"]["steps"]
    (route,) = [s for s in steps if s.get("decision") in ("search", "no_search")]
    assert route["decision"] == "search"
    searches = [s for s in steps if s["tool"] == "search_toxicology_evidence"]
    assert searches and searches[0]["reason"].startswith("fixed query: ")
    assert any(s["tool"] == "read_evidence_text" and s["status"] == "completed" for s in steps)
    material = state["materials"][0]
    assert material["full_text"][0]["text_access"] == "full_text_retrieved"
    assert state["submitted"]["status"] == "completed", state["submitted"]


async def test_an_exposure_question_asks_the_researcher_instead_of_guessing(db, arm_b, tmp_path):
    record = frozen_screening()
    state: dict = {}
    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, _synthesis(state, answer=_screening_answer))
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        run = await _post(client, session_id, "At what exposure would this matter?")
        decision = await _decision_state(client, session_id, run["run_id"])

    material = state["materials"][0]
    assert material["needs_from_researcher"] == [
        "the free plasma Cmax (or exposure) at the intended dose"
    ]
    assert any(s.get("decision") == "ask_researcher" for s in decision["workflow"]["steps"])


async def test_b_goes_through_the_same_release_gate_and_repairs_in_its_synthesis_step(
    db, arm_b, tmp_path, monkeypatch,
):
    monkeypatch.setenv("TOXAGENT_FLAG_CLAIM_REVIEWER_V1", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_RELEASE_GATE_V1", "1")
    record = frozen_screening()
    state: dict = {"verdict": "partially_supported"}
    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, _synthesis(state, answer=_screening_answer))
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        run = await _post(client, session_id, "What is the estimated pIC50 here?")
        messages = (await client.get(f"/v1/sessions/{session_id}/messages",
                                     headers=AUTH)).json()["messages"]
        answer_id = next(p["content"]["answer_id"] for p in messages[-1]["parts"]
                         if p["type"] == "answer_ref")
        answer = (await client.get(f"/v1/sessions/{session_id}/answers/{answer_id}",
                                   headers=AUTH)).json()

    assert run["status"] == "completed", {k: v for k, v in run.items() if "fail" in k or "error" in k}
    # synthesis, review (blocks), repair in the same one-tool step, review again
    assert state["profiles"] == ["workflow_synthesis", "claim_review", "workflow_synthesis",
                                 "claim_review"]
    assert answer["release"]["outcome"] == "repaired_and_reviewed"
