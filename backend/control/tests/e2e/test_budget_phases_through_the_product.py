"""#46/#53 through the product: every call counts, and every call has a phase.

Under total accounting the answer, its review and its one repair are
guaranteed their reserved calls however much the investigation read; invalid
calls are recorded and capped, never free; and the run's calls and usage are
reported per phase -- investigator, hard correction, review, repair, review
again -- over every dispatch. A preset is declared here small enough to reach
its limits in a scripted turn; the semantics are investigator-pilot-v2's.
"""
from __future__ import annotations

import json
import re

import pytest

from tests.e2e.test_release_gate import ACCEPTED_HIT, OVERREACH, REPAIR_MARKER, SUPPORTED, _draft
from tests.e2e.test_scientific_case_e2e import _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor
from tests.support.research import StubResearchProvider
from toxagent.platform import config

pytestmark = pytest.mark.anyio

_BUNDLE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


@pytest.fixture
def small_total(monkeypatch):
    monkeypatch.setitem(config.RUN_BUDGET_PRESETS, "test-total-v2", {
        "max_tool_calls": 8, "reserved_submission_slots": 5, "max_invalid_calls": 2,
        "max_searches": 3, "max_evidence_reads": 8, "max_full_text_reads": 4,
        "max_repairs": 1, "review_reserve_s": 1,
        "call_accounting": config.TOTAL_CALL_ACCOUNTING,
    })
    monkeypatch.setenv("TOXAGENT_RUN_BUDGET_PRESET", "test-total-v2")
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_CLAIM_REVIEWER_V1", "1")
    monkeypatch.setenv("TOXAGENT_RELEASE_POLICY", "strict-reviewed-v1")


def _script(state: dict):
    async def script(turn) -> None:
        profile = turn.tool_context.profile
        if profile == "claim_review":
            state["reviews"] = state.get("reviews", 0) + 1
            bundle = json.loads(_BUNDLE.search(turn.system_prompt).group(1))
            verdict = "not_supported" if state["reviews"] == 1 else "supported"
            await turn.call_tool("submit_claim_review", {"reviews": [
                {"claim_id": c["claim_id"],
                 "verdict": verdict if c["text"] == OVERREACH else "supported",
                 "reason": "judged"} for c in bundle["claims"]
            ]})
            return
        if REPAIR_MARKER in turn.system_prompt:
            state["repair"] = await turn.call_tool(
                "submit_grounded_answer", _draft(state["evidence_id"], {"lit": SUPPORTED}),
            )
            return
        found = await turn.call_tool("search_toxicology_evidence", {
            "analysis_id": state["analysis_id"], "query": "hERG blockade", "limit": 3,
        })
        evidence_id = state["evidence_id"] = found["model_view"]["results"][0]["evidence_id"]
        await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        # The third gathering call is the last one gathering may spend (8 - 5).
        state["third"] = await turn.call_tool("get_artifact_inventory", {})
        state["fourth"] = await turn.call_tool("get_analysis_slice", {
            "analysis_id": state["analysis_id"], "section": "herg",
        })
        state["invalid"] = [await turn.call_tool("get_evidence_record", {"nope": i})
                            for i in range(3)]
        # A first draft that fails validation, then its hard correction.
        state["refused"] = await turn.call_tool("submit_grounded_answer", {
            **_draft(evidence_id, {"lit": SUPPORTED, "overreach": OVERREACH}),
            "limitations": [],
        })
        state["first"] = await turn.call_tool(
            "submit_grounded_answer", _draft(evidence_id, {"lit": SUPPORTED, "overreach": OVERREACH}),
        )

    return script


async def test_submissions_keep_their_calls_and_every_call_has_a_phase(db, small_total):
    state: dict = {}
    async with api_client(db, StubPredictor(),
                          research_provider=StubResearchProvider(hits=[ACCEPTED_HIT])) as client:
        await _install_scripted_runtime(client, _script(state))
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id, "Should hERG stop us developing this compound?")
        detail = (await client.get(f"/v1/sessions/{session_id}/runs/{run['run_id']}",
                                   headers=AUTH)).json()
        result = (await client.get(f"/v1/sessions/{session_id}/runs/{run['run_id']}/result",
                                   headers=AUTH)).json()
        product = (await client.get("/v1/system/effective-product", headers=AUTH)).json()

    assert state["third"]["status"] == "completed"
    fourth = state["fourth"]
    assert fourth["status"] == "error" and fourth["error"]["code"] == "tool_denied"
    assert "reserved calls" in fourth["error"]["message"]
    # Invalid calls are recorded; past the cap every call is refused for it.
    first_invalid, second_invalid, third_invalid = state["invalid"]
    assert first_invalid["error"]["code"] == "invalid_request"
    assert third_invalid["error"]["code"] == "tool_denied"
    # The answer, its correction, two reviews and the repair still ran.
    assert state["first"]["status"] == "completed", state["first"]
    assert state["repair"]["status"] == "completed", state["repair"]
    assert state["reviews"] == 2
    assert result["disposition"] == "publish"
    assert result["review_status"] == "repaired_and_passed"

    calls = detail["tool_calls"]
    admitted = [c for c in calls if c["status"] != "denied"]
    assert len(admitted) <= 8
    phases = {(c["tool_name"], c["phase"]) for c in calls}
    assert ("search_toxicology_evidence", "investigator") in phases
    assert ("submit_grounded_answer", "hard_correction") in phases
    assert ("submit_claim_review", "review_initial") in phases
    assert ("submit_grounded_answer", "semantic_repair") in phases
    assert ("submit_claim_review", "review_final") in phases
    by_phase = detail["phase_usage"]["phases"]
    for phase in ("investigator", "review_initial", "semantic_repair", "review_final"):
        assert by_phase[phase]["dispatches"] == 1, phase
    assert by_phase["investigator"]["tool_calls"]["denied"] >= 1
    budget = product["intents"]["decision_support"]["budget"]
    assert budget["call_accounting"] == "total-logical-invocations-v2"
    assert budget["reserved_submission_slots"] == 5


async def test_the_investigator_cannot_spend_the_review_reserve(db, small_total, monkeypatch):
    config.RUN_BUDGET_PRESETS["test-total-v2"]["review_reserve_s"] = 100_000
    state: dict = {}
    async with api_client(db, StubPredictor(),
                          research_provider=StubResearchProvider(hits=[ACCEPTED_HIT])) as client:
        await _install_scripted_runtime(client, _script(state))
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id, "Should hERG stop us developing this compound?")
        result = (await client.get(f"/v1/sessions/{session_id}/runs/{run['run_id']}/result",
                                   headers=AUTH)).json()

    assert run["status"] == "failed"
    assert run["failure_code"] == "phase_budget_exhausted"
    assert result["disposition"] == "withhold"
    assert result["execution_outcome"] == "budget_exhausted"
