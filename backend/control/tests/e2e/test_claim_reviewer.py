"""W9-12: an independent reviewer of claim support, recorded and never applied.

RETHINK §4.4 step 5: a reviewer is one role measured on its own. With
``claim_reviewer_v1`` on, a decision-support run whose answer was accepted gets
one more runtime turn, under the one-tool ``claim_review`` profile, whose prompt
is the server-built bundle of claims and the sources each cites.
"""
from __future__ import annotations

import json
import re

import pytest

from tests.e2e.test_scientific_case_e2e import _answer, _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor
from tests.support.research import ACCEPTED_HIT, StubResearchProvider

pytestmark = pytest.mark.anyio

_BUNDLE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
TEXT = "A retrieved study reports hERG channel blockade in a screened series."


@pytest.fixture
def reviewer_on(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_CLAIM_REVIEWER_V1", "1")


def _script(state: dict, *, cite: bool = True):
    async def script(turn) -> None:
        if turn.tool_context.profile == "claim_review":
            state["review_prompt"] = turn.system_prompt
            bundle = json.loads(_BUNDLE.search(turn.system_prompt).group(1))
            state["bundle"] = bundle
            state["review"] = await turn.call_tool("submit_claim_review", {"reviews": [
                {"claim_id": claim["claim_id"], "verdict": "partially_supported",
                 "reason": "the study is about a series, not this compound"}
                for claim in bundle["claims"]
            ]})
            return
        state["turns"] = state.get("turns", 0) + 1
        evidence_id = None
        if cite:
            found = await turn.call_tool("search_toxicology_evidence", {
                "analysis_id": state["analysis_id"], "query": "hERG blockade", "limit": 3,
            })
            evidence_id = found["model_view"]["results"][0]["evidence_id"]
            await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        await turn.call_tool("submit_grounded_answer", _answer(evidence_id, TEXT))

    return script


async def _state(client, session_id: str, run_id: str) -> dict:
    response = await client.get(
        f"/v1/sessions/{session_id}/runs/{run_id}/decision-state", headers=AUTH
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_the_reviewer_judges_each_claim_from_the_sources_the_server_hands_it(
    db, reviewer_on
):
    state: dict = {}
    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, _script(state))
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id, "Should hERG stop us developing this compound?")
        assert run["status"] == "completed", run

        visible = client.app.state.tool_registry.visible_for("claim_review")
        assert [tool.name for tool in visible] == ["submit_claim_review"]
        [claim] = state["bundle"]["claims"]
        assert claim["text"] == TEXT
        assert claim["sources"][0]["ref"].startswith("evidence:")
        assert "hERG channel blockade" in claim["sources"][0]["excerpt"]
        assert state["review"]["status"] == "completed", state["review"]

        decision = await _state(client, session_id, run["run_id"])
        review = decision["claim_review"]
        assert review["status"] == "completed"
        assert review["counts"]["partially_supported"] == 1
        # The reviewer's turn is not the answering agent's work.
        assert decision["usage"]["tool_calls"] == 3

        messages = (await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)).json()
        assert TEXT in json.dumps(messages["messages"][-1])


async def test_an_answer_without_claims_is_not_reviewed(db, reviewer_on):
    state: dict = {}
    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, _script(state, cite=False))
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id, "Anything to note?")
        assert run["status"] == "completed", run
        assert "review_prompt" not in state
        review = (await _state(client, session_id, run["run_id"]))["claim_review"]
        assert review["status"] == "skipped"


async def test_with_the_flag_off_no_reviewer_turn_runs(db, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.delenv("TOXAGENT_FLAG_CLAIM_REVIEWER_V1", raising=False)
    state: dict = {}
    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, _script(state))
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id, "Should hERG stop us?")
        assert run["status"] == "completed", run
        assert "review_prompt" not in state
        assert (await _state(client, session_id, run["run_id"]))["claim_review"] == {}
