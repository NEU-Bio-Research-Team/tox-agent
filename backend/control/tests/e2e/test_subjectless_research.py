"""W9-08: a literature question does not need a molecule (flag subjectless_research_v1).

Before this, "find literature on hERG block by macrolides" in a session with no
analysis was answered with a clarification asking for a molecule
(``research_subject_missing``): a product decision exposed while building the
SciFact transfer study (W7-04). With the flag on it runs as a decision-support
turn on a subjectless case, and the search is judged against the endpoint.
"""
from __future__ import annotations

import pytest

from tests.e2e.test_scientific_case_e2e import _answer, _post
from tests.e2e.test_scripted_runtime import _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor
from tests.support.research import ACCEPTED_HIT, StubResearchProvider

pytestmark = pytest.mark.anyio

QUESTION = "Find literature on hERG channel blockade by macrolide antibiotics."


async def test_a_literature_question_without_a_molecule_is_answered(db, monkeypatch):
    for flag in ("SUBJECTLESS_RESEARCH_V1", "SCIENTIFIC_CASE_V1", "ANSWER_DRAFT_V2"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{flag}", "1")
    state: dict = {}

    async def script(turn) -> None:
        state["prompt"] = turn.system_prompt
        found = await turn.call_tool("search_toxicology_evidence", {
            "query": "hERG blockade macrolide", "endpoint": "herg", "limit": 3,
        })
        state["search"] = found
        evidence_id = found["model_view"]["results"][0]["evidence_id"]
        await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        state["answer"] = await turn.call_tool("submit_grounded_answer", _answer(
            evidence_id, "One retrieved screening study reports hERG channel blockade.",
        ))

    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        run = await _post(client, session_id, QUESTION)
        assert run["status"] == "completed", run
        assert state["search"]["status"] == "completed", state["search"]
        assert state["answer"]["status"] == "completed", state["answer"]
        assert "names no molecule this session has analysed" in state["prompt"]
        cases = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()["cases"]
        assert [c["subject_key"] for c in cases] == ["none"]


async def test_with_the_flag_off_the_router_still_asks_for_a_molecule(db, monkeypatch):
    monkeypatch.delenv("TOXAGENT_FLAG_SUBJECTLESS_RESEARCH_V1", raising=False)
    async with api_client(db, StubPredictor()) as client:
        session_id = await _new_session(client)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"intent_hint": "auto", "content": [{"type": "text", "text": QUESTION}]},
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        messages = (await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)).json()
        text = str(messages)
        assert "research_subject_missing" in text
        schema = client.app.state.tool_registry.get("search_toxicology_evidence").json_schema()
        assert "analysis_id" in schema["required"]
