"""Admission refuses what no handler can serve, before writing a run (I02).

The bug: `report_qa` and `attribution` were not gated at all, and
`evidence_research` was gated on a research provider *object existing* rather
than on a handler being registered. In the default predictor-only stack a
request therefore passed admission, wrote a message and a run, and only then
failed inside the scheduler with "no handler is registered".

Two costs, both visible here: the caller got a failed run instead of an
immediate, explainable refusal, and the database carried a run row for work
the deployment was never able to do.
"""
from __future__ import annotations

import pytest

from tests.support.api import AUTH, api_client, wait_for_run
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

ASKS = {
    "ask_report": "What does the hERG number mean?",
    "request_attribution": "Which atoms drive the hERG prediction?",
    "research_evidence": "What does the literature say about hERG blockers?",
}


async def _session_with_an_analysis(client) -> str:
    response = await client.post("/v1/sessions", json={}, headers=AUTH)
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"molecule": {"smiles": ASPIRIN}},
        headers=AUTH,
    )
    assert submitted.status_code == 202, submitted.text
    await wait_for_run(client, session_id, submitted.json()["run_id"])
    return session_id


async def _ask(client, session_id: str, hint: str):
    return await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"intent_hint": hint, "content": [{"type": "text", "text": ASKS[hint]}]},
        headers=AUTH,
    )


@pytest.mark.parametrize("hint", sorted(ASKS))
async def test_a_conversational_ask_is_refused_deterministically_without_a_runtime(db, hint):
    async with api_client(db, StubPredictor()) as client:
        session_id = await _session_with_an_analysis(client)
        submitted = await _ask(client, session_id, hint)
        assert submitted.status_code == 202, submitted.text

        run = await wait_for_run(client, session_id, submitted.json()["run_id"])
        # Deterministic refusal, not a run that reached the scheduler and died.
        assert run["status"] == "completed", run

        messages = (
            await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        ).json()["messages"]
        reply = messages[-1]
        content = reply["parts"][0]["content"]
        assert content["code"] == "capability_unavailable"
        assert content["capability"] == {
            "ask_report": "report_qa",
            "request_attribution": "attribution",
            "research_evidence": "evidence_research",
        }[hint]
        # Each gated intent gets its own message. `report_qa` and
        # `attribution` used to fall through to the research wording, which
        # told the reader about a literature feature they had not asked for.
        assert content["message"]
        if hint == "ask_report":
            assert "literature" not in content["message"].lower()


@pytest.mark.parametrize("hint", sorted(ASKS))
async def test_a_refused_ask_leaves_no_failed_run_behind(db, hint):
    async with api_client(db, StubPredictor()) as client:
        session_id = await _session_with_an_analysis(client)
        submitted = await _ask(client, session_id, hint)
        run_id = submitted.json()["run_id"]

        run = await wait_for_run(client, session_id, run_id)
        assert run["status"] != "failed", run
        assert run.get("failure_code") is None
