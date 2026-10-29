"""#62: the deterministic fallback answers about the run's subject, not the session's.

The fallback is the one answer the server builds without the agent, so it is
the easiest place for an answer to leave the subject it was about. Before
#62 it read whichever analysis the session had *active*: a run bound to
analysis A, in a session where B was analysed last, fell back on B's
prediction -- a well-formed answer about the wrong molecule.
"""
from __future__ import annotations

import pytest

from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client, wait_for_run
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


@pytest.fixture
def flags_on(monkeypatch):
    for name in ("ANSWER_DRAFT_V2", "SUBJECT_BINDING_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{name}", "1")


#: A draft the validator refuses every time: its one claim cites an
#: observation that does not exist, so the run spends its attempts and falls back.
_UNGROUNDED = {
    "answer_markdown": "The hERG probability is {{p}}.",
    "claims": [{"local_ref": "p", "kind": "numeric", "text": "The hERG probability.",
                "observation_id": "obs_" + "0" * 32, "field_path": "predictions.herg.probability_blocker"}],
    "limitations": [],
}


async def _ask(client, session_id: str, text: str, subject_ref: dict | None) -> dict:
    body: dict = {"content": [{"type": "text", "text": text}]}
    if subject_ref is not None:
        body["subject_ref"] = subject_ref
    submitted = await client.post(f"/v1/sessions/{session_id}/messages", headers=AUTH, json=body)
    assert submitted.status_code == 202, submitted.text
    return await wait_for_run(client, session_id, submitted.json()["run_id"])


async def _observation_ids(db, analysis_id: str) -> set[str]:
    async with db.unit_of_work() as uow:
        return {o.id for o in await uow.observations.list_for_analysis(analysis_id)}


async def test_a_bound_runs_fallback_cites_its_own_subjects_prediction(db, flags_on):
    results: list[dict] = []

    async def script(turn) -> None:
        for _ in range(4):
            result = await turn.call_tool("submit_grounded_answer", _UNGROUNDED)
            results.append(result)
            if result["status"] == "completed":
                return

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        first = await _analyse(client, session_id)
        second = await _analyse(client, session_id, smiles=CAFFEINE)
        assert first != second
        run = await _ask(client, session_id, "Is hERG a concern for this compound?",
                         {"kind": "analysis", "analysis_id": first})

    assert run["status"] == "completed", run
    fallback = results[-1]
    assert fallback["status"] == "completed", fallback
    assert fallback["model_view"]["is_fallback"] is True
    async with db.unit_of_work() as uow:
        answer = await uow.answers.get_for_run(run["run_id"])
    cited = {claim.observation_id for claim in answer.claims}
    assert cited, answer
    first_observations = await _observation_ids(db, first)
    second_observations = await _observation_ids(db, second)
    assert first_observations and second_observations
    assert cited <= first_observations
    assert not cited & second_observations
