"""#50 through the product: a toxicity-only case names its frozen analysis.

A toxicity-only study case has no screening; its subject is a toxicity
analysis frozen in the deployment's pinned replay file. Every arm must be
given that same analysis without the predictor being called, so a message
names it by its origin id and the run is bound to a session-local copy:

* the copy is made in the run's transaction, hash-checked, never by calling
  the predictor;
* the binding keeps the origin id and hash beside the local id -- what study
  arms compare, since the local id is minted per session;
* a second turn about the same frozen analysis reuses the copy and continues
  the same case;
* an analysis the replay file does not hold is a 404, never a predictor run.
"""
from __future__ import annotations

import pytest

from tests.e2e.test_scripted_runtime import _install_scripted_runtime, _new_session
from tests.support.activity import frozen_analysis, replay_file
from tests.support.api import AUTH, api_client, settings, wait_for_run
from tests.support.predictor import StubPredictor
from toxagent.platform.config import ActivitySettings

pytestmark = pytest.mark.anyio


@pytest.fixture
def flags_on(monkeypatch):
    for name in ("ANSWER_DRAFT_V2", "ACTIVITY_REPLAY_V1", "SUBJECT_BINDING_V1",
                 "SCIENTIFIC_CASE_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{name}", "1")


def _config(tmp_path, frozen: dict):
    path, sha = replay_file(tmp_path, analyses={frozen["origin_analysis_id"]: frozen})
    return settings(activity=ActivitySettings(replay_path=str(path), replay_sha256=sha))


def _ref(frozen: dict) -> dict:
    return {"kind": "frozen_analysis", "analysis_ref": frozen["origin_analysis_id"]}


async def _ask(client, session_id: str, text: str, subject_ref: dict) -> dict:
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages", headers=AUTH,
        json={"content": [{"type": "text", "text": text}], "subject_ref": subject_ref},
    )
    assert submitted.status_code == 202, submitted.text
    return await wait_for_run(client, session_id, submitted.json()["run_id"])


async def _binding(client, session_id: str, run_id: str) -> dict:
    run = (await client.get(f"/v1/sessions/{session_id}/runs/{run_id}", headers=AUTH)).json()
    return run["configuration_snapshot"]["subject_binding"]


async def test_a_frozen_analysis_is_bound_by_reference_without_the_predictor(
    db, flags_on, tmp_path,
):
    frozen = frozen_analysis(herg_probability=0.81)
    predictor = StubPredictor()
    seen: list[dict] = []

    async def script(turn) -> None:
        seen.append((await turn.call_tool("get_artifact_inventory", {}))["model_view"])
        turn.say("noted")

    async with api_client(db, predictor, config=_config(tmp_path, frozen)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        first = await _ask(client, session_id, "What does the hERG score mean?", _ref(frozen))
        second = await _ask(client, session_id, "And the Tox21 assays?", _ref(frozen))
        bindings = [await _binding(client, session_id, r["run_id"]) for r in (first, second)]
        local_id = bindings[0]["subject_ref"]["analysis_id"]
        analysis = (await client.get(f"/v1/sessions/{session_id}/analyses/{local_id}",
                                     headers=AUTH)).json()
        cases = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()

    # Never computed: the predictor saw no request at all.
    assert predictor.requests == []
    assert [b["kind"] for b in bindings] == ["analysis", "analysis"]
    refs = {r["kind"]: r for r in bindings[0]["artifact_refs"]}
    origin = refs["frozen_analysis"]
    assert origin["source_id"] == frozen["origin_analysis_id"]
    assert origin["content_sha256"] == frozen["content_sha256"]
    assert origin["local_id"] == local_id != frozen["origin_analysis_id"]
    # The copy is the frozen analysis, byte for byte.
    assert refs["analysis"]["content_sha256"] == frozen["content_sha256"]
    assert analysis["canonical_smiles"] == frozen["canonical_smiles"]
    # The second turn reuses the copy and the case.
    assert bindings[1]["subject_ref"]["analysis_id"] == local_id
    assert bindings[0]["subject_key"] == bindings[1]["subject_key"] == f"analysis:{local_id}"
    assert len(cases["cases"]) == 1
    assert [v["subject"]["subject_ref"]["analysis_id"] for v in seen] == [local_id, local_id]


async def test_an_analysis_the_replay_file_does_not_hold_is_not_found(db, flags_on, tmp_path):
    frozen = frozen_analysis()
    predictor = StubPredictor()
    async with api_client(db, predictor, config=_config(tmp_path, frozen)) as client:
        session_id = await _new_session(client)
        response = await client.post(
            f"/v1/sessions/{session_id}/messages", headers=AUTH,
            json={"content": [{"type": "text", "text": "Explain it."}],
                  "subject_ref": {"kind": "frozen_analysis",
                                  "analysis_ref": "ana_" + "0" * 32}},
        )
        mixed = await client.post(
            f"/v1/sessions/{session_id}/messages", headers=AUTH,
            json={"content": [{"type": "text", "text": "Explain it."}],
                  "subject_ref": {"kind": "frozen_analysis",
                                  "analysis_id": frozen["origin_analysis_id"]}},
        )
    assert response.status_code == 404, response.text
    assert mixed.status_code == 400  # a validation error
    assert predictor.requests == []
