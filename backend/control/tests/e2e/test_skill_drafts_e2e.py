"""W9-11 end to end: propose, review by another expert, export, promote.

RETHINK §4.8: a draft a model or a person writes is reviewed by an expert and
never activates by itself. Through the product API and the scripted runtime.
"""
from __future__ import annotations

import json
import shutil

import pytest

from toxagent.application.investigation.skill_catalog import load_catalog
from toxagent.application.investigation.skill_drafts import compose_package
from toxagent.config import PACKAGE_ROOT
from tests.e2e.test_scientific_case_e2e import _answer, _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, EXPERT_AUTH, OTHER_AUTH, REVIEWER_AUTH, api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

BODY = "# Weigh species differences\n\n" + "Check whether each source's species matches the question. " * 4


@pytest.fixture
def drafts_on(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_SKILL_DRAFTS_V1", "1")


def _proposal(**overrides) -> dict:
    skill_md, manifest = compose_package(
        skill_id="weigh-species-differences",
        description="Use when evidence comes from a species other than the one the decision is about.",
        body=BODY, required_capabilities=["get_evidence_record"],
        output_contract="uncertainties naming the species gap",
    )
    body = {"skill_md": skill_md, "manifest": manifest, "references": {},
            "rationale": "rat data kept being read as if it were human"}
    body.update(overrides)
    return body


async def test_a_person_proposes_an_expert_reviews_and_it_ships_only_by_promotion(
    db, drafts_on, tmp_path
):
    async with api_client(db, StubPredictor()) as client:
        catalog_before = client.app.state.skill_catalog.catalog_sha256
        created = await client.post("/v1/skill-drafts", json=_proposal(), headers=AUTH)
        assert created.status_code == 201, created.text
        draft_id = created.json()["draft_id"]
        assert created.json()["status"] == "proposed"

        # Not visible to someone else; the expert list shows it.
        assert (await client.get(f"/v1/skill-drafts/{draft_id}", headers=OTHER_AUTH)).status_code == 404
        listed = (await client.get("/v1/skill-drafts?status=proposed", headers=REVIEWER_AUTH)).json()
        assert [d["draft_id"] for d in listed["drafts"]] == [draft_id]
        assert "skill_md" not in listed["drafts"][0]

        # Its author holds the expert role too, and still cannot approve it.
        self_review = await client.post(
            f"/v1/skill-drafts/{draft_id}:review",
            json={"decision": "approve", "note": "looks good"}, headers=EXPERT_AUTH,
        )
        assert self_review.status_code == 403, self_review.text
        not_yet = await client.get(f"/v1/skill-drafts/{draft_id}/package", headers=AUTH)
        assert not_yet.status_code == 409

        approved = await client.post(
            f"/v1/skill-drafts/{draft_id}:review",
            json={"decision": "approve", "note": "matches how we read cross-species data"},
            headers=REVIEWER_AUTH,
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["review"]["reviewer"] == "user-3"
        # Approval does not reach what a run is offered.
        assert client.app.state.skill_catalog.catalog_sha256 == catalog_before
        assert client.app.state.skill_catalog.get("weigh-species-differences") is None

        package = (await client.get(f"/v1/skill-drafts/{draft_id}/package", headers=AUTH)).json()

    # Promotion is a file change a person reviews and commits.
    profiles = tmp_path / "agent_profiles"
    shutil.copytree(PACKAGE_ROOT / "agent_profiles", profiles)
    path = tmp_path / "package.json"
    path.write_text(json.dumps(package))
    from scripts.promote_skill_draft import main as promote

    assert promote([str(path), "--profiles-dir", str(profiles)]) == 0
    promoted = load_catalog(profiles).get("weigh-species-differences")
    assert promoted is not None and promoted.status == "active"

    tampered = {**package, "files": {**package["files"], "SKILL.md": package["files"]["SKILL.md"] + "x"}}
    path.write_text(json.dumps(tampered))
    with pytest.raises(SystemExit, match="digest"):
        promote([str(path), "--profiles-dir", str(profiles), "--replace"])


async def test_a_malformed_proposal_is_refused_with_the_catalogs_reason(db, drafts_on):
    async with api_client(db, StubPredictor()) as client:
        bad = _proposal()
        bad["manifest"] = {**bad["manifest"], "required_capabilities": ["bash"]}
        refused = await client.post("/v1/skill-drafts", json=bad, headers=AUTH)
        assert refused.status_code == 400
        assert "unknown tools" in refused.text


async def test_a_run_can_propose_a_draft_that_it_is_never_offered(db, drafts_on, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    seen: dict = {}

    async def script(turn) -> None:
        seen["proposal"] = await turn.call_tool("propose_skill_draft", {
            "skill_id": "weigh-species-differences",
            "description": "Use when evidence comes from a species other than the one decided about.",
            "instructions": BODY,
            "required_tools": ["get_evidence_record"],
            "output_contract": "uncertainties naming the species gap",
            "rationale": "this case mixed rat and human data without saying so",
        })
        await turn.call_tool("submit_grounded_answer", _answer(None, "Noted."))

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        run = await _post(client, session_id, "Is hERG a concern here?")
        assert run["status"] == "completed", run
        assert seen["proposal"]["status"] == "completed", seen["proposal"]
        draft_id = seen["proposal"]["model_view"]["draft_id"]
        draft = (await client.get(f"/v1/skill-drafts/{draft_id}", headers=EXPERT_AUTH)).json()
        assert draft["author"]["actor"] == "model"
        assert draft["author"]["run_id"] == run["run_id"]
        # The session owner is not its author, so an expert owner may review it.
        reviewed = await client.post(
            f"/v1/skill-drafts/{draft_id}:review",
            json={"decision": "reject", "note": "too close to assess-conflicting-evidence"},
            headers=EXPERT_AUTH,
        )
        assert reviewed.status_code == 200 and reviewed.json()["status"] == "rejected"


async def test_with_the_flag_off_there_are_no_drafts(db, monkeypatch):
    monkeypatch.delenv("TOXAGENT_FLAG_SKILL_DRAFTS_V1", raising=False)
    async with api_client(db, StubPredictor()) as client:
        assert (await client.post("/v1/skill-drafts", json=_proposal(), headers=AUTH)).status_code == 404
        assert client.app.state.tool_registry.get("propose_skill_draft") is None
