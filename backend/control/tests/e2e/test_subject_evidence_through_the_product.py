"""AG-EVIDENCE through the product: literature found about a screening.

With subject_binding_v1, on the scripted runtime and a frozen snapshot:

* a screening-only run searches about its own ligand on its own target with
  no analysis at all: the paper on that target is direct, the paper on the
  ligand's other target is contextual, the paper on the target's other ligand
  is not citable;
* a record about another target cannot be direct evidence, and a relation
  cannot restate the target an activity artifact states;
* a search cannot be pointed at another subject, and the researcher's data
  scope stops it as it does for an analysis;
* a target outside the catalog is reported as such, and nothing about it is
  promoted;
* arm B searches a screening by templates built from its binding -- its
  ligand, target and readout -- and never for hERG.
"""
from __future__ import annotations

import json

import pytest

from tests.e2e.test_fixed_workflow_arm import _decision_state, _synthesis
from tests.e2e.test_scripted_runtime import _install_scripted_runtime, _new_session
from tests.support.activity import frozen_screening, replay_file, screening_id
from tests.support.api import AUTH, api_client, settings, wait_for_run
from tests.support.predictor import StubPredictor
from toxagent.domain.provenance import content_sha256
from toxagent.platform.config import ActivitySettings, ResearchSettings
from toxagent.research.providers import build_provider

pytestmark = pytest.mark.anyio

REQUIRED = [
    "toxicity_not_run_not_negative", "pic50_not_therapeutic_efficacy",
    "activity_model_not_admitted", "activity_interval_is_marginal_coverage",
    "policy_decision_not_a_measurement",
]

RECORDS = {
    "MED:1": "Zaltinib inhibits EGFR kinase in a biochemical IC50 assay",
    "MED:2": "Zaltinib blocks the hERG potassium channel in patch clamp",
    "MED:3": "Gefitinib inhibits EGFR kinase in a biochemical IC50 assay",
}


@pytest.fixture
def flags_on(monkeypatch):
    for name in ("ANSWER_DRAFT_V2", "ACTIVITY_REPLAY_V1", "SUBJECT_BINDING_V1",
                 "SCIENTIFIC_CASE_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{name}", "1")


def _provider(tmp_path, *, index_key: str = "egfr"):
    evidence = {
        "search": {index_key: list(RECORDS), "zaltinib": list(RECORDS)},
        "records": {
            rid: {"provider_record_id": rid, "source_type": "article", "title": title,
                  "authors": ["Doe J"], "published_at": "2025-01-01",
                  "identifier": {"pmid": rid.split(":")[1]},
                  "abstract_or_excerpt": title + ". Values are reported in the results."}
            for rid, title in RECORDS.items()
        },
    }
    body = {"fixture_version": "eval-fixture-v1", "name": "subject-evidence",
            "evidence": evidence}
    body["content_sha256"] = content_sha256(body)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(body))
    return build_provider(ResearchSettings(provider="snapshot", snapshot_path=str(path)))


def _config(tmp_path, *records):
    path, sha = replay_file(tmp_path, *records)
    return settings(activity=ActivitySettings(replay_path=str(path), replay_sha256=sha))


def _ref(record) -> dict:
    return {"kind": "activity_screening", "screening_id": screening_id(record)}


async def _session(client, *records) -> str:
    session_id = await _new_session(client)
    for record in records:
        response = await client.post(f"/v1/sessions/{session_id}/screenings",
                                     json={"screening_id": screening_id(record)}, headers=AUTH)
        assert response.status_code == 201, response.text
    return session_id


async def _ask(client, session_id, text, record) -> dict:
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"content": [{"type": "text", "text": text}], "subject_ref": _ref(record)},
        headers=AUTH,
    )
    assert submitted.status_code == 202, submitted.text
    return await wait_for_run(client, session_id, submitted.json()["run_id"])


def _by_title(results: list[dict]) -> dict[str, dict]:
    return {r["title"]: r for r in results}


async def test_a_screening_is_searched_on_its_ligand_and_target(db, flags_on, tmp_path):
    record = frozen_screening(ligand_names=("zaltinib",))
    seen: dict = {}

    async def script(turn) -> None:
        found = await turn.call_tool("search_toxicology_evidence",
                                     {"query": "zaltinib EGFR", "limit": 5})
        seen["found"] = found
        ids = {r["title"]: r["evidence_id"] for r in found["model_view"]["results"]}
        for evidence_id in ids.values():
            await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        estimate = await turn.call_tool("get_activity_snapshot",
                                        {"screening_id": screening_id(record)})
        state = await turn.call_tool("get_screening_state",
                                     {"screening_id": screening_id(record)})
        decision = await turn.call_tool("get_prioritisation_decision",
                                        {"screening_id": screening_id(record)})
        other_target = ids[RECORDS["MED:2"]]

        def draft(relations):
            return {
                "schema_version": "grounded-answer-v2",
                "answer_markdown": "The estimate is {{p}}; a study reports EGFR inhibition.",
                "claims": [
                    {"local_ref": "p", "kind": "numeric", "text": "The estimate.",
                     "observation_id": estimate["observation_ids"][0],
                     "field_path": "estimate.predicted_pic50", "transform": "round:1"},
                    {"local_ref": "s", "kind": "scientific", "text": "A study tested it.",
                     "citation_ids": [ids[RECORDS["MED:1"]]]},
                    {"local_ref": "t", "kind": "classification", "text": "Toxicity ran.",
                     "observation_id": state["observation_ids"][0], "field_path": "toxicity.run"},
                    {"local_ref": "d", "kind": "classification", "text": "Decision.",
                     "observation_id": decision["observation_ids"][0], "field_path": "status"},
                ],
                "limitations": [{"code": c, "text": ""}
                                for c in (*REQUIRED, "evidence_scope_limited")],
                "evidence_relations": relations,
            }

        seen["direct_from_other_target"] = await turn.call_tool("submit_grounded_answer", draft([
            {"proposition": "zaltinib is active on EGFR", "source_class": "external_experimental",
             "source_id": other_target, "relation": "contradicts", "directness": "direct",
             "reason_codes": ["measured"]},
        ]))
        seen["restated_target"] = await turn.call_tool("submit_grounded_answer", draft([
            {"proposition": "zaltinib is active on EGFR", "source_class": "predictor_fact",
             "source_id": estimate["observation_ids"][0], "relation": "supports",
             "directness": "direct", "reason_codes": ["model_estimate"], "target_id": "KCNH2"},
        ]))

    async with api_client(db, StubPredictor(), config=_config(tmp_path, record),
                          research_provider=_provider(tmp_path)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _session(client, record)
        run = await _ask(client, session_id, "Is there measured EGFR activity for this ligand?",
                         record)
        evidence = (await client.get(f"/v1/sessions/{session_id}/evidence",
                                     params={"status": "all"}, headers=AUTH)).json()["evidence"]
        drafts = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/answer-drafts", headers=AUTH,
        )).json()["drafts"]

    view = seen["found"]["model_view"]
    assert view["subject"]["screening_id"] == screening_id(record)
    assert view["subject"]["target_capability"] == "in_catalog"
    assert set(_by_title(view["results"])) == {RECORDS["MED:1"], RECORDS["MED:2"]}
    stored = {e["title"]: e for e in evidence}
    assert stored[RECORDS["MED:1"]]["relevance_assessment"]["relevance"] == "direct"
    assert stored[RECORDS["MED:1"]]["relevance_assessment"]["assessed_scope"]["target_id"] == "EGFR"
    assert stored[RECORDS["MED:2"]]["relevance_assessment"]["relevance"] == "contextual"
    assert stored[RECORDS["MED:3"]]["status"] == "rejected"

    def codes(result):
        return [v["code"] for v in result["error"]["details"]["violations"]]

    assert codes(seen["direct_from_other_target"]) == ["evidence_relation_direct_out_of_subject"]
    # The second draft spends the run's last attempt, so the run falls back;
    # the draft record keeps why it was refused.
    assert seen["restated_target"]["model_view"]["is_fallback"] is True
    second = next(d for d in drafts if d["generation"] == 2)
    assert [v["code"] for v in second["violations"]] == [
        "evidence_relation_scope_contradicts_source"
    ]


async def test_a_search_cannot_change_the_subject_or_pass_the_data_scope(
    db, flags_on, tmp_path,
):
    egfr = frozen_screening(ligand_names=("zaltinib",))
    herg = frozen_screening(target_id="KCNH2", uniprot_id="Q12809", ligand_names=("zaltinib",))
    seen: list = []

    async def script(turn) -> None:
        seen.append(await turn.call_tool("search_toxicology_evidence", {
            "query": "zaltinib", "screening_id": screening_id(herg),
        }))
        seen.append(await turn.call_tool("search_toxicology_evidence", {"query": "zaltinib"}))
        turn.say("noted")

    async with api_client(db, StubPredictor(), config=_config(tmp_path, egfr, herg),
                          research_provider=_provider(tmp_path)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _session(client, egfr, herg)
        await _ask(client, session_id, "Any literature?", egfr)
        (case,) = (await client.get(f"/v1/sessions/{session_id}/cases",
                                    headers=AUTH)).json()["cases"]
        scoped = await client.post(
            f"/v1/sessions/{session_id}/cases/{case['case_id']}/scope",
            json={"external_search": False, "reason": "confidential series"}, headers=AUTH,
        )
        assert scoped.status_code == 200, scoped.text
        await _ask(client, session_id, "Any literature now?", egfr)

    mismatch, allowed, _, restricted = seen
    assert mismatch["status"] == "error"
    assert mismatch["error"]["details"]["reason"] == "subject_mismatch"
    assert allowed["status"] == "completed"
    assert restricted["status"] == "error"
    assert restricted["error"]["details"]["reason"] == "case_data_scope"


async def test_a_target_outside_the_catalog_promotes_nothing(db, flags_on, tmp_path):
    record = frozen_screening(target_id="BRAF", uniprot_id="P15056",
                              ligand_names=("zaltinib",))
    seen: dict = {}

    async def script(turn) -> None:
        seen["found"] = await turn.call_tool("search_toxicology_evidence", {"query": "zaltinib"})
        turn.say("noted")

    async with api_client(db, StubPredictor(), config=_config(tmp_path, record),
                          research_provider=_provider(tmp_path)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _session(client, record)
        await _ask(client, session_id, "Any literature on this target?", record)

    view = seen["found"]["model_view"]
    assert view["subject"]["target_capability"] == "not_in_catalog"
    assert view["results"] == [] and view["promoted"] == 0


async def test_arm_b_searches_a_screening_from_its_binding(db, flags_on, tmp_path, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_FIXED_WORKFLOW_V1", "1")
    record = frozen_screening(ligand_names=("zaltinib",))
    state: dict = {}

    def answer(material: dict) -> dict:
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

    async with api_client(db, StubPredictor(), config=_config(tmp_path, record),
                          research_provider=_provider(tmp_path)) as client:
        await _install_scripted_runtime(client, _synthesis(state, answer=answer))
        session_id = await _session(client, record)
        run = await _ask(client, session_id, "Why was this sent to review?", record)
        decision_state = await _decision_state(client, session_id, run["run_id"])

    assert run["status"] == "completed", run
    steps = decision_state["workflow"]["steps"]
    assert decision_state["workflow"]["version"] == "fixed-workflow-b-v2"
    searches = [s for s in steps if s.get("tool") == "search_toxicology_evidence"]
    assert searches and all("EGFR" in s["reason"] for s in searches)
    assert not any("hERG" in s["reason"] for s in searches)
    decided = next(s for s in steps if s.get("decision") == "search")
    assert "review" in decided["reason"]
    (material,) = state["materials"]
    assert [e["title"] for e in material["evidence"]][:1] == [RECORDS["MED:1"]]
