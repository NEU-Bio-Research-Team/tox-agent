"""The comparison study end to end against an in-process product (Wave 5).

The runner drives the product API the way the live pilot will — through the
same adapter, over HTTP (an ASGI transport here) — with the scripted runtime
standing in for the model. It checks what the lab will depend on: the arm is
verified against the deployment's effective product, a turn's structured
context reaches the case on a case arm, every run is traced raw, and the
packet built from the records carries no identity.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evals.investigation import packet as packet_module
from evals.investigation.adapters.predictor_template import PredictorTemplateAdapter
from evals.investigation.adapters.toxagent import ToxAgentAdapter
from evals.investigation.record import StudyStore
from evals.investigation.run import run_study
from tests.e2e.test_scripted_runtime import _install_scripted_runtime
from tests.support.api import USER_TOKEN, api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

CASE_IDS = ["inv-05-astemizole-inhouse-contradicts", "inv-06-aspirin-numeric-lookup"]


@pytest.fixture
def investigator(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_CASE_V1", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_SKILLS_V1", "1")


async def script(turn) -> None:
    case = await turn.call_tool("get_scientific_case", {})
    context_ids = [c["id"] for c in case["model_view"]["context"]]
    if not case["model_view"]["hypotheses"]:
        await turn.call_tool("update_scientific_case", {"operations": [
            {"op": "add_hypothesis", "statement": "The compound blocks hERG at relevant exposure",
             "hypothesis_kind": "model_signal", "refutation_condition": "A functional IC50 far above exposure"},
        ]})
    if context_ids:
        await turn.call_tool("read_scientific_skill", {"skill_id": "assess-conflicting-evidence"})
        await turn.call_tool("update_scientific_case", {"operations": [
            {"op": "record_evidence", "claim": "In-house flux shows weak inhibition at 1 µM",
             "source_class": "user_supplied", "source_ref": f"context:{context_ids[0]}",
             "stance": "contradicts", "directness": "direct", "hypothesis_ids": ["h1"],
             "scope": {"assay": "thallium flux"}},
        ]})
    await turn.call_tool("submit_grounded_answer", {
        "schema_version": "grounded-answer-v2",
        "answer_markdown": "ToxAgent reply: the flux result should be compared with patch clamp.",
        "claims": [], "limitations": [], "evidence_relations": [],
    })


async def test_a_study_runs_logs_everything_and_packs_blind(db, investigator, tmp_path):
    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        transport = httpx.ASGITransport(app=client.app)
        adapters = {
            "D_toxagent_investigator": ToxAgentAdapter("http://control.test", USER_TOKEN,
                                                       transport=transport, poll_delay_s=0.01,
                                                       run_timeout_s=30),
            # The shipped-product arm, pointed at a deployment with the case on:
            # it must refuse to record rather than mislabel the arm.
            "C_toxagent_current": ToxAgentAdapter("http://control.test", USER_TOKEN,
                                                  transport=transport, poll_delay_s=0.01,
                                                  run_timeout_s=30),
            "A_predictor_template": PredictorTemplateAdapter(),
        }
        manifest = await run_study(
            study_id="pilot-test", root=tmp_path,
            system_ids=["A_predictor_template", "C_toxagent_current", "D_toxagent_investigator"],
            case_ids=CASE_IDS, trials=1, toxagent_urls={}, snapshot_from="http://control.test",
            token=USER_TOKEN, transport=transport, adapters=adapters,
        )

    store = StudyStore(tmp_path / "pilot-test")
    records = store.latest_records()
    assert manifest["denominators"]["expected_records"] == 6
    assert manifest["denominators"]["by_system"]["C_toxagent_current"]["error"] == 2
    refused = records[(CASE_IDS[0], "C_toxagent_current", 1)]
    assert "deployment is not this arm" in refused.error

    investigator_record = records[(CASE_IDS[0], "D_toxagent_investigator", 1)]
    assert investigator_record.status == "ok", investigator_record.error
    assert len(investigator_record.turns) == 2
    assert investigator_record.model["skills_mode"] == "dynamic"
    assert investigator_record.turns[1].meta["skills"]["loaded"][0]["skill_id"] == \
        "assess-conflicting-evidence"
    trace = json.loads((store.root / investigator_record.artifacts["trace"]).read_text())
    assert trace["context_posts"] == [{"turn": 1, "case_revision": trace["context_posts"][0]["case_revision"]}]
    case = trace["cases"][0]["case"]
    assert case["context"][0]["key"] == "thallium_flux_result"
    assert [e["source_ref"] for e in case["evidence"]] == ["context:c1"]
    assert all(run["dossier"]["schema_version"] == "decision-dossier-v1" for run in trace["runs"])

    template = records[(CASE_IDS[1], "A_predictor_template", 1)]
    assert template.status == "ok" and "hERG channel blockade" in template.final_text
    snapshot_file = json.loads((store.root / "snapshots" / f"{CASE_IDS[1]}.json").read_text())
    assert snapshot_file["snapshot"]["predictions"]["herg"]["probability_blocker"] is not None

    assert manifest["systems"]["D_toxagent_investigator"]["adapter_config"]["flags"]["scientific_case_v1"]
    assert manifest["environment"]["git_commit"]

    result = packet_module.build_packet(study_dir=store.root, packet_id="lab-test", seed=1)
    packet_text = "\n".join(p.read_text() for p in Path(result["packet_dir"]).rglob("*.md"))
    assert "ToxAgent" not in packet_text
    assert "the flux result should be compared" in packet_text
    counts = result["responses_per_case"][CASE_IDS[0]]
    assert counts == {"expected": 3, "included": 2, "missing": 1}
