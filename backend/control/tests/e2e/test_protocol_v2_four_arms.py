"""#50: the four arms of protocol v2 on one case set's frozen inputs.

On the technical set (synthetic, machinery only), the runner puts a
toxicity-only case, an activity case and the multi-turn case that adds a
measurement and switches target to A, B, C and D -- one deployment per arm,
run one after another the way a laptop runs them -- and the study holds:

* no arm calls the predictor; A reads the set's pinned replay file, B/C/D
  are bound to the same frozen artifacts, and the content hashes every arm
  was bound to are the ones the inputs declare, identical across arms;
* every arm is sent the same text for every turn, the researcher's context
  rendered into it; D also files that context, typed, into the subject's case;
* switching target opens a second case on D, and the first keeps its history;
* each arm's effective product is kept and later arms are held to it: a
  deployment on another effective budget is refused even under the same
  preset name, as is one replaying another file;
* a (case, arm, trial) is attempted once as its primary; rerunning adds a
  diagnostic attempt in its own raw directory and never replaces the primary.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from evals.investigation.adapters.toxagent_replay import ToxAgentReplayAdapter
from evals.investigation.record import StudyStore
from evals.investigation.run import StudyManifestMismatch, run_protocol_study
from tests.e2e.test_scripted_runtime import _install_scripted_runtime
from tests.support.activity import replay_file
from tests.support.api import USER_TOKEN, api_client, settings
from tests.support.predictor import StubPredictor
from toxagent.platform.config import ActivitySettings, PolicySettings, ResearchSettings

pytestmark = pytest.mark.anyio

SET = Path(__file__).resolve().parents[2] / "evals" / "investigation" / "cases_protocol_v2_technical"
CASES = ["inv-t01-herg-score-explained", "inv-a01-screening-review-explained",
         "inv-a05-update-then-switch-target"]
_BUNDLE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
_MATERIAL = re.compile(r"Material:\n```json\n(.*?)\n```", re.DOTALL)

COMMON = {"ANSWER_DRAFT_V2", "EVIDENCE_FULL_TEXT_V1", "ACTIVITY_REPLAY_V1", "CLAIM_REVIEWER_V1",
          "RELEASE_GATE_V1", "TYPED_STOP_V1", "SUBJECT_BINDING_V1"}
ARM_FLAGS = {
    "B2_fixed_workflow": COMMON | {"FIXED_WORKFLOW_V1"},
    "C2_generic_react": COMMON,
    "D2_investigator": COMMON | {"SCIENTIFIC_CASE_V1", "SCIENTIFIC_SKILLS_V1"},
}


def _config(**overrides):
    manifest = json.loads((SET / "seed-manifest.json").read_text())
    return settings(**{
        "activity": ActivitySettings(replay_path=str(SET / "replay.json"),
                                     replay_sha256=manifest["replay"]["sha256"]),
        "research": ResearchSettings(provider="snapshot",
                                     snapshot_path=str(SET / "evidence.json")),
        **overrides,
    })


def _owed(view) -> list[dict]:
    return [{"code": c, "text": ""} for c in sorted(_codes(view))]


def _codes(value) -> set[str]:
    """Every required-limitation code anywhere in a tool's model view."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "required_limitations" and isinstance(item, list):
                found.update(str(c) for c in item)
            else:
                found |= _codes(item)
    elif isinstance(value, list):
        for item in value:
            found |= _codes(item)
    return found


def _answer(limitations: list[dict]) -> dict:
    return {"schema_version": "grounded-answer-v2",
            "answer_markdown": "The frozen artifacts were read; see the limitations.",
            "claims": [], "limitations": limitations, "evidence_relations": []}


def _script(seen: dict):
    async def script(turn) -> None:
        profile = turn.tool_context.profile
        seen.setdefault("profiles", []).append(profile)
        if profile == "claim_review":
            bundle = json.loads(_BUNDLE.search(turn.system_prompt).group(1))
            await turn.call_tool("submit_claim_review", {"reviews": [
                {"claim_id": c["claim_id"], "verdict": "supported", "reason": "checked"}
                for c in bundle["claims"]]})
            return
        if profile == "workflow_synthesis":
            material = json.loads(_MATERIAL.search(turn.system_prompt).group(1))
            await turn.call_tool("submit_grounded_answer", _answer(_owed(material)))
            return
        views = []
        inventory = (await turn.call_tool("get_artifact_inventory", {}))["model_view"]
        views.append(inventory)
        subject = inventory["subject"]
        if seen.get("keeps_case"):
            views.append((await turn.call_tool("get_scientific_case", {}))["model_view"])
        if subject["kind"] == "activity_screening":
            sid = subject["subject_ref"]["screening_id"]
            for tool in ("get_screening_state", "get_activity_snapshot",
                         "get_prioritisation_decision"):
                read = await turn.call_tool(tool, {"screening_id": sid})
                views.append(read.get("model_view") or {})
        else:
            read = await turn.call_tool(
                "get_analysis_bundle", {"analysis_id": subject["subject_ref"]["analysis_id"]})
            views.append(read.get("model_view") or {})
        seen.setdefault("prompts", []).append(turn.system_prompt)
        await turn.call_tool("submit_grounded_answer", _answer(_owed(views)))

    return script


def _arm_env(monkeypatch, system_id: str) -> None:
    for name in {f for flags in ARM_FLAGS.values() for f in flags}:
        monkeypatch.delenv(f"TOXAGENT_FLAG_{name}", raising=False)
    for name in ARM_FLAGS[system_id]:
        monkeypatch.setenv(f"TOXAGENT_FLAG_{name}", "1")
    monkeypatch.setenv("TOXAGENT_RUN_BUDGET_PRESET", "investigator-pilot-v2")
    monkeypatch.setenv("TOXAGENT_RELEASE_POLICY", "strict-reviewed-v1")


async def _run_arm(db, monkeypatch, tmp_path, system_id: str, *, predictor, seen,
                   config=None, case_ids=CASES):
    _arm_env(monkeypatch, system_id)
    async with api_client(db, predictor, config=config or _config()) as client:
        await _install_scripted_runtime(client, _script(seen))
        adapter = ToxAgentReplayAdapter(
            "http://control.test", USER_TOKEN, transport=httpx.ASGITransport(app=client.app),
            poll_delay_s=0.01, run_timeout_s=60)
        return await run_protocol_study(
            study_id="v2", root=tmp_path, case_set_dir=SET, system_ids=[system_id],
            case_ids=case_ids, adapters={system_id: adapter})


async def test_four_arms_share_inputs_and_the_study_holds_them_to_it(db, monkeypatch, tmp_path):
    predictor = StubPredictor()
    seen: dict[str, dict] = {}
    for system_id in ("B2_fixed_workflow", "C2_generic_react", "D2_investigator"):
        seen[system_id] = {"keeps_case": system_id == "D2_investigator"}
        await _run_arm(db, monkeypatch, tmp_path, system_id, predictor=predictor,
                       seen=seen[system_id])
    manifest = await run_protocol_study(study_id="v2", root=tmp_path, case_set_dir=SET,
                                        system_ids=["A2_template"], case_ids=CASES)

    store = StudyStore(tmp_path / "v2")
    primary = store.primary_records()
    assert predictor.requests == []
    assert manifest["denominators"]["expected_records"] == 12
    for system_id in ("A2_template", "B2_fixed_workflow", "C2_generic_react", "D2_investigator"):
        row = manifest["denominators"]["by_system"][system_id]
        assert row == {"ok": 3, "error": 0, "not_run": 0, "diagnostic_reruns": 0}, (system_id, row)

    for case_id in CASES:
        observed = {sid: primary[(case_id, sid, 1)].toxagent["observed_source_sha256"]
                    for sid in ("A2_template", "B2_fixed_workflow", "C2_generic_react",
                                "D2_investigator")}
        subject_keys = [{k: v for k, v in o.items() if not k.startswith("toxicity_snapshot")}
                        for o in observed.values()]
        assert all(keys == subject_keys[0] for keys in subject_keys), (case_id, observed)
        sent = {sid: [t.sent_text for t in primary[(case_id, sid, 1)].turns]
                for sid in ("B2_fixed_workflow", "C2_generic_react", "D2_investigator")}
        assert len({json.dumps(v) for v in sent.values()}) == 1, sent

    multi = primary[(CASES[2], "D2_investigator", 1)]
    assert "measured_pIC50_EGFR = 6.2 pIC50" in multi.turns[1].sent_text
    assert "target EGFR" in multi.turns[1].sent_text
    assert [t.meta["subject"].split(":")[0] for t in multi.turns] == ["screening"] * 3
    assert multi.turns[0].meta["subject"] == multi.turns[1].meta["subject"] \
        != multi.turns[2].meta["subject"]
    trace = json.loads((store.root / multi.artifacts["trace"]).read_text())
    (post,) = trace["context_posts"]
    assert post["status"] == "filed" and post["turn"] == 1
    cases = {c["case"]["subject"]["identity"]["target"]["target_id"]: c["case"]
             for c in trace["cases"]}
    assert set(cases) == {"EGFR", "KCNH2"}
    (context,) = cases["EGFR"]["context"]
    assert context["unit"] == "pIC50" and context["scope"]["target_id"] == "EGFR"
    assert context["source"] == "researcher"
    assert cases["KCNH2"]["context"] == []
    # C keeps no case, so it has nowhere to file the context; it reads it in
    # the message like B does.
    c_trace = json.loads((store.root / primary[(CASES[2], "C2_generic_react", 1)]
                          .artifacts["trace"]).read_text())
    assert c_trace["context_posts"] == [] and c_trace["cases"] == []

    assert primary[(CASES[0], "D2_investigator", 1)].artifacts["trace"].split("/")[-2] \
        .startswith("p-")
    assert sorted(p.stem for p in (store.root / "products").glob("*.json")) == [
        "B2_fixed_workflow", "C2_generic_react", "D2_investigator"]
    order = manifest["protocol_v2"]["execution_order"]
    assert len(order) == 12 and manifest["protocol_v2"]["order_seed"]
    assert all(r.attempt_kind == "primary" for r in store.records())


async def test_an_arm_on_another_effective_budget_is_refused(db, monkeypatch, tmp_path):
    predictor = StubPredictor()
    await _run_arm(db, monkeypatch, tmp_path, "C2_generic_react", predictor=predictor,
                   seen={}, case_ids=CASES[:1])
    # The same preset label, a longer deadline and more calls: another budget.
    longer = _config(policy=PolicySettings(run_deadline_s=900, max_tool_calls_per_run=50))
    with pytest.raises(ValueError, match="budget differs across arms"):
        await _run_arm(db, monkeypatch, tmp_path, "D2_investigator", predictor=predictor,
                       seen={"keeps_case": True}, config=longer, case_ids=CASES[:1])
    # And a deployment replaying another file is not running this set.
    other_path, other_sha = replay_file(tmp_path)
    other = _config(activity=ActivitySettings(replay_path=str(other_path),
                                              replay_sha256=other_sha))
    with pytest.raises(ValueError, match="do not replay this case set"):
        await _run_arm(db, monkeypatch, tmp_path, "D2_investigator", predictor=predictor,
                       seen={"keeps_case": True}, config=other, case_ids=CASES[:1])


async def test_a_repeat_invocation_skips_the_primary_and_a_changed_study_is_refused(tmp_path):
    first = await run_protocol_study(study_id="v2", root=tmp_path, case_set_dir=SET,
                                     system_ids=["A2_template"], case_ids=CASES[:1])
    again = await run_protocol_study(study_id="v2", root=tmp_path, case_set_dir=SET,
                                     system_ids=["A2_template"], case_ids=CASES[:1])
    store = StudyStore(tmp_path / "v2")
    assert len(store.records()) == 1  # attempted once; the second invocation skips it
    assert first["denominators"] == again["denominators"]
    with pytest.raises(StudyManifestMismatch, match="order_seed"):
        await run_protocol_study(study_id="v2", root=tmp_path, case_set_dir=SET,
                                 system_ids=["A2_template"], case_ids=CASES[:1], order_seed=7)
