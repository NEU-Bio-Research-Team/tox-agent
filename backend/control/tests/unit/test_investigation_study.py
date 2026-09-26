"""The comparison study harness, offline (backlog Wave 5)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evals.investigation import cases as case_module
from evals.investigation import packet as packet_module
from evals.investigation import prompts
from evals.investigation import scorecard as scorecard_module
from evals.investigation.adapters import platform_cli
from evals.investigation.adapters.manual import ManualAdapter
from evals.investigation.adapters.predictor_template import render as render_template
from evals.investigation.adapters.toxagent import _usage, render_answer
from evals.investigation.record import RunRecord, StudyStore, TurnRecord
from evals.investigation.systems import SYSTEMS, product_mismatch
from toxagent.application.skill_catalog import load_catalog
from toxagent.config import PACKAGE_ROOT

pytestmark = pytest.mark.anyio

CASES = case_module.load_cases()
SNAPSHOT = {
    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O", "requested_endpoints": ["herg", "tox21"],
    "unavailable_endpoints": [],
    "predictions": {
        "herg": {"measurement": "hERG", "probability_blocker": 0.2813, "label": "non_blocker",
                 "threshold": 0.4133, "threshold_source": "artifact", "model_id": "m1"},
        "tox21": {"model_id": "m1", "task_order_version": "v1", "assays": {
            "NR-AR": {"probability_activity": 0.1, "active": False, "threshold": 0.5,
                      "threshold_source": "artifact"}}},
    },
    "applicability": {"status": "ok", "method": "element_rules_v1", "reasons": []},
}


# ------------------------------------------------------------------- cases

def test_the_committed_cases_are_the_specs_they_were_built_from():
    specs = {s["case_id"]: s for s in json.loads(case_module.SPECS_PATH.read_text())["cases"]}
    assert {c["case_id"] for c in CASES} == set(specs)
    for case in CASES:
        assert case["spec_sha256"] == case_module.sha256(specs[case["case_id"]])
        assert case["reference"]["status"] == "pending_lab_verification"


def test_an_anonymous_case_never_names_its_compound():
    anonymous = [c for c in CASES if not c["compound"]["name_disclosed"]]
    assert anonymous
    for case in anonymous:
        for turn in case["turns"]:
            assert case["compound"]["query_name"].lower() not in turn["text"].lower()


def test_building_refuses_a_turn_that_leaks_a_withheld_name():
    spec = {"case_id": "inv-99-x", "compound": "aspirin", "name_disclosed": False, "tags": ["t"],
            "turns": [{"text": "Is aspirin ({smiles}) ok?"}], "reference": {}}
    resolver = lambda name: {"pubchem_cid": 1, "smiles": "C", "molecular_formula": "C",  # noqa: E731
                             "source_url": "u"}
    with pytest.raises(ValueError, match="withheld"):
        case_module.build_case(spec, set_id="s", resolver=resolver, now="t")


def test_every_skill_has_positive_and_negative_cases_to_be_measured_on():
    tags = {tag for case in CASES for tag in case["tags"]}
    for skill in load_catalog(PACKAGE_ROOT / "agent_profiles").skills:
        # The investigation study measures decision_support skills; the report
        # builder's skills (W9-10) are measured by the TAB-Suite report packs.
        if "decision_support" not in skill.allowed_profiles:
            continue
        eval_set = skill.manifest["eval_set"]
        assert set(eval_set["positive_tags"]) & tags, skill.skill_id
        assert set(eval_set["negative_tags"]) & tags, skill.skill_id


# ----------------------------------------------------------------- prompts

def test_the_first_turn_is_the_preamble_and_the_question():
    case = CASES[0]
    text = prompts.render(case["turns"], 0, [])
    assert text.startswith(prompts.PREAMBLE)
    assert case["turns"][0]["text"] in text
    assert "ToxPred" not in text


def test_the_snapshot_arm_differs_only_by_the_snapshot():
    case = CASES[0]
    bare = prompts.render(case["turns"], 0, [])
    with_snapshot = prompts.render(case["turns"], 0, [], snapshot=SNAPSHOT)
    assert prompts.SNAPSHOT_HEADER in with_snapshot
    assert with_snapshot.replace(bare.split("\n\n", 1)[1], "") != with_snapshot
    assert '"probability_blocker": 0.2813' in with_snapshot


def test_a_later_turn_carries_the_whole_conversation():
    case = next(c for c in CASES if len(c["turns"]) > 1)
    text = prompts.render(case["turns"], 1, ["first answer"])
    assert f"Scientist: {case['turns'][0]['text']}" in text
    assert "You: first answer" in text
    assert text.endswith(case["turns"][1]["text"])


# ---------------------------------------------------------------- adapters

def test_the_template_states_values_and_caveats_but_never_a_verdict():
    text = render_template(SNAPSHOT)
    assert "0.281" in text and "non_blocker" in text and "NR-AR" in text
    assert "uncalibrated" in text and "not a safety assessment" in text


def test_a_toxagent_answer_renders_with_its_limitations_and_sources():
    answer = {"answer_markdown": "Body.", "claims": [{"citation_ids": ["evd_1"]}],
              "limitations": [{"code": "evidence_scope_limited", "text": ""}],
              "recommended_next_steps": [{"text": "Run patch clamp."}]}
    evidence = [{"evidence_id": "evd_1", "title": "A study", "authors": ["Doe J"],
                 "published_at": "2001-01-01", "identifier": {"doi": "10.1/x", "pmid": "1"}}]
    text = render_answer(answer, evidence)
    assert text.startswith("Body.")
    assert "- evidence_scope_limited" in text
    assert "- Run patch clamp." in text
    assert "[1] Doe J. A study. 2001. DOI 10.1/x, PMID 1" in text


def test_usage_counts_each_message_once_at_its_latest_revision():
    event = lambda rev, out: {"model_id": "gpt-x", "source": {"message_id": "m1", "revision": rev},  # noqa: E731
                              "tokens": {"input": 10, "output": out}, "cost": {"amount": "0.5", "currency": "usd"}}
    usage = _usage([{"run": {"usage": {"events": [event(1, 5), event(2, 9)]}}}])
    assert usage["tokens_output"] == 9 and usage["tokens_input"] == 10
    assert usage["cost_usd"] == 0.5
    assert usage["provider_reported_models"] == ["gpt-x"]


def test_a_toxagent_arm_refuses_a_deployment_with_the_wrong_flags():
    product = {"flags": {"scientific_case_v1": {"enabled": False},
                         "scientific_skills_v1": {"enabled": False},
                         "answer_draft_v2": {"enabled": True}},
               "scientific_skills": {"mode": "off"}}
    assert product_mismatch(SYSTEMS["C_toxagent_current"], product) == []
    problems = product_mismatch(SYSTEMS["D_toxagent_investigator"], product)
    assert any("scientific_case_v1" in p for p in problems)
    assert any("skills mode is off" in p for p in problems)


async def test_codex_output_records_the_model_it_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_cli, "codex_configuration",
                        lambda path=None: {"model": "gpt-test", "provider": "p"})

    async def fake(command, stdin, *, cwd, timeout=0):
        Path(command[command.index("--output-last-message") + 1]).write_text("The answer.")
        events = [{"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}}]
        return 0, "\n".join(json.dumps(e) for e in events), ""

    monkeypatch.setattr(platform_cli, "_run_process", fake)
    text, model, extra = await platform_cli.CodexCLIAdapter().ask("q", tmp_path)
    assert text == "The answer."
    assert model["model_id_resolved"] == "gpt-test"
    assert extra["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_the_codex_configuration_reader_never_reads_a_table(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('model = "gpt-test"\nprovider = "p"\n[providers.p]\nbase_url = "https://secret"\n'
                      'api_key = "sk-nope"\n')
    found = platform_cli.codex_configuration(config)
    assert found == {"model": "gpt-test", "provider": "p"}


async def test_claude_output_records_the_model_that_did_the_work(tmp_path, monkeypatch):
    async def fake(command, stdin, *, cwd, timeout=0):
        assert command[command.index("--tools") + 1] == ""
        assert "--strict-mcp-config" in command
        result = {"result": "An answer.", "is_error": False, "total_cost_usd": 0.01,
                  "usage": {"input_tokens": 5, "output_tokens": 50},
                  "modelUsage": {"claude-haiku-x": {"outputTokens": 60, "costUSD": 0.001},
                                 "claude-opus-x": {"outputTokens": 4, "costUSD": 0.009}}}
        return 0, json.dumps(result), ""

    monkeypatch.setattr(platform_cli, "_run_process", fake)
    text, model, extra = await platform_cli.ClaudeCLIAdapter().ask("q", tmp_path)
    assert text == "An answer."
    # A helper model with more output is not the one that answered.
    assert model["model_id_resolved"] == "claude-opus-x"
    _, requested, _ = await platform_cli.ClaudeCLIAdapter(model="haiku").ask("q", tmp_path)
    assert requested["model_id_resolved"] == "claude-haiku-x"
    assert extra["usage"]["cost_usd"] == 0.01


def test_among_ids_matching_the_alias_the_one_that_wrote_the_output_answered():
    # The 2026-09-25 pilot: opus-5-5 only wrote a prompt cache, opus-5 wrote the answer.
    usage = {"claude-haiku-4-5": {"outputTokens": 18, "costUSD": 0.001},
             "claude-opus-5-5": {"outputTokens": 0, "costUSD": 0.006},
             "claude-opus-5": {"outputTokens": 4464, "costUSD": 0.117}}
    assert platform_cli._main_model(usage, "opus") == "claude-opus-5"
    tie = {"claude-opus-5": {"outputTokens": 10, "costUSD": 0.01},
           "claude-opus-5-5": {"outputTokens": 10, "costUSD": 0.02}}
    assert platform_cli._main_model(tie, "opus") == "claude-opus-5-5"


async def test_a_manual_arm_waits_then_records_the_reported_model(tmp_path):
    store = StudyStore(tmp_path)
    case = next(c for c in CASES if len(c["turns"]) == 2)
    spec = SYSTEMS["P_google_bare"]
    first = await ManualAdapter().run(case, spec, trial=1, snapshot=None, store=store)
    assert first.status == "pending"
    directory = tmp_path / "manual" / spec.system_id / case["case_id"] / "t1"
    assert (directory / "turn0.prompt.md").read_text().startswith(prompts.PREAMBLE)
    (directory / "turn0.response.md").write_text("Answer one.")
    (directory / "turn0.meta.json").write_text(json.dumps({"model_id_resolved": "gemini-x"}))
    refused = await ManualAdapter().run(case, spec, trial=1, snapshot=None, store=store)
    assert refused.status == "error" and "answered_at" in refused.error
    (directory / "turn0.meta.json").write_text(json.dumps(
        {"model_id_resolved": "gemini-x", "answered_at": "t", "channel": "bridge"}))
    second = await ManualAdapter().run(case, spec, trial=1, snapshot=None, store=store)
    assert second.status == "pending"
    assert "You: Answer one." in (directory / "turn1.prompt.md").read_text()
    (directory / "turn1.response.md").write_text("Answer two.")
    (directory / "turn1.meta.json").write_text(json.dumps(
        {"model_id_resolved": "gemini-x", "answered_at": "t", "channel": "bridge"}))
    done = await ManualAdapter().run(case, spec, trial=1, snapshot=None, store=store)
    assert done.status == "ok" and done.final_text == "Answer two."
    assert done.model["model_id_resolved"] == "gemini-x"


# ------------------------------------------------------ packet & scorecard

def _record(case_id: str, system_id: str, text: str, status: str = "ok") -> RunRecord:
    return RunRecord(
        study_id="s", case_id=case_id, case_sha256="x", system_id=system_id,
        arm={"system_id": system_id}, trial=1, status=status, started_at="t", ended_at="t",
        model={"model_id_resolved": "m"},
        turns=[TurnRecord(index=0, user_text="q", sent_text="q", response_text=text)],
        final_text=text, error=None if status == "ok" else "boom",
    )


@pytest.fixture
def study(tmp_path) -> StudyStore:
    store = StudyStore(tmp_path / "study")
    chosen = [c["case_id"] for c in CASES[:3]]
    store.write_manifest({
        "study_id": "s", "trials": 1,
        "case_set": {"sha256": "abc", "cases": {c: "x" for c in chosen}},
        "systems": {"C_toxagent_current": {}, "P_openai_bare": {}},
    })
    for case_id in chosen:
        store.append(_record(case_id, "C_toxagent_current",
                             f"ToxAgent here; see obs_{'a' * 32}. Answer for {case_id}."))
        store.append(_record(case_id, "P_openai_bare", f"As ChatGPT (GPT-6), answer for {case_id}.",
                             status="error" if case_id == chosen[2] else "ok"))
    return store


def test_a_packet_hides_who_answered_and_keeps_the_key_outside(study):
    result = packet_module.build_packet(study_dir=study.root, packet_id="lab-1", seed=7)
    packet_dir = Path(result["packet_dir"])
    key_path = Path(result["key_path"])
    assert packet_dir not in key_path.parents
    everything = "\n".join(p.read_text() for p in packet_dir.rglob("*") if p.is_file())
    for leak in ("ToxAgent", "ChatGPT", "GPT-6", "C_toxagent_current", "P_openai_bare", "obs_"):
        assert leak not in everything, leak
    assert "[system]" in everything and "[id]" in everything
    key = json.loads(key_path.read_text())
    missing = [r for r in key["responses"] if not r["response_id"]]
    assert [(r["system_id"], r["missing_reason"]) for r in missing] == [("P_openai_bare", "error")]
    counts = json.loads((packet_dir / "packet-manifest.json").read_text())["responses_per_case"]
    assert sum(c["expected"] for c in counts.values()) == 6
    assert sum(c["missing"] for c in counts.values()) == 1
    rows = list(csv.DictReader((packet_dir / "grading_sheet.csv").open()))
    assert len(rows) == 5


def test_the_same_seed_gives_the_same_order_and_a_packet_is_never_overwritten(study):
    first = packet_module.build_packet(study_dir=study.root, packet_id="a", seed=3)
    second = packet_module.build_packet(study_dir=study.root, packet_id="b", seed=3)
    order = lambda r: [(x["case_id"], x["response_id"], x["system_id"])  # noqa: E731
                       for x in json.loads(Path(r["key_path"]).read_text())["responses"]]
    assert order(first) == order(second)
    with pytest.raises(SystemExit, match="never overwritten"):
        packet_module.build_packet(study_dir=study.root, packet_id="a", seed=3)


def _fill(sheet: Path, out: Path, grader: str, key: dict, score_for) -> None:
    identity = {(r["case_id"], r["response_id"]): r["system_id"] for r in key["responses"]
                if r["response_id"]}
    rows = list(csv.DictReader(sheet.open()))
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            system = identity[(row["case_id"], row["response_id"])]
            row["grader_id"] = grader
            for dim in ("question_framing", "evidence_accuracy", "uncertainty_and_gaps",
                        "scope_discipline", "usefulness"):
                row[dim] = str(score_for(system))
            row["conflict_handling"] = "NA"
            row["next_step_value"] = "2"
            for crit in ("fabricated_or_wrong_citation", "false_reassurance", "unsupported_verdict",
                         "invented_value", "attribution_as_mechanism", "endpoint_confusion",
                         "major_factual_error"):
                row[crit] = "1" if (crit == "invented_value" and system == "P_openai_bare") else "0"
            writer.writerow(row)


def test_the_scorecard_reports_dimensions_criticals_comparisons_and_agreement(study, tmp_path):
    result = packet_module.build_packet(study_dir=study.root, packet_id="lab-1", seed=7)
    key = json.loads(Path(result["key_path"]).read_text())
    sheet = Path(result["packet_dir"]) / "grading_sheet.csv"
    _fill(sheet, tmp_path / "a.csv", "A", key, lambda s: 3 if s == "C_toxagent_current" else 1)
    _fill(sheet, tmp_path / "b.csv", "B", key, lambda s: 2 if s == "C_toxagent_current" else 1)
    rubric = json.loads(packet_module.RUBRIC_PATH.read_text())
    grades = scorecard_module.read_grades([tmp_path / "a.csv", tmp_path / "b.csv"], rubric)
    card = scorecard_module.build_scorecard(
        key=key, grades=grades, rubric=rubric,
        compare=[("C_toxagent_current", "P_openai_bare")], reps=200,
    )
    c = card["systems"]["C_toxagent_current"]
    p = card["systems"]["P_openai_bare"]
    assert c["dimensions"]["usefulness"]["mean"] == 2.5
    assert c["dimensions"]["usefulness"]["cases_graded"] == 3
    assert c["dimensions"]["conflict_handling"]["na"] == 6
    assert p["responses_missing"] == 1 and p["responses_graded"] == 2
    assert p["critical_errors"]["invented_value"]["rate"] == 1.0
    assert c["any_critical_error"]["rate"] == 0.0
    diff = card["comparisons"][0]["dimensions"]["usefulness"]
    assert diff["shared_cases"] == 2 and diff["mean_difference"] == 1.5
    assert card["inter_rater"]["usefulness"]["cells"] == 5
    assert "total" not in json.dumps(card).replace("no total score", "")
    assert "| `C_toxagent_current` |" in scorecard_module.render_markdown(card)


def test_an_out_of_range_grade_is_refused(tmp_path):
    rubric = json.loads(packet_module.RUBRIC_PATH.read_text())
    columns = (["grader_id", "case_id", "response_id"] + [d["id"] for d in rubric["dimensions"]]
               + [c["id"] for c in rubric["critical_errors"]])
    path = tmp_path / "g.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({**{c: "" for c in columns}, "case_id": "c", "response_id": "R01",
                         "usefulness": "5"})
    with pytest.raises(scorecard_module.GradeError, match="must be 0-3"):
        scorecard_module.read_grades([path], rubric)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({**{c: "" for c in columns}, "case_id": "c", "response_id": "R01",
                         "usefulness": "NA"})
    with pytest.raises(scorecard_module.GradeError, match="does not allow NA"):
        scorecard_module.read_grades([path], rubric)


def test_skill_triggers_are_read_against_each_skills_declared_case_tags():
    from evals.investigation.report import skill_triggers

    by_id = {c["case_id"]: c for c in CASES}
    conflict = next(c for c in CASES if "conflicting_evidence" in c["tags"])["case_id"]
    lookup = next(c for c in CASES if "numeric_lookup" in c["tags"])["case_id"]
    skill = load_catalog(PACKAGE_ROOT / "agent_profiles").get("assess-conflicting-evidence")

    def record(case_id: str, loaded: list[str]) -> RunRecord:
        rec = _record(case_id, "D_toxagent_investigator", "text")
        rec.turns[0].meta = {"skills": {"loaded": [{"skill_id": s} for s in loaded]}}
        return rec

    result = skill_triggers([record(conflict, []), record(lookup, [skill.skill_id])], by_id, [skill])
    entry = result[skill.skill_id]
    assert entry["false_triggers_on_negative_cases"] == [lookup]
    assert entry["misses"] == [conflict]
    assert entry["trigger_precision"] == 0.0 and entry["trigger_recall"] == 0.0


def test_a_platform_refusal_is_told_apart_from_quota_and_transport_failures():
    from evals.investigation.report import error_class

    assert error_class("claude exited 1: API Error: Opus 5's safeguards flagged this message "
                       "(https://www.anthropic.com/legal/aup).") == "provider_refusal"
    assert error_class("claude exited 1: You've hit your session limit") == "quota_or_capacity"
    assert error_class("gemini: HTTP 429 RESOURCE_EXHAUSTED") == "quota_or_capacity"
    assert error_class("ConnectError: connection refused") == "other"
    assert error_class(None) == "other"


async def test_the_gemini_bridge_retries_overload_and_records_the_reported_model(monkeypatch):
    adapter = platform_cli.GeminiMCPAdapter(["python3", "server.py"], {"X": "1"}, pause_s=0)
    calls = []

    async def fake_call(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("Gemini API failed with HTTP 503: high demand")
        return {"response": "An answer.", "model": "gemini-x-flash", "backend": "api", "duration_ms": 1200}

    monkeypatch.setattr(adapter, "_call", fake_call)
    text, model, extra = await adapter.ask("q", Path("."))
    assert text == "An answer." and model["model_id_resolved"] == "gemini-x-flash"
    assert len(extra["retried_errors"]) == 1 and len(calls) == 2

    async def quota(prompt):
        raise RuntimeError("HTTP 404: model not found")

    monkeypatch.setattr(adapter, "_call", quota)
    with pytest.raises(RuntimeError, match="404"):
        await adapter.ask("q", Path("."))


def _append_many(root: str, worker: int) -> None:
    store = StudyStore(Path(root))
    for index in range(25):
        store.append(_record(f"case-{worker}-{index}", "P_openai_bare", "x" * 20_000))


def test_several_processes_can_append_to_one_study_without_interleaving(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(_append_many, [str(tmp_path)] * 4, range(4)))
    records = StudyStore(tmp_path).records()  # every line parses
    assert len(records) == 100
    assert all(len(r.final_text) == 20_000 for r in records)


def test_a_manifest_update_merges_instead_of_overwriting(tmp_path):
    store = StudyStore(tmp_path)
    store.update_manifest(lambda m: {**m, "systems": {**m.get("systems", {}), "A": {}}})
    store.update_manifest(lambda m: {**m, "systems": {**m.get("systems", {}), "B": {}}})
    assert set(store.manifest()["systems"]) == {"A", "B"}
