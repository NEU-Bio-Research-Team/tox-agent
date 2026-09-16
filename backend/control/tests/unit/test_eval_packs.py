"""Declared task packs, the v1 -> v3 loader, conservation and manifest v2.

The review that motivated these (2026-09-16, P0-01..P0-03) found a regression
task that existed in the repository, changed no suite hash and ran in no suite.
Each test below pins one way that could happen again.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from evals import task_packs
from evals.graders.model import TaskOutcome
from evals.manifest import release_blockers
from evals.runner import (
    SKIPPED_REASONS,
    exit_code,
    features_satisfied,
    infra_failure,
    load_tasks,
)

ADS_00 = "ads-00-benzene-drug-decision-baseline-vi"


def _manifest(tmp: Path, name: str, **extra) -> task_packs.PackManifest:
    return task_packs.PackManifest(
        name=name, owner="test", description="", task_dirs=(tmp,), suites=("pr",),
        path=None, **extra,
    )


# ---------------------------------------------------------------- discovery

def test_the_default_selection_includes_the_regression_pack():
    assert task_packs.DEFAULT_PACKS == ("core", "regression")
    discovery = task_packs.discover()
    ids = {t["task_id"] for t in discovery.tasks}
    assert ADS_00 in ids
    assert discovery.invalid == []
    assert discovery.discovered == len(discovery.tasks)


def test_selecting_regression_changes_the_suite_hash():
    core_only = task_packs.suite_hash(task_packs.discover(("core",)))
    with_regression = task_packs.suite_hash(task_packs.discover(("core", "regression")))
    assert core_only != with_regression


def test_the_suite_hash_covers_grader_code():
    sources = {p.name for p in task_packs.GRADERS_DIR.glob("*.py")}
    assert {"hard_gates.py", "trajectory.py", "budget.py"} <= sources


def test_every_declared_pack_has_an_owner_and_a_location():
    manifests = task_packs.load_pack_manifests()
    assert set(manifests) >= {"core", "regression", "security", "report", "ocr", "sealed"}
    for manifest in manifests.values():
        assert manifest.owner
        if manifest.external:
            assert manifest.external_dir_env and not manifest.task_dirs
        else:
            assert all(d.is_dir() for d in manifest.task_dirs), manifest.name


def test_a_broken_file_in_a_declared_pack_is_invalid_not_absent(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "wrong.json").write_text(json.dumps({"task_id": "x", "schema_version": "eval-task-v1"}))
    manifests = {"scratch": _manifest(tmp_path, "scratch")}
    discovery = task_packs.discover(("scratch",), manifests)
    assert discovery.discovered == 2
    assert len(discovery.invalid) == 2
    assert discovery.tasks == []


def test_a_duplicate_task_id_across_packs_is_invalid(tmp_path):
    core = json.loads((task_packs.HERE / "tasks" / "qa-09-out-of-scope-clinical-advice.json").read_text())
    (tmp_path / "dup.json").write_text(json.dumps(core))
    manifests = task_packs.load_pack_manifests()
    manifests["scratch"] = _manifest(tmp_path, "scratch")
    discovery = task_packs.discover(("core", "scratch"), manifests)
    assert len(discovery.invalid) == 1
    assert "also declared" in discovery.invalid[0].problems[0]


def test_parse_packs_expands_suites_and_rejects_unknown_names():
    assert "regression" in task_packs.parse_packs("suite:pr")
    assert "sealed" in task_packs.parse_packs("suite:release")
    with pytest.raises(SystemExit):
        task_packs.parse_packs("core,nonexistent")


# ------------------------------------------------------------------- sealed

def test_an_absent_sealed_directory_is_unavailable_never_passed(monkeypatch):
    monkeypatch.delenv("TOXAGENT_SEALED_TASKS_DIR", raising=False)
    discovery = task_packs.discover(("sealed",))
    assert discovery.packs["sealed"].available is False
    assert discovery.tasks == []


def test_a_sealed_task_must_carry_a_declared_sealed_id(tmp_path, monkeypatch):
    task = _v3_task("sealed-probe", pack="sealed", sealed_id="sealed-undeclared")
    (tmp_path / "t.json").write_text(json.dumps(task))
    monkeypatch.setenv("TOXAGENT_SEALED_TASKS_DIR", str(tmp_path))
    manifests = task_packs.load_pack_manifests()
    discovery = task_packs.discover(("sealed",), manifests)
    assert discovery.packs["sealed"].available
    assert "not declared" in discovery.invalid[0].problems[0]
    # The on-disk location of sealed content is never recorded.
    assert discovery.invalid[0].path.startswith("<external>/")


# ------------------------------------------------------------------ adapter

def _v3_task(task_id: str, *, pack: str = "regression", **extra) -> dict:
    task = {
        "task_id": task_id, "schema_version": "eval-task-v3", "category": "decision_support",
        "title": "t", "fixture": "herg-blocker", "capability_pack": pack,
        "risk_tier": "high", "runtime_requirement": "agentic_runtime",
        "conversation": [{"role": "user", "content": "hi"}], "expect": {},
    }
    task.update(extra)
    return task


def test_v3_schema_is_a_superset_of_v1():
    v1 = json.loads(task_packs.SCHEMAS["eval-task-v1"].read_text())
    v3 = json.loads(task_packs.SCHEMAS["eval-task-v3"].read_text())
    for name, spec in v1["properties"].items():
        assert name in v3["properties"], name
        if name in ("schema_version", "category", "expect", "graders"):
            continue
        assert v3["properties"][name] == spec, name
    assert set(v1["properties"]["category"]["enum"]) <= set(v3["properties"]["category"]["enum"])


def test_every_v1_task_adapts_without_losing_a_field():
    for raw in load_tasks():
        adapted = task_packs.adapt_v1(raw, "core")
        for key, value in raw.items():
            assert adapted[key] == value
        assert adapted["capability_pack"] == "core"
        assert adapted["runtime_requirement"] in ("scripted", "agentic_runtime", "process_control")
        if raw.get("critical"):
            assert adapted["trial_policy"]["aggregation"] == "worst_of_n"


def test_a_v3_task_validates(tmp_path):
    assert task_packs.validate_task(_v3_task("probe")) == []
    bad = _v3_task("probe", risk_tier="apocalyptic")
    assert task_packs.validate_task(bad)


# ------------------------------------------------------------- conservation

def test_conservation_catches_a_silently_dropped_task():
    discovery = task_packs.discover()
    n = discovery.discovered
    assert task_packs.check_conservation(discovery, executed=n, skipped=0, invalid=0) == []
    assert task_packs.check_conservation(discovery, executed=n - 1, skipped=0, invalid=0)


def test_every_skip_reason_is_typed():
    assert {"not_selected", "feature_requirements_unmet", "needs_live_evidence"} <= set(SKIPPED_REASONS)


def test_invalid_or_infra_error_is_never_a_clean_exit():
    base = {"invalid": 0, "conservation_violations": [], "infra_error": 0,
            "critical_all_pass": True, "pass_rate": 1.0}
    assert exit_code(base) == 0
    assert exit_code({**base, "invalid": 1}) == 1
    assert exit_code({**base, "infra_error": 1}) == 1
    assert exit_code({**base, "conservation_violations": ["x"]}) == 1


# ------------------------------------------------------- features and infra

def _product(**flags: bool) -> dict:
    return {
        "flags": {name: {"enabled": value} for name, value in flags.items()},
        "topology": {"external_worker_mode": False},
        "runtime": {"kind": "opencode", "provider_id": "p"},
        "providers": {"research_provider": "europepmc", "ocr_configured": False},
        "intents": {"build_report": {"capability_profile": "report_synthesis"}},
    }


def test_a_task_is_skipped_when_the_product_is_not_the_one_it_was_written_for():
    task = _v3_task("probe", feature_requirements={"flags": {"report_orchestrator_v2": True}})
    assert features_satisfied(task, _product(report_orchestrator_v2=True))
    assert not features_satisfied(task, _product(report_orchestrator_v2=False))
    assert not features_satisfied(task, {"unavailable": "no endpoint"})
    assert features_satisfied(_v3_task("plain"), None)
    ocr = _v3_task("probe", feature_requirements={"ocr": True})
    assert not features_satisfied(ocr, _product())


def test_an_unexpected_infra_failure_is_not_a_product_failure():
    outcome = TaskOutcome(run={"status": "failed", "failure_code": "runtime_unavailable"}, session={})
    assert infra_failure(_v3_task("probe"), outcome)
    expected = _v3_task("probe", expect={"error_code": "runtime_unavailable"})
    assert infra_failure(expected, outcome) is None
    product = TaskOutcome(run={"status": "failed", "failure_code": "answer_rejected"}, session={})
    assert infra_failure(_v3_task("probe"), product) is None


# ----------------------------------------------------------------- manifest

def _settings(**policy):
    from toxagent.config import (
        CompoundSettings, OcrSettings, PolicySettings, PredictorSettings, PredictSettings,
        ResearchSettings, RuntimeSettings, SecuritySettings, Settings,
    )

    return Settings(
        database_url="sqlite+aiosqlite:///x.db",
        predictor=PredictorSettings(base_url="http://user:hunter2@predictor.internal:8080"),
        policy=PolicySettings(**policy), predict=PredictSettings(), runtime=RuntimeSettings(),
        research=ResearchSettings(), compound=CompoundSettings(), ocr=OcrSettings(),
        security=SecuritySettings(capability_secret="topsecret-capability"),
    )


def test_the_effective_product_carries_no_secret():
    from toxagent.application.effective_product import describe_effective_product

    document = json.dumps(describe_effective_product(_settings()))
    assert "hunter2" not in document
    assert "topsecret-capability" not in document
    assert "predictor.internal" in document


def test_a_flag_changes_the_lane_and_the_product_hash(monkeypatch):
    from toxagent.application.effective_product import describe_effective_product

    off = describe_effective_product(_settings())
    monkeypatch.setenv("TOXAGENT_FLAG_REPORT_ORCHESTRATOR_V2", "1")
    on = describe_effective_product(_settings())
    assert off["intents"]["build_report"]["lane"] == "agentic"
    assert on["intents"]["build_report"]["lane"] == "orchestrated"
    assert on["intents"]["build_report"]["capability_profile"] == "report_synthesis"
    assert on["flags"]["report_orchestrator_v2"]["overridden"] is True
    assert off["effective_product_hash"] != on["effective_product_hash"]


def test_the_budget_in_the_product_is_the_enforced_one():
    from toxagent.application.effective_product import describe_effective_product
    from toxagent.tools.definitions.evidence import DECISION_SUPPORT_MAX_SEARCHES_PER_RUN

    document = describe_effective_product(_settings(max_tool_calls_per_run=17))
    budget = document["intents"]["decision_support"]["budget"]
    assert budget["max_tool_calls"] == 17
    assert budget["max_searches"] == DECISION_SUPPORT_MAX_SEARCHES_PER_RUN


def test_expired_flags_are_visible():
    from toxagent.application.effective_product import flag_snapshot

    snapshot = flag_snapshot(today=date(2027, 1, 1))
    assert all(flag["expired"] for flag in snapshot.values())


def test_release_blockers_name_every_gap():
    clean = {
        "runtime_kind": "opencode",
        "source": {"commit": "abc", "dirty_worktree": False},
        "effective_product": {k: {} for k in ("flags", "intents", "runtime", "topology", "hashes")},
        "summary": {"invalid": 0, "infra_error": 0, "conservation_violations": [],
                    "not_evaluated_packs": []},
    }
    assert release_blockers(clean) == []
    dirty = json.loads(json.dumps(clean))
    dirty["source"]["dirty_worktree"] = True
    dirty["effective_product"] = {"unavailable": "HTTP 404"}
    dirty["summary"]["not_evaluated_packs"] = ["sealed"]
    blockers = release_blockers(dirty)
    assert len(blockers) == 3
