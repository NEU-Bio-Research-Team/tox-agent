"""Paired cutover comparison (Wave 4) and the single-control-plane guard."""
from __future__ import annotations

import ast
from pathlib import Path

from evals.paired import compare, mcnemar_exact


def _manifest(orchestrator: bool, **extra) -> dict:
    product = {"flags": {"report_orchestrator_v2": {"enabled": orchestrator}},
               "intents": {"build_report": {"lane": "orchestrated" if orchestrator else "agentic"}},
               "topology": {"external_worker_mode": False}, "effective_product_hash": str(orchestrator)}
    product.update(extra)
    return {"eval_suite_hash": "h", "fixture_mode": "predictor_integration", "trial_count": 3,
            "runtime_kind": "opencode", "effective_product": product}


def _row(task_id, status, critical=False, outcomes=()):
    return {"task_id": task_id, "status": status, "critical": critical,
            "trials": [{"answer_outcome": o} for o in outcomes]}


def test_a_clean_cutover_is_supported_and_counts_transitions():
    baseline = [_row("a", "fail"), _row("b", "pass"), _row("c", "skipped"), _row("d", "fail")]
    candidate = [_row("a", "pass"), _row("b", "pass"), _row("c", "pass"), _row("d", "pass")]
    report = compare(_manifest(False), baseline, _manifest(True), candidate,
                     expected_change=["flags.report_orchestrator_v2", "intents.build_report"])
    assert report["counts"]["fixes"] == 2 and report["counts"]["regressions"] == 0
    assert report["transitions"]["not_paired"] == ["c"]
    assert report["confounds"] == []
    assert report["cutover_supported"] is True


def test_a_critical_regression_or_a_confound_blocks_cutover():
    baseline = [_row("a", "pass", critical=True), _row("b", "fail")]
    candidate = [_row("a", "fail", critical=True), _row("b", "pass")]
    report = compare(_manifest(False), baseline, _manifest(True), candidate,
                     expected_change=["flags.report_orchestrator_v2", "intents.build_report"])
    assert report["critical_regressions"] == ["a"]
    assert report["cutover_supported"] is False

    confounded = compare(
        _manifest(False), [_row("b", "fail")],
        _manifest(True, topology={"external_worker_mode": True}), [_row("b", "pass")],
        expected_change=["flags.report_orchestrator_v2", "intents.build_report"],
    )
    assert confounded["confounds"] == ["topology.external_worker_mode"]
    assert confounded["cutover_supported"] is False


def test_different_suites_are_not_comparable():
    other = {**_manifest(True), "eval_suite_hash": "other"}
    report = compare(_manifest(False), [], other, [], expected_change=["flags", "intents"])
    assert report["comparability"] and not report["cutover_supported"]


def test_mcnemar_is_exact_and_symmetric():
    assert mcnemar_exact(0, 0) is None
    assert mcnemar_exact(0, 6) == mcnemar_exact(6, 0) == 0.03125
    assert mcnemar_exact(3, 3) == 1.0


def test_no_live_module_imports_the_superseded_kernel():
    """ADR 0011: AgentRuntimeGateway + DecisionSupportStateV1 is the one live
    control plane. The dormant kernel may be read by its own persistence
    adapters until its tables are retired, and by nothing that runs a turn."""
    root = Path(__file__).resolve().parents[2] / "src" / "toxagent"
    allowed = {
        "persistence/investigations.py", "persistence/interfaces.py",
        "persistence/sql/repositories/investigation.py",
    }
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in allowed or rel.startswith("superseded/"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and (
                "superseded" in node.module.split(".")
            ):
                offenders.append(rel)
    assert not offenders, offenders


def test_every_conformance_drill_names_tests_that_exist():
    from evals.conformance import DRILLS, SERVICE_ROOT

    for name, drill in DRILLS.items():
        for node in drill["tests"]:
            path, _, test = node.partition("::")
            source = (SERVICE_ROOT / path).read_text()
            assert f"def {test}(" in source, (name, node)
        if not drill["tests"]:
            assert drill.get("requires"), name
