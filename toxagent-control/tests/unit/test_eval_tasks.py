"""The initial 50-task set is well-formed (plan section 16.2).

These do not run the tasks — they check the set is internally consistent and
matches the plan's category budget, so a task added later cannot silently break
the schema or the counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.build_tasks import main as build_tasks_main
from evals.frozen import FIXTURES_DIR
from evals.graders.hard_gates import _CHECKS
from evals.runner import TASKS_DIR, load_tasks

# Plan section 16.2.
EXPECTED_COUNTS = {
    "numeric_fidelity": 12,
    "endpoint_semantics": 8,
    "report_qa": 10,
    "evidence_synthesis": 8,
    "failure_recovery": 6,
    "adversarial_session": 6,
}
KNOWN_GRADERS = {"schema", "state", "transcript", "rubric", "sme"}


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    return load_tasks()


def test_there_are_exactly_fifty_tasks(tasks):
    assert len(tasks) == 50


def test_category_counts_match_the_plan(tasks):
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["category"]] = counts.get(task["category"], 0) + 1
    assert counts == EXPECTED_COUNTS


def test_every_task_validates_against_the_schema(tasks):
    # load_tasks() runs jsonschema when it is installed; assert it is, so CI
    # cannot silently downgrade to a structural check.
    import jsonschema  # noqa: F401

    load_tasks()  # raises on any invalid task


def test_task_ids_are_unique_and_kebab(tasks):
    ids = [t["task_id"] for t in tasks]
    assert len(ids) == len(set(ids))
    assert all(i.islower() and " " not in i for i in ids)


def test_every_referenced_fixture_exists(tasks):
    for task in tasks:
        assert (FIXTURES_DIR / f"{task['fixture']}.json").exists(), task["task_id"]


def test_every_hard_gate_is_implemented(tasks):
    for task in tasks:
        for gate in task.get("hard_gates", []):
            assert gate in _CHECKS, f"{task['task_id']} lists unimplemented gate {gate}"


def test_every_declared_grader_is_known(tasks):
    for task in tasks:
        for grader in task.get("graders", []):
            assert grader in KNOWN_GRADERS, f"{task['task_id']}: {grader}"


def test_all_ten_hard_gates_are_exercised_somewhere(tasks):
    used = {gate for task in tasks for gate in task.get("hard_gates", [])}
    assert used == set(_CHECKS), f"unused hard gates: {set(_CHECKS) - used}"


def test_both_languages_and_the_critical_subset_are_represented(tasks):
    languages = {t.get("language", "en") for t in tasks}
    assert languages == {"vi", "en"}
    critical = [t for t in tasks if t.get("critical")]
    assert len(critical) >= 10
    # Every category carries at least one critical task.
    assert {t["category"] for t in critical} == set(EXPECTED_COUNTS)


def test_build_tasks_is_idempotent():
    """Re-running the generator produces byte-identical files."""
    before = {p.name: p.read_text() for p in sorted(TASKS_DIR.glob("*.json"))}
    build_tasks_main()
    after = {p.name: p.read_text() for p in sorted(TASKS_DIR.glob("*.json"))}
    assert before == after
