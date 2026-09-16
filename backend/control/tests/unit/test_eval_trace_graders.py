"""EvalTraceV1 and the graders that read it (Wave 1).

Each grader is exercised against a hand-built TaskOutcome, the same frozen
snapshot shape every driver produces, so these hold for scripted, live and
replayed runs alike.
"""
from __future__ import annotations

from evals.graders import GRADER_REGISTRY, grade_task, grader_versions
from evals.graders.budget import grade_budget
from evals.graders.decision_state import grade_decision_state
from evals.graders.model import TaskOutcome
from evals.graders.outcome_split import breakdown, grade_outcome_split
from evals.graders.trajectory import grade_trajectory
from evals.trace import project


def _call(name: str, args: str = "a", status: str = "completed", error: str | None = None) -> dict:
    return {"tool_name": name, "status": status, "error_code": error,
            "arguments_sha256": args, "duration_ms": 3, "observation_ids": []}


def _outcome(calls=(), answer=None, state=None, budget=None) -> TaskOutcome:
    return TaskOutcome(
        run={"run_id": "run_1", "status": "completed", "intent": "decision_support",
             "lane": "agentic", "runtime": {"capability_profile": "decision_support"}},
        session={}, answer=answer, tool_calls=list(calls),
        decision_state=state, budget=budget,
    )


def _task(**expect) -> dict:
    return {"task_id": "t", "category": "decision_support", "expect": expect}


def test_the_trace_counts_duplicates_by_argument_hash_not_by_name():
    trace = project(_outcome([
        _call("search_toxicology_evidence", "q1"),
        _call("search_toxicology_evidence", "q2"),
        _call("search_toxicology_evidence", "q1"),
        _call("get_evidence_record", "r1"),
        _call("get_analysis_slice", "s", status="failed", error="tool_denied"),
    ]))
    assert trace.counters["searches"] == 3
    assert trace.counters["duplicate_calls"] == 1
    assert trace.counters["evidence_reads"] == 1
    assert trace.counters["denied_calls"] == 1
    assert trace.capability_profile == "decision_support"


def test_answer_outcome_distinguishes_first_pass_correction_and_fallback():
    assert project(_outcome(answer={"candidate_generation": 1})).answer_outcome == "first_pass"
    assert project(_outcome(answer={"candidate_generation": 2})).answer_outcome == (
        "accepted_after_correction"
    )
    assert project(_outcome(answer={"candidate_generation": 2, "is_fallback": True})).answer_outcome == (
        "fallback"
    )
    assert project(_outcome()).answer_outcome == "none"


def test_trajectory_requires_an_ordered_subsequence():
    calls = [_call("get_artifact_inventory"), _call("search_toxicology_evidence"),
             _call("submit_grounded_answer")]
    ok = _task(trajectory={"ordered_subsequence": ["get_artifact_inventory", "submit_grounded_answer"]})
    assert grade_trajectory(ok, _outcome(calls)).passed
    wrong = _task(trajectory={"ordered_subsequence": ["submit_grounded_answer", "get_artifact_inventory"]})
    assert not grade_trajectory(wrong, _outcome(calls)).passed
    floor = _task(trajectory={"min_searches": 2})
    assert not grade_trajectory(floor, _outcome(calls)).passed


def test_budget_grades_against_the_recorded_budget_and_refuses_an_unknown_one():
    calls = [_call("search_toxicology_evidence", str(i)) for i in range(5)] + [
        _call("submit_grounded_answer")
    ]
    assert not grade_budget(_task(), _outcome(calls)).passed  # nothing recorded
    within = _outcome(calls, budget={"max_tool_calls": 5, "max_searches": 5})
    assert grade_budget(_task(), within).passed  # the submit is exempt
    over = _outcome(calls, budget={"max_tool_calls": 24, "max_searches": 4})
    result = grade_budget(_task(), over)
    assert not result.passed and "searches=5" in result.reasons[0]


def test_a_fallback_is_never_an_agent_capability_pass():
    fallback = _outcome(answer={"candidate_generation": 2, "is_fallback": True})
    assert not grade_outcome_split(_task(), fallback).passed
    corrected = _outcome(answer={"candidate_generation": 2})
    assert grade_outcome_split(_task(), corrected).passed
    assert not grade_outcome_split(_task(outcome={"capability": "first_pass"}), corrected).passed
    facts = breakdown(fallback)
    assert facts["fallback_used"] and not facts["model_draft_valid_first_pass"]


def _state(**overrides) -> dict:
    state = {
        "goal": "g", "stop_reason": "sufficient", "revision": 2,
        "coverage": {"required": 2, "resolved": 2},
        "propositions": [
            {"id": "p1", "status": "supported", "artifact_refs": ["observation:obs_1"]},
            {"id": "p2", "status": "conflicted", "artifact_refs": ["evidence:evd_1"]},
        ],
    }
    state.update(overrides)
    return state


def test_decision_state_requires_a_stop_reason_and_grounded_resolutions():
    assert grade_decision_state(_task(), _outcome(state=_state())).passed
    assert not grade_decision_state(_task(), _outcome()).passed
    assert not grade_decision_state(_task(), _outcome(state=_state(stop_reason=None))).passed
    ungrounded = _state(propositions=[{"id": "p1", "status": "supported", "artifact_refs": []}])
    assert not grade_decision_state(_task(), _outcome(state=ungrounded)).passed
    premature = _state(propositions=[{"id": "p1", "status": "open", "artifact_refs": []}])
    assert not grade_decision_state(_task(), _outcome(state=premature)).passed
    expects_conflict = _task(decision_state={"statuses_include": ["conflicted"],
                                             "stop_reason_in": ["sufficient"]})
    assert grade_decision_state(expects_conflict, _outcome(state=_state())).passed


def test_required_graders_run_and_model_graders_are_deferred():
    task = {"task_id": "t", "category": "decision_support", "graders": ["schema"],
            "required_graders": ["outcome_split", "semantic"], "expect": {}}
    report = grade_task(task, _outcome(answer={"candidate_generation": 1, "claims": [],
                                               "limitations": [], "answer_markdown": ""}))
    assert {r.grader for r in report.results} >= {"run", "schema", "outcome_split"}
    assert report.deferred_graders == ("semantic",)


def test_every_grader_has_a_version():
    versions = grader_versions()
    assert set(versions) == set(GRADER_REGISTRY)
    assert all(versions.values())
