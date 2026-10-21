"""#46: a run says why it stopped, in a type, and runs under a named budget.

Two fixes the live runs of 2026-10-21 asked for. A run that read a replayed
screening and answered from it ended "insufficient_evidence" -- the reason a
run gets when its sources did not settle the question, which is the opposite
of what happened. And a stop only the answer can know -- that it is waiting on
the researcher -- had no way to be said at all.
"""
from __future__ import annotations

import pytest

from toxagent.domain import decision_state as ds
from toxagent.validation.answer.draft_wire import (
    GroundedAnswerDraftV2,
    GroundedAnswerDraftV2SpansStop,
    StopDeclarationV2,
    draft_model,
)


def _answered(**kw) -> ds.DecisionSupportStateV1:
    state = ds.DecisionSupportStateV1(session_id="ses_x", run_id="run_x", goal="g",
                                      subject_refs=())
    return ds.apply_answer(state, relations=(), is_fallback=False, candidate_generation=1, **kw)


def _stop(state) -> str:
    return ds.finalize(state, run_status="completed").stop_reason


def test_an_answer_read_from_the_runs_own_artifacts_is_a_lookup():
    assert _stop(_answered(basis={"observations": 6, "evidence": 0})) == "answered_from_artifacts"


def test_an_answer_citing_evidence_with_nothing_resolved_is_still_insufficient():
    assert _stop(_answered(basis={"observations": 1, "evidence": 2})) == "insufficient_evidence"
    assert _stop(_answered()) == "insufficient_evidence"


def test_waiting_on_the_researcher_is_its_own_stop_and_names_the_input():
    state = _answered(
        basis={"observations": 1, "evidence": 0},
        declared_stop={"reason": "needs_user_input",
                       "needed_inputs": ["free plasma Cmax at the intended dose"]},
    )
    assert _stop(state) == "needs_user_input"
    assert state.to_dict()["declared_stop"]["needed_inputs"] == [
        "free plasma Cmax at the intended dose"
    ]


def test_a_declared_stop_other_than_needs_input_is_recorded_not_obeyed():
    """The server derives sufficiency from the run; the answer saying so is
    recorded beside it for a grader, not taken over it."""
    state = _answered(basis={"observations": 0, "evidence": 1},
                      declared_stop={"reason": "sufficient", "needed_inputs": []})
    assert _stop(state) == "insufficient_evidence"
    assert state.declared_stop["reason"] == "sufficient"


def test_a_needs_input_stop_must_name_the_input():
    with pytest.raises(ValueError, match="needed_inputs"):
        StopDeclarationV2(reason="needs_user_input")


def test_the_stop_field_exists_only_where_the_flag_puts_it():
    assert "stop" not in GroundedAnswerDraftV2.model_fields
    assert draft_model(spans=True, stop=True) is GroundedAnswerDraftV2SpansStop
    assert "stop" in draft_model(spans=False, stop=True).model_fields
    assert "evidence_spans" not in str(draft_model(spans=False, stop=True).model_json_schema())


def test_the_state_round_trips_its_new_fields():
    state = _answered(basis={"observations": 2, "evidence": 0},
                      declared_stop={"reason": "budget_reached", "needed_inputs": []})
    again = ds.DecisionSupportStateV1.from_dict(state.to_dict())
    assert again.answer_basis == {"observations": 2, "evidence": 0}
    assert again.declared_stop["reason"] == "budget_reached"


# ── the named budget ──────────────────────────────────────────────────────

def test_the_pilot_preset_sets_the_whole_budget(monkeypatch):
    from toxagent.application.runs.budget import effective_run_budget
    from toxagent.platform.config import PolicySettings, RuntimeSettings

    monkeypatch.setenv("TOXAGENT_RUN_BUDGET_PRESET", "investigator-pilot-v1")
    policy, runtime = PolicySettings.from_env(), RuntimeSettings.from_env()
    budget = effective_run_budget("decision_support", policy, runtime)

    assert budget.preset == "investigator-pilot-v1"
    assert (budget.deadline_s, budget.turn_deadline_s, budget.max_tool_calls) == (360, 360, 20)
    assert (budget.max_searches, budget.max_evidence_reads, budget.max_full_text_reads) == (3, 8, 4)


def test_an_explicit_variable_still_wins_and_shows(monkeypatch):
    from toxagent.platform.config import PolicySettings

    monkeypatch.setenv("TOXAGENT_RUN_BUDGET_PRESET", "investigator-pilot-v1")
    monkeypatch.setenv("TOXAGENT_RUN_DEADLINE_S", "900")
    assert PolicySettings.from_env().run_deadline_s == 900


def test_an_unknown_preset_stops_the_deployment(monkeypatch):
    from toxagent.platform.config import PolicySettings

    monkeypatch.setenv("TOXAGENT_RUN_BUDGET_PRESET", "whatever-v9")
    with pytest.raises(ValueError, match="TOXAGENT_RUN_BUDGET_PRESET"):
        PolicySettings.from_env()


def test_without_a_preset_nothing_changes(monkeypatch):
    from toxagent.application.runs.budget import (
        DECISION_SUPPORT_MAX_SEARCHES_PER_RUN,
        decision_support_limit,
    )

    monkeypatch.delenv("TOXAGENT_RUN_BUDGET_PRESET", raising=False)
    assert decision_support_limit("max_searches") == DECISION_SUPPORT_MAX_SEARCHES_PER_RUN
