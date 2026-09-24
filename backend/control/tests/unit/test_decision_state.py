"""DecisionSupportStateV1 transitions and the canonical evidence ontology.

Pure functions, no database: every transition a run can take is replayable
from its inputs, which is what lets a benchmark grade plan, coverage and stop
reason rather than inferring them from a tool list.
"""
from __future__ import annotations

import pytest

from toxagent.domain import decision_state as ds
from toxagent.domain import evidence_ontology as onto
from toxagent.domain.evidence_relation import RelationLabel
from toxagent.domain.report import EvidenceRelation as ReportRelation
from toxagent.research.relevance import Relevance

BUDGET = {"max_tool_calls": 24, "max_searches": 2, "max_evidence_reads": 8}


def _state(**kw) -> ds.DecisionSupportStateV1:
    return ds.initial(
        session_id="ses_1", run_id="run_1", goal="Should we develop this?",
        subject_refs=["analysis:ana_1", "analysis:ana_1"], budget_snapshot=kw.pop("budget", BUDGET),
    )


def _plan(state):
    return ds.apply_plan(state, [
        {"question": "Does it block hERG?", "required_sources": ["prediction", "evidence"]},
        {"question": "Is there hepatotoxicity evidence?", "required_sources": ["evidence"]},
    ])


def test_the_server_issues_proposition_ids_and_bumps_revision():
    state = _state()
    assert state.subject_refs == ("analysis:ana_1",)
    planned = _plan(state)
    assert [p.id for p in planned.propositions] == ["p1", "p2"]
    assert planned.revision == state.revision + 1
    assert planned.coverage == {"required": 2, "resolved": 0}


@pytest.mark.parametrize("bad", [
    [{"question": "", "required_sources": ["evidence"]}],
    [{"question": "x", "required_sources": []}],
    [{"question": "x", "required_sources": ["shell"]}],
    [{"question": f"q{i}", "required_sources": ["evidence"]} for i in range(ds.MAX_PROPOSITIONS + 1)],
])
def test_a_bad_plan_is_refused(bad):
    with pytest.raises(ds.InvalidPlan):
        ds.apply_plan(_state(), bad)


def test_a_replan_keeps_what_is_resolved():
    state = _plan(_state())
    state = ds.apply_answer(state, relations=[
        {"proposition": "Does it block hERG?", "source_class": "predictor_fact",
         "source_id": "obs_1", "relation": "supports"},
    ], is_fallback=False, candidate_generation=1)
    replanned = ds.apply_plan(state, [{"question": "New question", "required_sources": ["report"]}])
    questions = {p.question: p.status for p in replanned.propositions}
    assert questions == {"Does it block hERG?": "supported", "New question": "open"}
    assert {p.id for p in replanned.propositions} == {"p1", "p3"}


def test_tool_results_count_usage_and_availability():
    state = _state()
    state = ds.apply_tool_result(state, tool_name="search_toxicology_evidence", status="completed",
                                 found_evidence_ids=["evd_1", "evd_2"])
    state = ds.apply_tool_result(state, tool_name="get_evidence_record", status="completed",
                                 observation_ids=["evd_1"])
    state = ds.apply_tool_result(state, tool_name="get_analysis_slice", status="denied",
                                 error_code="tool_denied")
    assert state.usage["searches"] == 1 and state.usage["evidence_reads"] == 1
    assert state.usage["denied_calls"] == 1 and state.usage["tool_calls"] == 3
    assert state.available_refs == ("evidence:evd_1",)
    assert state.found_refs == ("evidence:evd_1", "evidence:evd_2")


def test_answer_relations_resolve_propositions_only_with_refs():
    state = _plan(_state())
    state = ds.apply_answer(state, relations=[
        {"proposition": "Does it block hERG?", "source_class": "predictor_fact",
         "source_id": "obs_1", "relation": "supports"},
        {"proposition": "Does it block hERG?", "source_class": "external_experimental",
         "source_id": "evd_1", "relation": "contradicts"},
        {"proposition": "Is there hepatotoxicity evidence?", "source_class": "agent_synthesis",
         "source_id": "obs_9", "relation": "supports"},
        {"proposition": "Unplanned question", "source_class": "report_fact",
         "source_id": "rpt_1", "relation": "contradicts"},
    ], is_fallback=False, candidate_generation=2)
    by_question = {p.question: p for p in state.propositions}
    assert by_question["Does it block hERG?"].status == "conflicted"
    assert set(by_question["Does it block hERG?"].artifact_refs) == {"observation:obs_1", "evidence:evd_1"}
    # A synthesis alone is not a grounded resolution.
    assert by_question["Is there hepatotoxicity evidence?"].status == "insufficient"
    assert by_question["Unplanned question"].status == "contradicted"
    assert by_question["Unplanned question"].origin == "answer"
    assert state.answer_outcome == "accepted_after_correction"


@pytest.mark.parametrize("setup, run_status, expected", [
    ("all_resolved", "completed", "sufficient"),
    ("open_budget_left", "completed", "insufficient_evidence"),
    ("open_searches_spent", "completed", "budget_exhausted"),
    ("fallback", "completed", "blocked"),
    ("open_budget_left", "failed", "blocked"),
    ("open_budget_left", "cancelled", "cancelled"),
    ("no_plan", "completed", "insufficient_evidence"),
])
def test_the_stop_reason_says_why_the_run_stopped(setup, run_status, expected):
    state = _plan(_state())
    if setup == "no_plan":
        state = ds.apply_answer(_state(), relations=[], is_fallback=False, candidate_generation=1)
    elif setup == "fallback":
        state = ds.apply_answer(state, relations=[], is_fallback=True, candidate_generation=2)
    else:
        if setup == "open_searches_spent":
            for _ in range(2):
                state = ds.apply_tool_result(state, tool_name="search_toxicology_evidence",
                                             status="completed")
        relations = [{"proposition": "Does it block hERG?", "source_class": "predictor_fact",
                      "source_id": "obs_1", "relation": "supports"}]
        if setup == "all_resolved":
            relations.append({"proposition": "Is there hepatotoxicity evidence?",
                              "source_class": "external_experimental", "source_id": "evd_2",
                              "relation": "contradicts"})
        state = ds.apply_answer(state, relations=relations, is_fallback=False, candidate_generation=1)
    final = ds.finalize(state, run_status=run_status)
    assert final.stop_reason == expected
    assert ds.finalize(final, run_status="completed") is final  # idempotent
    with pytest.raises(ds.InvalidPlan):
        ds.apply_plan(final, [{"question": "late", "required_sources": ["evidence"]}])


def test_state_round_trips_through_its_persisted_form():
    state = ds.finalize(_plan(_state()), run_status="failed")
    assert ds.DecisionSupportStateV1.from_dict(state.to_dict()) == state
    summary = ds.checkpoint_summary(state)
    assert "Still open" in summary and "blocked" in summary


# --------------------------------------------------------------- ontology

def test_every_legacy_relation_value_has_a_canonical_mapping():
    for label in RelationLabel:
        onto.canonical_relation("decision_support", label.value)
    for label in ReportRelation:
        onto.canonical_relation("report", label.value)
    for relevance in Relevance:
        onto.canonical_relevance(relevance.value)
    assert onto.canonical_assessor("rule") is onto.Assessor.DETERMINISTIC


def test_the_two_paths_agree_on_equivalent_values():
    pairs = [("contextual", "contextualizes"), ("supports", "supports"),
             ("contradicts", "contradicts"), ("insufficient", "insufficient")]
    for ds_value, report_value in pairs:
        assert onto.canonical_relation("decision_support", ds_value) == onto.canonical_relation(
            "report", report_value
        )


def test_unrelated_or_unresolved_is_never_citable():
    for relation in onto.CanonicalRelation:
        citable = onto.citable_for_claim(relation, exists=True, read=True)
        assert citable is (relation not in (onto.CanonicalRelation.UNRELATED,
                                            onto.CanonicalRelation.UNRESOLVED))
    assert not onto.citable_for_claim(onto.CanonicalRelation.SUPPORTS, exists=True, read=False)
