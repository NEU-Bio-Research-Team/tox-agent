"""#53: the v2 scorer, against numbers worked out by hand.

The controls the master plan asks the evaluator to demonstrate (§14.5): a
response that never came stays in the denominator; a case whose data settles
it is not passed by a generic refusal while a case needing input is; deleting
every claim cannot buy limitation precision at the cost of required
information; a reviewer's verdict is not Task Success; dependent cases are
one cluster in the interval; and an unverified gold or an ungraded cell keeps
the scientific table from being called final.
"""
from __future__ import annotations

import pytest

from evals.investigation.scorecard_v2 import (
    agreement,
    build_scorecard_v2,
    completeness,
    limitations,
    mandatory_information,
    outcome_rows,
    paired_delta,
    render_markdown,
    response_outcome,
    task_success,
    usage_summary,
)

CRITICALS = ["false_reassurance", "invented_value"]


def _case(case_id: str, *, group: str = "T1", context: str = "toxicity_only",
          stress: bool = False, compounds: tuple[str, ...] = (), fault: str | None = None) -> dict:
    case: dict = {"case_id": case_id, "task_group": group, "workflow_context": context,
                  "stress_set": stress,
                  "dependency_groups": {"compounds": list(compounds), "documents": [],
                                        "measurements": [], "sources": []}}
    if fault:
        case["fault"] = {"mechanism": fault}
    return case


def _key(*responses: dict, gold_status: dict | None = None) -> dict:
    out = {"study_id": "s", "packet_id": "p", "responses": [], "gold_status": gold_status or {}}
    for response in responses:
        out["responses"].append({"trial": 1, **response})
    return out


def _grade(case_id: str, response_id: str, *, grader: str = "g1", success: bool | None = True,
           critical: dict | None = None, facts: tuple[str, ...] = (),
           limits: tuple[str, ...] = (), forbidden: tuple[str, ...] = ()) -> dict:
    return {"grader_id": grader, "case_id": case_id, "response_id": response_id,
            "task_success": success,
            "critical": {c: (critical or {}).get(c) for c in CRITICALS},
            "mandatory_facts_met": list(facts), "limitations_stated": list(limits),
            "forbidden_inferences_committed": list(forbidden), "comments": ""}


def test_a_response_that_never_came_is_in_the_denominator_and_is_not_a_success():
    key = _key(
        {"case_id": "c1", "response_id": "R01", "system_id": "D"},
        {"case_id": "c2", "response_id": None, "system_id": "D", "missing_reason": "error"},
    )
    cases = {"c1": _case("c1"), "c2": _case("c2")}
    rows = outcome_rows(key=key, grades=[_grade("c1", "R01")], cases=cases)
    table = task_success(rows)["D"]
    assert table["all_assigned"] == {
        "successes": 1, "assigned": 2, "rate": 0.5,
        "ci95_clopper_pearson": table["all_assigned"]["ci95_clopper_pearson"],
    }
    assert table["failure_causes"] == {"no_response": 1}


def test_a_critical_error_makes_a_response_unsuccessful_whatever_the_verdict():
    success, cause = response_outcome([_grade("c", "R", success=True,
                                              critical={"false_reassurance": True})])
    assert (success, cause) == (False, "critical_error")


def test_graders_who_disagree_leave_it_unsuccessful_until_adjudicated():
    judged = [_grade("c", "R", grader="a", success=True),
              _grade("c", "R", grader="b", success=False)]
    assert response_outcome(judged) == (False, "graders_disagree")
    report = agreement(judged)
    assert report["exact_agreement"] == 0.0 and report["pending_adjudication"] == ["c/R"]


def test_a_withheld_result_is_named_as_such_not_as_a_wrong_answer():
    key = _key({"case_id": "c1", "response_id": "R01", "system_id": "D",
                "disposition": "withhold"})
    rows = outcome_rows(key=key, grades=[_grade("c1", "R01", success=False)],
                        cases={"c1": _case("c1")})
    assert rows["D"][0]["cause"] == "withheld"


def test_the_stress_set_and_the_contexts_are_reported_apart():
    key = _key(
        {"case_id": "tox", "response_id": "R01", "system_id": "D"},
        {"case_id": "act", "response_id": "R02", "system_id": "D"},
        {"case_id": "fault", "response_id": "R03", "system_id": "D"},
    )
    cases = {"tox": _case("tox", context="toxicity_only"),
             "act": _case("act", context="activity_guided", group="T2"),
             "fault": _case("fault", group="T6", stress=True, fault="instruction_in_source")}
    grades = [_grade("tox", "R01", success=True), _grade("act", "R02", success=False),
              _grade("fault", "R03", success=False)]
    table = task_success(outcome_rows(key=key, grades=grades, cases=cases))["D"]
    assert table["scientific_tasks"]["assigned"] == 2
    assert table["scientific_tasks"]["successes"] == 1
    assert table["stress_set"] == {"successes": 0, "assigned": 1, "rate": 0.0,
                                   "ci95_clopper_pearson": (0.0, 0.9750)}
    assert table["by_workflow_context"]["toxicity_only"]["rate"] == 1.0
    assert table["by_workflow_context"]["activity_guided"]["rate"] == 0.0
    assert table["by_fault_mechanism"]["instruction_in_source"]["rate"] == 0.0


def test_the_paired_delta_counts_discordant_cases_and_clusters_the_interval():
    """Four cases: D wins two, B wins one, both succeed on one.
    delta = (2 - 1) / 4 = 0.25. Two of them share a compound, so the four
    cases are three clusters."""
    responses = []
    for case_id in ("c1", "c2", "c3", "c4"):
        for system in ("D", "B"):
            responses.append({"case_id": case_id, "response_id": f"R-{case_id}-{system}",
                              "system_id": system})
    key = _key(*responses)
    cases = {"c1": _case("c1", compounds=("CCO",)), "c2": _case("c2", compounds=("CCO",)),
             "c3": _case("c3", compounds=("CCN",)), "c4": _case("c4", compounds=("CCC",))}
    wins = {("c1", "D"): True, ("c1", "B"): False,
            ("c2", "D"): True, ("c2", "B"): False,
            ("c3", "D"): False, ("c3", "B"): True,
            ("c4", "D"): True, ("c4", "B"): True}
    grades = [_grade(case_id, f"R-{case_id}-{system}", success=win)
              for (case_id, system), win in wins.items()]
    rows = outcome_rows(key=key, grades=grades, cases=cases)
    paired = paired_delta(rows, system="D", baseline="B", cases=cases, reps=200)
    assert paired["paired_assigned"] == 4
    assert (paired["n_system_only"], paired["n_baseline_only"]) == (2, 1)
    assert (paired["n_both"], paired["n_neither"]) == (1, 0)
    assert paired["delta"] == 0.25
    assert paired["clusters"] == 3  # c1+c2 share a compound
    low, high = paired["ci95_cluster_bootstrap"]
    assert low <= 0.25 <= high and low < high


def test_deleting_every_claim_cannot_buy_precision_at_the_cost_of_required_facts():
    """One arm states both facts and both required limitations; the other
    states no fact and one limitation. Precision is 100% for both -- and the
    required-information recall separates them."""
    key = _key({"case_id": "c1", "response_id": "R01", "system_id": "full"},
               {"case_id": "c1", "response_id": "R02", "system_id": "empty"})
    gold = {"c1": {"mandatory_facts": [{"id": "ic50", "statement": "x"},
                                        {"id": "unit", "statement": "y"}],
                    "required_limitations": ["L1", "L7"], "limitation_scoring": "scored"}}
    grades = [_grade("c1", "R01", facts=("ic50", "unit"), limits=("L1", "L7")),
              _grade("c1", "R02", facts=(), limits=("L1",))]
    facts = mandatory_information(key=key, grades=grades, gold=gold)
    assert facts["full"]["recall"] == 1.0 and facts["empty"]["recall"] == 0.0
    limits = limitations(key=key, grades=grades, gold=gold)
    assert limits["full"]["stated_precision"] == 1.0 == limits["empty"]["stated_precision"]
    assert limits["full"]["required_recall"] == 1.0
    assert limits["empty"]["required_recall"] == 0.5
    assert limits["full"]["all_required_pass_rate"] == 1.0
    assert limits["empty"]["all_required_pass_rate"] == 0.0


def test_a_case_requiring_no_limitation_is_na_not_a_free_hundred_percent():
    key = _key({"case_id": "c1", "response_id": "R01", "system_id": "D"})
    gold = {"c1": {"required_limitations": [], "limitation_scoring": "not_applicable"}}
    row = limitations(key=key, grades=[_grade("c1", "R01", limits=("L1",))], gold=gold)["D"]
    assert row["na"] == 1 and row["cases"] == 0
    assert row["required_recall"] is None and row["all_required_pass_rate"] is None


def test_a_missing_response_owes_every_fact_of_its_case():
    key = _key({"case_id": "c1", "response_id": None, "system_id": "D",
                "missing_reason": "error"})
    gold = {"c1": {"mandatory_facts": [{"id": "a", "statement": "x"}]}}
    row = mandatory_information(key=key, grades=[], gold=gold)["D"]
    assert row == {"facts_met": 0, "facts_required": 1, "recall": 0.0,
                   "cases_with_required_facts": 1, "all_facts_cases": 0}


def test_an_ungraded_cell_or_unverified_gold_keeps_the_table_unpublishable():
    key = _key({"case_id": "c1", "response_id": "R01", "system_id": "D"},
               gold_status={"c1": "pending_sme_verification"})
    cases = {"c1": _case("c1")}
    rows = outcome_rows(key=key, grades=[], cases=cases)
    report = completeness(key=key, grades=[], rows=rows)
    assert report["ungraded_responses"] == ["c1/R01"]
    assert report["scientific_table_publishable"] is False
    assert "responses are ungraded" in report["why_not"]
    assert "the gold is not SME-verified" in report["why_not"]
    assert rows["D"][0]["cause"] == "not_graded"

    verified = _key({"case_id": "c1", "response_id": "R01", "system_id": "D"},
                    gold_status={"c1": "sme_verified"})
    done = completeness(key=verified, grades=[_grade("c1", "R01")], rows=rows)
    assert done["scientific_table_publishable"] is True and done["why_not"] == []


def test_the_card_states_the_denominator_and_never_a_total_score():
    key = _key({"case_id": "c1", "response_id": "R01", "system_id": "D"},
               {"case_id": "c1", "response_id": "R02", "system_id": "B"},
               gold_status={"c1": "synthetic_engineering_gold"})
    cases = {"c1": _case("c1")}
    gold = {"c1": {"mandatory_facts": [], "required_limitations": ["L1"],
                    "limitation_scoring": "scored"}}
    card = build_scorecard_v2(
        key=key, grades=[_grade("c1", "R01", limits=("L1",)),
                         _grade("c1", "R02", success=False)],
        cases=cases, gold=gold, criticals=CRITICALS, compare=[("D", "B")], reps=50)
    assert card["task_success"]["D"]["all_assigned"]["rate"] == 1.0
    assert card["task_success"]["B"]["all_assigned"]["rate"] == 0.0
    assert card["paired"][0]["delta"] == 1.0
    assert "no total score" in card["note"]
    text = render_markdown(card)
    assert "Task success (every assigned case and trial)" in text
    assert "Not publishable as a scientific result" in text


class _Turn:
    def __init__(self, duration: float | None, phase_usage: dict | None = None) -> None:
        self.duration_s = duration
        self.meta = {"phase_usage": phase_usage} if phase_usage else {}


class _Record:
    def __init__(self, system_id: str, usage: dict, *, kind: str = "primary",
                 turns: list | None = None) -> None:
        self.system_id = system_id
        self.usage = usage
        self.attempt_kind = kind
        self.turns = turns or [_Turn(2.0)]


def test_unknown_usage_is_not_zero_and_every_attempt_is_counted():
    records = [
        _Record("D", {"status": "reported", "tokens_input": 10, "cost_usd": 0.5}),
        _Record("D", {"status": "unknown"}, kind="diagnostic_rerun",
                turns=[_Turn(4.0, {"phases": {"investigator": {"dispatches": 1,
                                                                "calls": {"completed": 3}}}})]),
    ]
    row = usage_summary(records)["D"]
    assert row["attempts"] == 2 and row["diagnostic_attempts"] == 1
    assert row["tokens"] == {"tokens_input": 10} and row["cost"] == {"cost_usd": 0.5}
    assert row["records_with_unknown_usage"] == 1
    assert row["median_turn_seconds"] == 3.0
    assert row["phases"]["investigator"] == {"dispatches": 1, "calls": 3}


@pytest.mark.parametrize(
    "answerability,generic_refusal_graded,expected",
    [("answerable_from_corpus", False, False), ("needs_user_input", True, True)],
)
def test_the_grader_decides_abstention_and_the_scorer_keeps_the_denominator(
    answerability, generic_refusal_graded, expected,
):
    """Abstention is right on a case needing input and wrong on one the
    sources settle; either way the case is in the denominator."""
    key = _key({"case_id": "c1", "response_id": "R01", "system_id": "D"})
    cases = {"c1": {**_case("c1"), "answerability": answerability}}
    rows = outcome_rows(key=key, grades=[_grade("c1", "R01", success=generic_refusal_graded)],
                        cases=cases)
    assert rows["D"][0]["success"] is expected
    assert task_success(rows)["D"]["all_assigned"]["assigned"] == 1


def test_retrieval_is_measured_from_the_trace_and_placed_before_or_after_the_read():
    """One gold span. The arm that found, read and quoted the row scores
    three out of three; the arm that only found the record scores the first
    and not the rest -- which is what says where the failure was."""
    from evals.investigation.scorecard_v2 import retrieval_summary

    gold = {"c1": {"evidence_spans": [{"evidence_ref": "pmid:90001",
                                        "locator": "table-wrap:1/row:2"}]}}
    key = _key({"case_id": "c1", "response_id": "R01", "system_id": "D"},
               {"case_id": "c1", "response_id": "R02", "system_id": "B"})
    evidence = [{"evidence_id": "evd_1", "identifier": {"pmid": "90001"},
                 "full_text": {"state": "full_text_retrieved"}}]
    quoted = {
        "evidence": evidence,
        "runs": [{"run": {"tool_calls": [{"tool_name": "read_evidence_text",
                                           "status": "completed",
                                           "observation_ids": ["evd_1"]}]},
                  "answer": {"claims": [{"citation_ids": ["evd_1"],
                                          "evidence_spans": [{"evidence_id": "evd_1",
                                                              "locator": "table-wrap:1/row:2"}]}]}}],
    }
    found_only = {"evidence": evidence, "runs": [{"run": {"tool_calls": []}, "answer": None}]}
    summary = retrieval_summary(key=key, gold=gold,
                                traces={("c1", "D", 1): quoted, ("c1", "B", 1): found_only})
    assert summary["D"]["records"]["read"] == {"n": 1, "of": 1, "rate": 1.0}
    assert summary["D"]["spans"]["quoted"] == {"n": 1, "of": 1, "rate": 1.0}
    assert summary["B"]["records"]["retrieved"] == {"n": 1, "of": 1, "rate": 1.0}
    assert summary["B"]["records"]["read"] == {"n": 0, "of": 1, "rate": 0.0}
    assert summary["B"]["spans"]["quoted"] == {"n": 0, "of": 1, "rate": 0.0}
    assert summary["D"]["unusable_refs"] == []
