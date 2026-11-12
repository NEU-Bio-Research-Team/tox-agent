"""The gap-driven loop (#70) and discriminating actions (#71), as pure rules."""
from __future__ import annotations

from toxagent.application.investigation import decision_loop as loop
from toxagent.domain import decision_state as ds
from toxagent.domain.investigation_gap import Gap

CONTRACT = {"must_answer": [
    {"id": "q1", "question": "Does it block hERG at exposure?", "needs": ["evidence"]},
    {"id": "q2", "question": "What is the free Cmax?", "needs": ["researcher_input"]},
]}


def c(**kw) -> loop.Candidate:
    base = {"action": "search_toxicology_evidence", "target": "q", "requirement_id": "q1",
            "discriminates": [Gap.NO_RELEVANT_EVIDENCE.value], "reason": "why"}
    base.update(kw)
    return loop.Candidate.from_dict(base)


def test_the_ledger_has_one_typed_gap_per_required_question():
    ledger = loop.seed_ledger(CONTRACT)
    gaps = {g["requirement_id"]: g for g in ledger["gaps"]}
    assert gaps["q1"]["gap"] == "no_relevant_evidence" and gaps["q1"]["decisive"]
    assert gaps["q2"]["gap"] == "only_researcher_has_it"
    assert [g["requirement_id"] for g in loop.open_gaps(ledger)] == ["q1", "q2"]


def test_ranking_follows_section_6_4_order():
    ledger = loop.seed_ledger(CONTRACT)
    candidates = [
        c(target="cheap reuse", source="observation", reuses="obs_1", requirement_id="q2"),
        c(target="decisive search", source="europepmc"),
        c(target="identity", gap=Gap.IDENTITY_OR_SCOPE.value,
          discriminates=["identity_mismapped"], requirement_id="q2"),
    ]
    order = [r["target"] for r in loop.rank(candidates, ledger=ledger)]
    # (1) identity blocker, then (2) the open decisive question, then the rest.
    assert order == ["identity", "decisive search", "cheap reuse"]


def test_an_unreachable_source_ranks_below_a_reachable_one():
    ledger = loop.seed_ledger(CONTRACT)
    ranked = loop.rank([c(target="a", source="chembl"), c(target="b", source="europepmc")],
                       ledger=ledger, access_failures=[{"provider": "chembl"}])
    assert [r["target"] for r in ranked] == ["b", "a"]
    assert ranked[1]["criteria"]["accessible"] is False


def test_a_chembl_timeout_makes_pubchem_the_reachable_assay_option():
    state = ds.initial(session_id="ses_1", run_id="run_1", goal="hERG",
                       subject_refs=(), budget_snapshot={})
    state = ds.apply_tool_result(state, tool_name="get_chembl_activities", status="error",
                                 error_code="tool_timeout")
    ranked = loop.rank([
        c(action="get_chembl_activities", source="chembl", target="hERG"),
        c(action="get_pubchem_bioassays", source="pubchem", target="hERG"),
    ], ledger=loop.seed_ledger(CONTRACT), access_failures=state.access_failures)
    assert ranked[0]["action"] == "get_pubchem_bioassays"
    assert ranked[1]["criteria"]["accessible"] is False


def test_reuse_comes_before_a_new_search_when_all_else_is_equal():
    ledger = loop.seed_ledger(CONTRACT)
    ranked = loop.rank([c(target="new", source="observation"),
                        c(target="reuse", source="observation", reuses="obs_1")], ledger=ledger)
    assert ranked[0]["target"] == "reuse"


def test_repeats_inadmissible_actions_and_undeclared_purpose_are_refused():
    ledger = loop.seed_ledger(CONTRACT)
    ledger["actions"].append({"action": "search_toxicology_evidence", "target": "hERG  IC50!"})
    refused = loop.refusals(
        [c(target="herg ic50"), c(action="shell"), c(discriminates=[]), c(discriminates=["vibes"]),
         c(action="answer", discriminates=[])],
        ledger=ledger, admissible=["search_toxicology_evidence"],
    )
    assert set(refused) == {0, 1, 2, 3}
    assert "already" in refused[0]


def test_counter_evidence_is_asked_once_before_a_decisive_answer_from_evidence():
    ledger = loop.seed_ledger(CONTRACT)
    answer = c(action="answer", discriminates=[])
    # A lookup that never searched is not asked to.
    assert loop.counter_evidence_due(ledger, answer, search_allowed=True, budget_left=5) is None
    ledger["actions"].append({"action": "search_toxicology_evidence", "target": "x"})
    assert loop.counter_evidence_due(ledger, answer, search_allowed=True, budget_left=5)
    assert loop.counter_evidence_due(ledger, answer, search_allowed=False, budget_left=5) is None
    assert loop.counter_evidence_due(ledger, answer, search_allowed=True, budget_left=0) is None
    loop.mark_counter_evidence(ledger, c(reason="counter-evidence: other assay"))
    assert loop.counter_evidence_due(ledger, answer, search_allowed=True, budget_left=5) is None


def test_an_insufficient_stop_says_what_is_missing_and_what_it_limits():
    ledger = loop.seed_ledger(CONTRACT)
    loop.resolve(ledger, [{"requirement_id": "q1", "status": "open",
                           "gap": Gap.MEASUREMENT_UNRESOLVED.value}])
    detail = loop.insufficiency(ledger)
    assert detail[0] == {"requirement_id": "q1", "missing": "measurement_unresolved",
                         "limits": "Does it block hERG at exposure?", "blocking": False}
    assert detail[1]["blocking"] is True


def _answered(**changes) -> ds.DecisionSupportStateV1:
    state = ds.initial(session_id="ses_" + "1" * 32, run_id="run_" + "2" * 32,
                       goal="g", subject_refs=(), budget_snapshot={})
    state = ds.apply_plan(state, [{"question": "q", "required_sources": ["evidence"]}])
    return ds._next(state, answer_outcome="accepted", **changes)


def test_read_but_unsettled_is_not_reported_as_no_evidence():
    ledger = loop.seed_ledger(CONTRACT)
    read = ds.finalize(_answered(decision_loop=ledger, usage={"evidence_reads": 2}),
                       run_status="completed")
    assert read.stop_reason == "sources_unresolved"
    assert read.stop_detail and read.stop_detail[0]["requirement_id"] == "q1"
    unread = ds.finalize(_answered(decision_loop=ledger), run_status="completed")
    assert unread.stop_reason == "insufficient_evidence"
    # Without the loop, the old reading stands.
    assert ds.finalize(_answered(usage={"evidence_reads": 2}), run_status="completed").stop_reason \
        == "insufficient_evidence"


def test_a_provider_failure_stays_distinct_from_unsettled_sources():
    ledger = loop.seed_ledger(CONTRACT)
    state = _answered(decision_loop=ledger, usage={"evidence_reads": 1},
                      access_failures=({"provider": "europepmc", "asked_for": "x",
                                        "reason": "timeout", "gap": "provider_or_corpus"},))
    assert ds.finalize(state, run_status="completed").stop_reason == "provider_failure"


def test_the_ledger_survives_a_round_trip():
    state = _answered(decision_loop=loop.seed_ledger(CONTRACT))
    again = ds.DecisionSupportStateV1.from_dict(state.to_dict())
    assert again.decision_loop == state.decision_loop
