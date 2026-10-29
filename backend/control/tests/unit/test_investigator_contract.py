"""The investigator's contract, route and case view (#69, #68, #63, #70).

What a run was asked to do, which route it takes, and what it reads before
each step -- the §6.1/§4.1/§5.4 trio. Pure functions: a benchmark has to be
able to replay a routing decision and a contract without a runtime.
"""
import inspect

import pytest

from toxagent.application.investigation import case_view, goal_contract, router
from toxagent.domain import decision_state as ds
from toxagent.domain import scientific_case as sc
from toxagent.domain.investigation_gap import Gap

SESSION = "ses_" + "1" * 32
CASE_ID = "scase_" + "2" * 32
RUN = "run_" + "6" * 32
OBS = "obs_" + "3" * 32
EVD = "evd_" + "4" * 32


def upd(op, actor="model", **payload):
    return sc.CaseUpdate(op=op, payload=payload, actor=actor,
                         at="2026-10-28T09:00:00+00:00", run_id=RUN)


def opened():
    return sc.apply(None, upd(
        "open", actor="server", case_id=CASE_ID, session_id=SESSION,
        subject_key="analysis:ana_x", subject_refs=["analysis:ana_x"],
        question="Does the measurement contradict the prediction?",
    ))


# ── the four routes (#68, §4.1/§4.2) ──────────────────────────────────────

def test_a_bound_value_question_is_a_read_not_an_investigation():
    decision = router.classify(
        question="What is the pIC50 and has the toxicity stage run?",
        has_subject=True, artifact_refs=["screening:scr_1"],
    )

    assert decision.route == router.READ_EXPLAIN
    assert decision.confidence == "high"
    assert "artifact_available" in decision.reason_codes


def test_a_lookup_phrase_alone_does_not_make_a_lookup():
    """§4.2: the cheap route needs the subject and the artifact to be bound.
    Wording alone routing a question is how a research question gets answered
    by printing a field."""
    decision = router.classify(question="What is the pIC50?", has_subject=False)

    assert decision.route == router.UNDETERMINED

    decision = router.classify(question="What is the pIC50?", has_subject=True,
                               artifact_refs=[])
    assert decision.route == router.INVESTIGATE


def test_an_assessment_question_outranks_every_lookup_marker():
    """"What is the margin" reads like a lookup and is a judgement. §4.2 names
    this: a keyword must not decide a research question."""
    decision = router.classify(
        question="What is the margin — do these two results contradict each other?",
        has_subject=True, artifact_refs=["analysis:ana_1"],
    )

    assert decision.route == router.INVESTIGATE
    assert "assessment_wording" in decision.reason_codes


def test_the_word_safe_does_not_settle_anything_by_itself():
    decision = router.classify(question="Is this compound safe?", has_subject=True,
                               artifact_refs=["analysis:ana_1"])

    assert decision.route == router.INVESTIGATE
    assert decision.confidence == "low"  # wording decided it, and says so


def test_a_changed_premise_makes_it_an_update_whatever_the_wording():
    """A value read off a stale claim is wrong, so this outranks the lookup
    route even for a question that only asks to see a number."""
    decision = router.classify(
        question="What is the exposure margin?", has_subject=True,
        artifact_refs=["analysis:ana_1"], stale_claims=["dc1"],
    )

    assert decision.route == router.UPDATE
    assert decision.confidence == "high"
    assert decision.detail["stale_claims"] == ["dc1"]


def test_researcher_input_is_an_update_declared_by_the_server():
    decision = router.classify(question="Free Cmax is 0.01 µM", has_subject=True,
                               artifact_refs=["analysis:ana_1"],
                               new_researcher_input=True)

    assert (decision.route, decision.confidence) == (router.UPDATE, "high")
    assert "researcher_supplied_input" in decision.reason_codes


def test_several_candidates_route_to_compare_batch():
    decision = router.classify(question="Which of these should we take forward?",
                               has_subject=True, artifact_refs=["analysis:a"],
                               candidate_count=4)

    assert decision.route == router.COMPARE_BATCH
    assert decision.detail["candidates"] == 4


def test_no_subject_and_no_artifact_asks_rather_than_guesses():
    decision = router.classify(question="Does it block hERG?", has_subject=False)

    assert decision.route == router.UNDETERMINED
    assert "Which compound" in decision.clarification
    assert decision.reason_codes == ("no_subject_bound",)


def test_the_light_routes_do_not_open_hypotheses():
    """§4.2: read/explain does not need eight competing explanations."""
    assert not router.opens_hypotheses(router.READ_EXPLAIN)
    assert not router.opens_hypotheses(router.UNDETERMINED)
    assert router.opens_hypotheses(router.INVESTIGATE)
    assert router.opens_hypotheses(router.UPDATE)


def test_no_decision_here_charges_a_classifier():
    """The deterministic check comes before any model call, and a deployment
    that adds one must be able to report its cost separately (§4.2)."""
    for question in ("What is the pIC50?", "Is this safe?", "Compare these"):
        assert not router.classify(question=question, has_subject=True,
                                   artifact_refs=["a:1"]).classifier_used


def test_a_reordered_query_is_not_a_new_query():
    assert router.query_signature("hERG IC50 terfenadine") == \
        router.query_signature("terfenadine herg ic50")
    assert router.query_signature("hERG IC50") != router.query_signature("hERG Kd")


# ── the goal contract (#69, §6.1) ─────────────────────────────────────────

def test_the_contract_cannot_be_handed_a_benchmark_answer():
    """§6.1: required claims come from the question, never from hidden gold.
    The signature is the guarantee -- there is no parameter to pass one in."""
    parameters = set(inspect.signature(goal_contract.build).parameters)

    assert not {"gold", "expected", "reference", "answer", "acceptable_conclusions"} & parameters
    assert parameters == {
        "route", "question", "question_needs", "question_gap", "subject", "endpoint", "admissible_actions", "budget",
        "open_questions", "stale_claims", "new_values", "researcher_values_seen", "candidates",
        "clarification",
    }


def test_the_question_becomes_the_first_thing_that_must_be_answered():
    contract = goal_contract.build(route=router.INVESTIGATE,
                                   question="Does the IC50 contradict the prediction?")

    assert [q.id for q in contract.must_answer] == ["q1"]
    assert contract.must_answer[0].question == "Does the IC50 contradict the prediction?"


def test_carried_over_propositions_join_the_contract():
    """What makes the update route a contract rather than a fresh start."""
    contract = goal_contract.build(
        route=router.UPDATE, question="And now with the new exposure?",
        open_questions=[{"question": "Is there a patch-clamp IC50?",
                         "required_sources": ["evidence"]}],
    )

    assert [q.question for q in contract.must_answer] == [
        "And now with the new exposure?", "Is there a patch-clamp IC50?",
    ]
    assert contract.must_answer[1].needs == ("evidence",)


def test_a_stale_claim_becomes_a_question_with_its_gap_named():
    contract = goal_contract.build(route=router.UPDATE, question="Recompute please",
                                   stale_claims=["dc1"])

    stale = contract.must_answer[-1]
    assert "dc1" in stale.question
    assert stale.gap == Gap.MEASUREMENT_UNRESOLVED.value


def test_the_rendered_contract_states_the_stop_conditions():
    contract = goal_contract.build(
        route=router.INVESTIGATE, question="Does it contradict?",
        admissible_actions=["search_toxicology_evidence", "get_evidence_record"],
        budget={"deadline_s": 900, "max_searches": 3, "max_repairs": None},
    )

    rendered = contract.render()

    assert "provider_failure" in rendered  # the distinction #70 added
    assert "needs_user_input" in rendered
    assert "900 seconds; 3 searches" in rendered
    assert "max_repairs" not in rendered  # an unset limit is not stated
    assert "No other tool exists for you" in rendered


def test_the_contract_survives_a_round_trip():
    contract = goal_contract.build(
        route=router.UPDATE, question="q", subject={"identity": {"target": "hERG"}},
        endpoint="herg", admissible_actions=["a", "a", "b"], budget={"deadline_s": 1},
        stale_claims=["dc1"],
    )

    assert goal_contract.GoalContractV1.from_dict(contract.to_dict()) == contract
    assert contract.admissible_actions == ("a", "b")  # deduplicated, order kept


def test_an_empty_question_contributes_nothing():
    contract = goal_contract.build(route=router.READ_EXPLAIN, question="   ")

    assert contract.must_answer == ()


# ── the case view (#63, §5.4) ─────────────────────────────────────────────

def _case_with_evidence():
    case = sc.apply(opened(), upd("add_context", actor="user", key="free_cmax",
                                  value="0.01", unit="µM"))
    case = sc.apply(case, upd("add_hypothesis", statement="It blocks hERG",
                              kind="model_signal",
                              refutation_condition="A patch-clamp IC50 far above exposure"))
    case = sc.apply(case, upd(
        "record_evidence", claim="Measured IC50 0.3 µM", source_class="external_experimental",
        source_ref=f"evidence:{EVD}", stance="contradicts", directness="direct",
        hypothesis_ids=["h1"], locator="table-wrap:1",
        scope={"target_id": "hERG", "assay": ""},
    ))
    return case


def test_the_view_has_the_six_groups_in_order():
    case = _case_with_evidence()
    contract = goal_contract.build(route=router.INVESTIGATE, question="Does it contradict?")

    view = case_view.build(case, contract=contract,
                           open_gaps=[{"gap": Gap.SPAN_NOT_READ.value,
                                       "question": "Read the decisive row"}],
                           searched=["search_toxicology_evidence"],
                           budget_left={"max_searches": 2})
    rendered = view.render()

    assert "Goal contract" in rendered
    assert "Observations in this case" in rendered
    assert "Counter-evidence on file" in rendered
    assert "Unresolved gaps" in rendered
    assert "Already searched this subject" in rendered
    assert "Budget left" in rendered
    assert rendered.index("Goal contract") < rendered.index("Unresolved gaps")


def test_the_view_carries_refs_not_values():
    """§5.4: the content stays behind a tool. A value quoted from a summary is
    a value nobody can check."""
    view = case_view.build(_case_with_evidence())

    assert any(row["ref"] == "context:c1" for row in view.observations)
    rendered = view.render()
    assert "read values with a tool, never from here" in rendered


def test_contradicting_evidence_is_kept_in_its_own_group():
    view = case_view.build(_case_with_evidence())

    assert [e["ref"] for e in view.counter_evidence] == ["e1"]
    assert view.evidence == ()
    assert "do not drop it from the answer" in view.render()


def test_unstated_scope_axes_are_shown_as_unknown():
    """A turn that cannot see what a record failed to state will read the
    silence as agreement (§5.2)."""
    view = case_view.build(_case_with_evidence())

    # the entry declared target_id and nothing else that decides comparability
    assert view.missing_metadata == (
        "e1: target_variant, assay, assay_type, readout, species, experimental_context",
    )
    assert "unknown, not a match" in view.render()


def test_what_is_left_out_is_counted_so_a_tool_can_fetch_it():
    case = opened()
    for index in range(9):
        case = sc.apply(case, upd("add_context", actor="user", key=f"k{index}", value="v"))

    view = case_view.build(case, limit=4)

    assert len(view.observations) == 4
    assert view.withheld["observations"] == 5
    assert "+5 more in the case" in view.render()


def test_a_changed_premise_is_in_the_view_as_what_is_no_longer_settled():
    case = sc.apply(_case_with_evidence(), upd(
        "record_derived_claim", actor="server", statement="exposure_margin",
        rule="ratio_same_concentration_unit", inputs=[OBS, "case_value:free_cmax@v1"],
    ))
    case = sc.apply(case, upd("add_context", actor="user", key="free_cmax",
                              value="0.02", unit="µM"))

    view = case_view.build(case)

    assert any("exposure_margin" in line for line in view.changed)
    assert "not settled any more" in view.render()


def test_the_first_turn_of_a_subject_still_gets_a_view():
    contract = goal_contract.build(route=router.INVESTIGATE, question="Does it block hERG?")

    view = case_view.build(None, contract=contract, budget_left={"max_searches": 3})

    assert view.observations == ()
    assert "Goal contract" in view.render()
    assert "Budget left" in view.render()


def test_the_view_is_a_pure_function_of_the_case():
    case = _case_with_evidence()

    assert case_view.build(case).to_dict() == case_view.build(case).to_dict()


# ── the gap an action closes (#70, §6.2) ──────────────────────────────────

def test_an_action_declares_which_gap_it_closes():
    case = sc.apply(opened(), upd(
        "record_action", action="read_evidence_record", purpose="read the decisive row",
        decision="continue", gap=Gap.SPAN_NOT_READ.value,
    ))

    assert case.actions[0].gap == "span_not_read"
    assert case.actions[0].to_dict()["gap"] == "span_not_read"


def test_an_invented_gap_is_refused_rather_than_coerced():
    """Free text in a field a benchmark groups by is worse than an empty one."""
    with pytest.raises(sc.InvalidCaseUpdate) as refused:
        sc.apply(opened(), upd("record_action", action="x", purpose="y",
                               decision="continue", gap="felt_like_it"))

    assert "gap must be one of" in str(refused.value)


def test_an_action_that_names_no_gap_keeps_the_field_empty():
    """Not defaulted to a plausible gap: a run that did not say is worth
    seeing as a run that did not say."""
    case = sc.apply(opened(), upd("record_action", action="x", purpose="y",
                                  decision="continue"))

    assert case.actions[0].gap == ""
    assert "gap" not in case.actions[0].to_dict()


# ── the contract on the run record (#69) ──────────────────────────────────

def _state():
    return ds.initial(session_id=SESSION, run_id=RUN, goal="g",
                      subject_refs=["analysis:ana_x"], budget_snapshot={})


def test_the_contract_is_recorded_on_the_state():
    contract = goal_contract.build(route=router.INVESTIGATE, question="q")

    state = ds.record_goal_contract(_state(), contract.to_dict())

    assert state.goal_contract["route"] == router.INVESTIGATE
    assert ds.DecisionSupportStateV1.from_dict(state.to_dict()) == state


def test_the_contract_is_written_once():
    """A contract that could be edited mid-run would let a run that answered a
    different question look like one that answered this one."""
    first = goal_contract.build(route=router.INVESTIGATE, question="first")
    state = ds.record_goal_contract(_state(), first.to_dict())

    again = ds.record_goal_contract(
        state, goal_contract.build(route=router.UPDATE, question="second").to_dict())

    assert again is state


def test_a_state_without_a_contract_serialises_as_it_did():
    state = _state()

    assert "goal_contract" not in state.to_dict()


# ── §6.2 requirements: what a margin question cannot do without ──────────


def test_a_margin_question_with_no_measurement_opens_the_gap_the_run_can_act_on():
    """Live, arm E on T5 (2026-10-29): with nothing telling it a margin needs a
    measured potency, the investigator read the predictor's label and stopped
    in four turns of four, without one search."""
    from toxagent.application.investigation import goal_contract

    needs, gap = goal_contract.requirements(
        "What is the margin now?", has_measurement=False, has_exposure=True)
    assert needs == ("evidence", "researcher_input")
    assert gap == "no_relevant_evidence"
    # Missing both: the measurement first, because only it can be searched for.
    assert goal_contract.requirements(
        "Is the hERG margin acceptable?", has_measurement=False, has_exposure=False,
    )[1] == "no_relevant_evidence"
    assert goal_contract.requirements(
        "Is the hERG margin acceptable?", has_measurement=True, has_exposure=False,
    )[1] == "only_researcher_has_it"
    assert goal_contract.requirements(
        "What is the margin?", has_measurement=True, has_exposure=True)[1] == ""


def test_a_question_that_is_not_a_calculation_gets_no_requirements():
    from toxagent.application.investigation import goal_contract

    assert goal_contract.requirements(
        "What does the hERG score mean?", has_measurement=False, has_exposure=False,
    ) == ((), "")


def test_the_open_gap_is_rendered_with_what_it_asks_of_the_turn():
    from toxagent.application.investigation import goal_contract

    contract = goal_contract.build(
        route="investigate", question="What is the margin now?",
        question_needs=("evidence", "researcher_input"), question_gap="no_relevant_evidence",
    )
    text = contract.render()
    assert "[open gap: no_relevant_evidence -- search for a measured value" in text
    assert "a predicted label or probability is not a measurement" in text
    assert goal_contract.GoalContractV1.from_dict(contract.to_dict()) == contract


# ── §4.1: each route runs differently ─────────────────────────────────────


def test_several_subjects_in_a_session_are_not_a_request_to_compare_them():
    decision = router.classify(question="Does it block hERG?", has_subject=True,
                               artifact_refs=["analysis:a"], candidate_count=3)
    assert decision.route != router.COMPARE_BATCH


def test_asking_to_compare_with_one_candidate_asks_which_others():
    decision = router.classify(question="Compare it with the alternatives", has_subject=True,
                               artifact_refs=["analysis:a"], candidate_count=1)
    assert decision.route == router.UNDETERMINED
    assert "Which others" in decision.clarification


def test_the_light_routes_run_under_the_read_profile():
    assert router.profile_for(router.READ_EXPLAIN) == "decision_support_read"
    assert router.profile_for(router.UNDETERMINED) == "decision_support_read"
    for route in (router.INVESTIGATE, router.UPDATE, router.COMPARE_BATCH):
        assert router.profile_for(route) == "decision_support"


def test_the_update_contract_names_each_new_value_and_records_what_it_saw():
    contract = goal_contract.build(
        route="update", question="Does this change anything?",
        new_values=[{"ref": "case_value:free_cmax@v2", "key": "free_cmax", "value": "0.02",
                     "unit": "uM"}],
        researcher_values_seen=["case_value:free_cmax@v2", "case_value:free_cmax@v2"],
    )
    texts = [q.question for q in contract.must_answer]
    assert any("free_cmax = 0.02 uM (case_value:free_cmax@v2)" in t
               and "what it leaves as it was" in t for t in texts)
    assert contract.researcher_values_seen == ("case_value:free_cmax@v2",)
    assert goal_contract.GoalContractV1.from_dict(contract.to_dict()) == contract


def test_the_undetermined_contract_asks_first():
    contract = goal_contract.build(route="undetermined", question="Does it block hERG?",
                                   clarification="Which compound is this about?")
    first = contract.must_answer[0]
    assert first.question == "Which compound is this about?"
    assert first.gap == "only_researcher_has_it"


def test_the_compare_contract_lists_the_candidates_and_forbids_a_combined_score():
    contract = goal_contract.build(
        route="compare_batch", question="Compare them by hERG probability",
        candidates=[{"subject_key": "analysis:a"}, {"subject_key": "analysis:b"}],
    )
    text = contract.render()
    assert "analysis:a; analysis:b" in text
    assert "never a combined score" in text
    assert any("compare_candidates" in q.question for q in contract.must_answer)
    assert goal_contract.GoalContractV1.from_dict(contract.to_dict()) == contract


def test_a_contract_written_before_these_fields_reads_back_unchanged():
    old = goal_contract.build(route="investigate", question="Why?").to_dict()
    assert "researcher_values_seen" not in old and "candidates" not in old


@pytest.mark.parametrize("question", [
    # Phrasings that are not in any case file: the patterns are for the shape
    # of a read question, not for the dev set's wording.
    "What does the predictor say about hERG for this compound?",
    "What did the activity model estimate for it on ABL1?",
    "What does the EGFR screening say?",
    "What is the hERG label?",
    "What was the prioritisation decision?",
    "Did the toxicity stage run?",
    "What does the model report for ClinTox?",
])
def test_a_question_shaped_as_a_read_takes_the_read_route(question):
    decision = router.classify(question=question, has_subject=True,
                               artifact_refs=["analysis:a"])
    assert decision.route == router.READ_EXPLAIN, decision


@pytest.mark.parametrize("question", [
    "What does the literature say about its hERG risk?",   # assessment outranks
    "What is the margin at our dose?",                     # a calculation is not a read
    "Why was it sent to review?",
    "What does it say, and does that contradict the paper?",
])
def test_a_read_shape_that_asks_for_a_judgement_is_investigated(question):
    decision = router.classify(question=question, has_subject=True,
                               artifact_refs=["analysis:a"])
    assert decision.route == router.INVESTIGATE, decision


def test_a_read_shape_without_a_bound_subject_is_not_a_read():
    decision = router.classify(question="What does the predictor say?", has_subject=False)
    assert decision.route == router.UNDETERMINED
