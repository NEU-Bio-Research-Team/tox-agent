"""Derived claims and what a change reaches (#60, §5.3 and §11 turn 3).

The scenario the proposal builds H2 on, end to end: the researcher gives a
free Cmax, the server computes a margin from it and a recommendation from the
margin, and then the researcher corrects the Cmax. What must be computed
again, what must be left alone, and what must never be deleted.
"""
import pytest

from toxagent.domain import scientific_case as sc
from toxagent.domain.scientific_case.dependencies import (
    CLAIM_NEEDS_REASSESSMENT,
    CLAIM_PROPOSED,
    CLAIM_RETRACTED,
    DerivedClaim,
    Impact,
    Justification,
    affected,
    apply_change,
    claim_ref,
    cycles,
    needs_reassessment,
    unresolved_references,
)

SESSION = "ses_" + "1" * 32
CASE_ID = "scase_" + "2" * 32
RUN = "run_" + "6" * 32

ASSAY = "obs_" + "a" * 32
CMAX_V1 = "case_value:free_cmax@v1"
CMAX_V2 = "case_value:free_cmax@v2"


def upd(op, actor="server", **payload):
    return sc.CaseUpdate(op=op, payload=payload, actor=actor,
                         at="2026-10-28T09:00:00+00:00", run_id=RUN)


def opened():
    return sc.apply(None, upd(
        "open", actor="server", case_id=CASE_ID, session_id=SESSION,
        subject_key="analysis:ana_x", subject_refs=["analysis:ana_x"],
        question="Does the measurement contradict the prediction?",
    ))


def _margin_case():
    """The §5.3 example: a margin over an assay measurement and an exposure."""
    case = sc.apply(opened(), upd("add_context", actor="user", key="free_cmax",
                                  value="0.01", unit="µM"))
    return sc.apply(case, upd(
        "record_derived_claim", statement="exposure_margin",
        rule="ratio_same_concentration_unit", inputs=[ASSAY, CMAX_V1],
        assumptions=["exposure_context_compatible"],
    ))


# ── the record §5.3 asks for ──────────────────────────────────────────────

def test_a_derived_claim_records_its_inputs_rule_and_assumptions():
    case = _margin_case()

    claim = case.derived_claims[0]
    assert claim.statement == "exposure_margin"
    assert claim.status == CLAIM_PROPOSED
    justification = claim.justifications[0]
    assert justification.rule == "ratio_same_concentration_unit"
    assert justification.inputs == (ASSAY, CMAX_V1)
    assert justification.assumptions == ("exposure_context_compatible",)


def test_a_claim_without_inputs_is_refused():
    """A claim nothing can reach is a claim no change can ever mark."""
    with pytest.raises(sc.InvalidCaseUpdate) as refused:
        sc.apply(opened(), upd("record_derived_claim", statement="exposure_margin",
                               rule="ratio_same_concentration_unit", inputs=[]))

    assert "needs its inputs" in str(refused.value)


def test_the_researcher_does_not_write_derivations():
    with pytest.raises(sc.InvalidCaseUpdate):
        sc.apply(opened(), upd("record_derived_claim", actor="user", statement="m",
                               rule="r", inputs=[ASSAY]))


def test_a_case_written_before_this_serialises_exactly_as_it_did():
    case = opened()

    assert "derived_claims" not in case.to_dict()
    assert sc.ScientificCaseV1.from_dict(case.to_dict()) == case


def test_a_case_with_claims_round_trips():
    case = _margin_case()

    assert sc.ScientificCaseV1.from_dict(case.to_dict()) == case


# ── §11 turn 3: correcting the exposure ───────────────────────────────────

def test_correcting_a_value_marks_what_was_computed_from_it():
    """§5.3: "Đánh dấu margin và những kết luận dùng margin cần tính/kiểm lại.\""""
    case = sc.apply(_margin_case(), upd("add_context", actor="user", key="free_cmax",
                                        value="0.02", unit="µM"))

    margin = case.derived_claims[0]
    assert margin.status == CLAIM_NEEDS_REASSESSMENT
    assert CMAX_V1 in margin.note
    assert margin.justifications[0].invalidated_by == (CMAX_V1,)


def test_marking_is_not_deleting():
    """The claim and its number stay: the earlier answer was computed from
    them, and a later answer has to be able to say what changed."""
    case = sc.apply(_margin_case(), upd("add_context", actor="user", key="free_cmax",
                                        value="0.02", unit="µM"))

    assert len(case.derived_claims) == 1
    assert case.derived_claims[0].justifications[0].inputs == (ASSAY, CMAX_V1)


def test_the_marking_reaches_claims_that_rest_on_the_margin():
    """A recommendation over the margin is downstream of the exposure, even
    though it never names the exposure."""
    case = sc.apply(_margin_case(), upd(
        "record_derived_claim", statement="progress_to_patch_clamp",
        rule="margin_below_declared_threshold", inputs=[claim_ref("dc1")],
    ))

    case = sc.apply(case, upd("add_context", actor="user", key="free_cmax",
                              value="0.02", unit="µM"))

    assert [c.status for c in case.derived_claims] == [
        CLAIM_NEEDS_REASSESSMENT, CLAIM_NEEDS_REASSESSMENT,
    ]
    assert case.derived_claims[1].justifications[0].invalidated_by == (claim_ref("dc1"),)


def test_an_unrelated_claim_is_left_alone():
    """Preservation accuracy is half of what H2 is measured on (#65)."""
    case = sc.apply(_margin_case(), upd(
        "record_derived_claim", statement="prediction_and_measurement_disagree",
        rule="classification_threshold", inputs=[ASSAY, "obs_" + "b" * 32],
    ))

    case = sc.apply(case, upd("add_context", actor="user", key="free_cmax",
                              value="0.02", unit="µM"))

    assert case.derived_claims[0].status == CLAIM_NEEDS_REASSESSMENT
    assert case.derived_claims[1].status == CLAIM_PROPOSED
    assert case.derived_claims[1].note == ""


def test_the_first_statement_of_a_value_marks_nothing():
    """Nothing cited v1 before it existed; only a correction is a change."""
    case = sc.apply(_margin_case(), upd("add_context", actor="user", key="ic50",
                                        value="0.3", unit="µM"))

    assert case.derived_claims[0].status == CLAIM_PROPOSED


# ── one justification going is not the claim going ────────────────────────

def test_a_claim_with_a_second_justification_keeps_its_status():
    """§5.3: "Nếu một claim còn một justification khác hợp lệ, việc mất một
    premise không được tự động xóa claim." The H2 gold needs this case (#65)."""
    case = sc.apply(_margin_case(), upd(
        "record_derived_claim", statement="exposure_margin",
        rule="ratio_from_reported_margin", inputs=[ASSAY, "evd_" + "c" * 32],
    ))
    assert len(case.derived_claims) == 1  # one claim, two ways to it

    case = sc.apply(case, upd("add_context", actor="user", key="free_cmax",
                              value="0.02", unit="µM"))

    claim = case.derived_claims[0]
    assert claim.status == CLAIM_PROPOSED
    assert "still stands" in claim.note
    assert [j.standing for j in claim.justifications] == [False, True]


def test_a_claim_whose_every_justification_is_hit_needs_reassessment():
    case = sc.apply(_margin_case(), upd(
        "record_derived_claim", statement="exposure_margin",
        rule="ratio_rounded", inputs=[CMAX_V1],
    ))

    case = sc.apply(case, upd("add_context", actor="user", key="free_cmax",
                              value="0.02", unit="µM"))

    assert case.derived_claims[0].status == CLAIM_NEEDS_REASSESSMENT


def test_a_lost_justification_does_not_put_the_claims_below_it_in_doubt():
    """Propagating from a claim that is not itself in doubt would make one
    corrected exposure invalidate the whole case."""
    claims = (
        DerivedClaim(id="dc1", statement="margin", justifications=(
            Justification(id="j1", rule="ratio", inputs=(CMAX_V1,)),
            Justification(id="j2", rule="reported", inputs=("evd_x",)),
        )),
        DerivedClaim(id="dc2", statement="recommendation", justifications=(
            Justification(id="j1", rule="threshold", inputs=(claim_ref("dc1"),)),
        )),
    )

    impacts = {row.claim_id: row.impact for row in affected(claims, (CMAX_V1,))}

    assert impacts["dc1"] is Impact.JUSTIFICATION_LOST
    assert impacts["dc2"] is Impact.UNAFFECTED


def test_an_idempotent_repeat_adds_no_second_justification():
    case = _margin_case()

    again = sc.apply(case, upd(
        "record_derived_claim", statement="exposure_margin",
        rule="ratio_same_concentration_unit", inputs=[ASSAY, CMAX_V1],
        assumptions=["exposure_context_compatible"],
    ))

    assert again is case


def test_a_retracted_claim_is_not_reopened_by_a_change():
    claims = (DerivedClaim(id="dc1", statement="margin", status=CLAIM_RETRACTED,
                           justifications=(Justification(id="j1", rule="r",
                                                         inputs=(CMAX_V1,)),)),)

    assert apply_change(claims, (CMAX_V1,))[0].status == CLAIM_RETRACTED


# ── the other premises ────────────────────────────────────────────────────

def test_a_premise_outside_the_cases_own_values_can_be_marked_too():
    """A re-run predictor snapshot, a withdrawn record, an assumption the
    researcher no longer accepts."""
    case = sc.apply(_margin_case(), upd("reassess_claims", changed=[ASSAY],
                                        note="the snapshot was re-run"))

    assert case.derived_claims[0].status == CLAIM_NEEDS_REASSESSMENT
    assert case.derived_claims[0].note == "the snapshot was re-run"


def test_reassess_claims_needs_the_refs_that_changed():
    with pytest.raises(sc.InvalidCaseUpdate):
        sc.apply(_margin_case(), upd("reassess_claims", changed=[]))


def test_only_the_server_marks_what_a_change_reached():
    with pytest.raises(sc.InvalidCaseUpdate):
        sc.apply(_margin_case(), upd("reassess_claims", actor="model", changed=[ASSAY]))


def test_needs_reassessment_lists_what_is_waiting():
    case = sc.apply(_margin_case(), upd("reassess_claims", changed=[ASSAY]))

    assert [c.id for c in needs_reassessment(case.derived_claims)] == ["dc1"]


# ── soundness of the graph ────────────────────────────────────────────────

def test_an_input_nothing_provides_is_reported_as_unresolved():
    case = _margin_case()

    unresolved = unresolved_references(case.derived_claims, known=[CMAX_V1])

    assert unresolved == (("dc1", ASSAY),)


def test_a_resolvable_graph_reports_nothing():
    case = _margin_case()

    assert unresolved_references(case.derived_claims, known=[CMAX_V1, ASSAY]) == ()


def test_a_claim_that_justifies_itself_is_a_cycle():
    claims = (DerivedClaim(id="dc1", statement="a", justifications=(
        Justification(id="j1", rule="r", inputs=(claim_ref("dc1"),)),),),)

    assert cycles(claims) == (("dc1", "dc1"),)


def test_a_longer_cycle_is_found_once():
    claims = (
        DerivedClaim(id="dc1", statement="a", justifications=(
            Justification(id="j1", rule="r", inputs=(claim_ref("dc2"),)),)),
        DerivedClaim(id="dc2", statement="b", justifications=(
            Justification(id="j1", rule="r", inputs=(claim_ref("dc1"),)),)),
    )

    found = cycles(claims)

    assert len(found) == 1
    assert set(found[0]) == {"dc1", "dc2"}


def test_an_acyclic_graph_has_no_cycles():
    case = sc.apply(_margin_case(), upd(
        "record_derived_claim", statement="recommendation",
        rule="threshold", inputs=[claim_ref("dc1")],
    ))

    assert cycles(case.derived_claims) == ()


def test_a_cycle_does_not_stop_the_change_walk_from_terminating():
    """A cycle is a finding, not a hang: the walk is bounded by the number of
    claims, so an unsound graph still produces an answer."""
    claims = (
        DerivedClaim(id="dc1", statement="a", justifications=(
            Justification(id="j1", rule="r", inputs=(claim_ref("dc2"), CMAX_V1)),)),
        DerivedClaim(id="dc2", statement="b", justifications=(
            Justification(id="j1", rule="r", inputs=(claim_ref("dc1"),)),)),
    )

    impacts = {row.claim_id: row.impact for row in affected(claims, (CMAX_V1,))}

    assert impacts["dc1"] is Impact.NEEDS_REASSESSMENT
    assert impacts["dc2"] is Impact.NEEDS_REASSESSMENT


# ── the researcher can see it ─────────────────────────────────────────────

def test_the_dossier_carries_the_graph_and_what_is_waiting():
    case = sc.apply(_margin_case(), upd("add_context", actor="user", key="free_cmax",
                                        value="0.02", unit="µM"))

    dossier = sc.compile_dossier(case, run_id=RUN, stop_reason="sufficient", answer_id=None)

    assert dossier["claims_needing_reassessment"] == ["dc1"]
    claim = dossier["derived_claims"][0]
    assert claim["statement"] == "exposure_margin"
    assert claim["justifications"][0]["inputs"] == [ASSAY, CMAX_V1]
    assert claim["justifications"][0]["invalidated_by"] == [CMAX_V1]


def test_the_checkpoint_tells_the_next_turn_what_a_changed_premise_left():
    case = sc.apply(_margin_case(), upd("add_context", actor="user", key="free_cmax",
                                        value="0.02", unit="µM"))

    summary = sc.checkpoint_summary(case)

    assert "left waiting" in summary
    assert "dc1 exposure_margin" in summary


def test_a_case_with_nothing_waiting_says_nothing_about_it():
    assert "left waiting" not in sc.checkpoint_summary(_margin_case())
