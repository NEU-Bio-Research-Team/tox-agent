"""A provider that could not answer is not evidence that nothing is there (#70).

§6.3: "Provider failure và thiếu bằng chứng khoa học là hai kết quả khác
nhau." Before this, both ended a run as `insufficient_evidence`, so a grader
reading a run record could not tell a finding about the science from a finding
about the product.
"""
import pytest

from toxagent.domain import decision_state as ds
from toxagent.domain.investigation_gap import BLOCKING_GAPS, Gap, is_blocking
from toxagent.domain.run_result import KnowledgeOutcome, knowledge_outcome_of

SESSION = "ses_" + "1" * 32
RUN = "run_" + "6" * 32


BUDGET = {"max_tool_calls": 24, "max_searches": 2, "max_evidence_reads": 8}

QUESTION = "Is there a measured IC50 in this context?"


def _answered(state, relation="insufficient"):
    """A run that produced an answer. A run that produced none is `blocked`
    before any of this -- that is the runtime axis, and it is already right.
    These stops are for a run that *did* answer and still left something
    open: the answer is where it says what it could not reach."""
    return ds.apply_answer(state, relations=[
        {"proposition": QUESTION, "source_class": "external_experimental",
         "source_id": "evd_" + "a" * 32, "relation": relation},
    ], is_fallback=False, candidate_generation=1)


def _state(budget=None):
    state = ds.initial(session_id=SESSION, run_id=RUN, goal="Does the assay contradict it?",
                       subject_refs=["analysis:ana_x"], budget_snapshot=budget or BUDGET)
    return ds.apply_plan(state, [
        {"question": QUESTION, "required_sources": ["evidence"]},
    ])


# ── recording the refusal ─────────────────────────────────────────────────

def test_an_access_failure_is_recorded_rather_than_inferred():
    """An empty result and a refused request look identical afterwards and
    mean opposite things."""
    state = ds.record_access_failure(_state(), provider="europepmc",
                                     asked_for="fullTextXML for PMC123",
                                     reason="http_503")

    assert state.access_failures == ({
        "provider": "europepmc", "asked_for": "fullTextXML for PMC123",
        "reason": "http_503", "gap": "provider_or_corpus",
    },)


def test_the_same_refusal_twice_is_recorded_once():
    state = ds.record_access_failure(_state(), provider="europepmc",
                                     asked_for="x", reason="http_503")
    again = ds.record_access_failure(state, provider="europepmc",
                                     asked_for="x", reason="http_503")

    assert again is state


def test_a_state_with_no_failures_serialises_as_it_did():
    state = _state()

    assert "access_failures" not in state.to_dict()
    assert ds.DecisionSupportStateV1.from_dict(state.to_dict()) == state


def test_failures_survive_a_round_trip():
    state = ds.record_access_failure(_state(), provider="chembl",
                                     asked_for="activities for CHEMBL1220",
                                     reason="breaker_open")

    assert ds.DecisionSupportStateV1.from_dict(state.to_dict()) == state


# ── the stop it produces ──────────────────────────────────────────────────

def test_a_run_a_provider_blocked_stops_as_provider_failure():
    state = ds.record_access_failure(_state(), provider="europepmc",
                                     asked_for="fullTextXML", reason="http_503")

    final = ds.finalize(_answered(state), run_status="completed")

    assert final.stop_reason == ds.StopReason.PROVIDER_FAILURE.value


def test_a_run_that_simply_found_nothing_still_stops_as_insufficient():
    """The existing meaning is untouched: this is the finding about the
    science, and no provider refused anything."""
    final = ds.finalize(_answered(_state()), run_status="completed")

    assert final.stop_reason == ds.StopReason.INSUFFICIENT_EVIDENCE.value


def test_budget_exhaustion_still_wins():
    """A run that also ran out of budget stopped for that: it never got to
    find out whether the provider would have answered the next question."""
    state = _state(budget={"max_searches": 1})
    state = ds.apply_tool_result(state, tool_name="search_toxicology_evidence",
                                 status="completed")
    state = ds.record_access_failure(state, provider="europepmc", asked_for="x",
                                     reason="http_503")

    final = ds.finalize(_answered(state), run_status="completed")

    assert final.stop_reason == ds.StopReason.BUDGET_EXHAUSTED.value


def test_a_resolved_run_is_not_reported_as_a_provider_failure():
    """One provider refusing a full text while the question got settled from
    the abstract is not a failed run."""
    state = ds.record_access_failure(_state(), provider="europepmc", asked_for="x",
                                     reason="http_404")
    state = _answered(state, relation="supports")

    final = ds.finalize(state, run_status="completed")

    assert final.stop_reason == ds.StopReason.SUFFICIENT.value


def test_a_final_state_does_not_take_more_failures():
    final = ds.finalize(_answered(_state()), run_status="completed")

    assert ds.record_access_failure(final, provider="p", asked_for="x", reason="y") is final


def test_the_knowledge_axis_does_not_call_it_unsettled_science():
    """evidence_insufficient says the science is unsettled. The product not
    being able to look is the execution axis's business, and it already has
    ExecutionOutcome.PROVIDER_FAILED for it."""
    assert knowledge_outcome_of("provider_failure", published=False) is KnowledgeOutcome.NONE
    assert knowledge_outcome_of("insufficient_evidence", published=False) is \
        KnowledgeOutcome.EVIDENCE_INSUFFICIENT


# ── the gap vocabulary ────────────────────────────────────────────────────

def test_the_seven_gaps_of_the_proposal_are_named():
    assert {g.value for g in Gap} == {
        "identity_or_scope", "no_relevant_evidence", "span_not_read",
        "measurement_unresolved", "disagreement_unexplained",
        "only_researcher_has_it", "provider_or_corpus",
    }


def test_a_provider_gap_is_not_a_missing_evidence_gap():
    """The distinction the whole module is for: one is about the corpus, the
    other about the science."""
    assert Gap.PROVIDER_OR_CORPUS is not Gap.NO_RELEVANT_EVIDENCE
    assert is_blocking(Gap.PROVIDER_OR_CORPUS)
    assert not is_blocking(Gap.NO_RELEVANT_EVIDENCE)


def test_the_two_gaps_no_further_work_can_close_are_the_blocking_ones():
    assert BLOCKING_GAPS == {Gap.ONLY_RESEARCHER_HAS_IT, Gap.PROVIDER_OR_CORPUS}
    assert not is_blocking("something_else")


def test_an_unknown_gap_is_refused_at_the_door():
    with pytest.raises(ValueError):
        ds.record_access_failure(_state(), provider="p", asked_for="x", reason="y",
                                 gap="made_up")
