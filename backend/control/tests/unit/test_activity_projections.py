"""The activity artifacts as things a claim can cite.

Three properties matter here: a pIC50 keeps its own scale and never becomes an
endpoint-keyed probability, a policy decision is marked as a rule's output
rather than a measurement, and a stage that did not run says so explicitly.
"""
from __future__ import annotations

from datetime import datetime, timezone

from toxagent.application.activity.projections import (
    ACTIVITY_LIMITATIONS,
    NOT_ADMITTED_LIMITATION,
    POLICY_LIMITATION,
    TOXICITY_NOT_RUN_LIMITATION,
    activity_observation,
    freeze_screening,
    prioritisation_observation,
    screening_state_observation,
)
from toxagent.domain.ids import RUN, SESSION, new_id
from toxagent.domain.observation import ObservationKind, Producer
from toxagent.domain.screening import (
    ActivityApplicability,
    ActivityEstimate,
    ActivityGuidedScreening,
    ActivityModelProvenance,
    ActivitySnapshot,
    ActivitySubject,
    ScreeningStatus,
    TargetIdentity,
    TargetPriorityPolicy,
    TargetRole,
)


NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)
CONTEXT = "IC50|wild_type"


def _target() -> TargetIdentity:
    return TargetIdentity(target_id="EGFR", uniprot_id="P00533")


def _snapshot(
    *, session_id: str, run_id: str, predicted: float = 8.1, interval=(7.8, 8.4)
) -> ActivitySnapshot:
    return ActivitySnapshot.create(
        session_id=session_id,
        run_id=run_id,
        input_smiles="CCO",
        canonical_smiles="CCO",
        target=_target(),
        assay_context=CONTEXT,
        estimate=ActivityEstimate(
            predicted_pic50=predicted,
            uncertainty=0.2,
            interval_low=interval[0] if interval else None,
            interval_high=interval[1] if interval else None,
        ),
        ligand_applicability=ActivityApplicability.IN_DOMAIN,
        target_applicability=ActivityApplicability.IN_DOMAIN,
        joint_applicability=ActivityApplicability.IN_DOMAIN,
        provenance=ActivityModelProvenance(
            model_id="sair-descriptor-hgb-b0",
            dataset_id="sair-phase1",
            split_id="exact_sequence_cold_target",
        ),
        now=NOW,
    )


def _screening(*, session_id: str, run_id: str) -> ActivityGuidedScreening:
    return ActivityGuidedScreening.create(
        session_id=session_id,
        run_id=run_id,
        input_smiles="CCO",
        subject=ActivitySubject(
            canonical_smiles="CCO", target=_target(), assay_context=CONTEXT
        ),
        intended_target_role=TargetRole.INTENDED_TARGET,
        now=NOW,
    )


def _policy() -> TargetPriorityPolicy:
    return TargetPriorityPolicy(
        policy_id="activity-priority-egfr-v1",
        target_id="EGFR",
        assay_context=CONTEXT,
        threshold_pic50=7.0,
        threshold_source="synthetic_test_policy",
        max_uncertainty=0.5,
    )


def _run(
    *, predicted: float = 8.1, interval: tuple[float, float] = (7.8, 8.4)
) -> tuple[ActivityGuidedScreening, ActivitySnapshot, object]:
    """A screening taken as far as the policy decision.

    The default estimate clears the threshold, so the screening ends
    ``prioritised``; pass an interval below it for the ``low_priority`` branch.
    """
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(session_id=session_id, run_id=run_id).start_activity(now=NOW)
    snapshot = _snapshot(
        session_id=session_id, run_id=run_id, predicted=predicted, interval=interval
    )
    screening = screening.record_activity(snapshot, now=NOW)
    decision = _policy().decide(snapshot, now=NOW)
    screening = screening.record_decision(decision, now=NOW)
    return screening, snapshot, decision


# ── the estimate ──────────────────────────────────────────────────────────

def test_a_pic50_keeps_its_own_scale_and_never_becomes_a_probability():
    screening, snapshot, _ = _run()
    observation = activity_observation(
        snapshot, screening_id=screening.id, model_admitted=False, now=NOW
    )

    estimate = observation.canonical_payload["estimate"]
    assert estimate["predicted_pic50"] == 8.1
    assert estimate["unit"] == "pIC50"
    # no endpoint enum, and nothing calling this a probability or a score
    assert "endpoint" not in observation.canonical_payload
    flat = str(observation.canonical_payload)
    assert "probability" not in flat
    assert observation.kind is ObservationKind.ACTIVITY_ESTIMATE
    assert observation.producer is Producer.PREDICTOR


def test_a_pic50_outside_zero_to_one_survives_the_projection():
    session_id, run_id = new_id(SESSION), new_id(RUN)
    snapshot = _snapshot(
        session_id=session_id, run_id=run_id, predicted=11.2, interval=(10.9, 11.5)
    )
    observation = activity_observation(
        snapshot, screening_id=new_id("scr"), model_admitted=False, now=NOW
    )
    assert observation.value_at("estimate.predicted_pic50") == 11.2


def test_the_estimate_carries_the_subject_it_was_measured_on():
    screening, snapshot, _ = _run()
    observation = activity_observation(
        snapshot, screening_id=screening.id, model_admitted=False, now=NOW
    )
    subject = observation.canonical_payload["subject"]
    assert subject["canonical_smiles"] == "CCO"
    assert subject["target"]["target_id"] == "EGFR"
    assert subject["assay_context"] == CONTEXT
    assert observation.canonical_payload["screening_id"] == screening.id
    assert observation.provenance["activity_snapshot_id"] == snapshot.id
    assert observation.provenance["model_id"] == "sair-descriptor-hgb-b0"


def test_an_unadmitted_model_adds_that_as_a_required_limitation():
    screening, snapshot, _ = _run()
    not_admitted = activity_observation(
        snapshot, screening_id=screening.id, model_admitted=False, now=NOW
    )
    admitted = activity_observation(
        snapshot, screening_id=screening.id, model_admitted=True, now=NOW
    )
    assert NOT_ADMITTED_LIMITATION in not_admitted.required_limitations
    assert NOT_ADMITTED_LIMITATION not in admitted.required_limitations
    for code in ACTIVITY_LIMITATIONS:
        assert code in admitted.required_limitations


def test_the_interval_says_what_kind_of_coverage_it_has():
    """A marginal interval is not a per-candidate guarantee, so the projection
    names what it is rather than leaving the reader to assume."""
    screening, snapshot, _ = _run()
    observation = activity_observation(
        snapshot, screening_id=screening.id, model_admitted=False, now=NOW
    )
    estimate = observation.canonical_payload["estimate"]
    assert estimate["interval_kind"] == "marginal_coverage_on_validation_residuals"
    assert "activity_interval_is_marginal_coverage" in observation.required_limitations


# ── the decision ──────────────────────────────────────────────────────────

def test_a_policy_decision_is_projected_as_a_rule_not_a_measurement():
    screening, _, decision = _run()
    observation = prioritisation_observation(
        decision,
        session_id=screening.session_id,
        run_id=screening.run_id,
        screening_id=screening.id,
        now=NOW,
    )
    assert observation.producer is Producer.POLICY
    assert observation.kind is ObservationKind.PRIORITISATION_DECISION
    assert POLICY_LIMITATION in observation.required_limitations
    policy = observation.canonical_payload["policy"]
    assert policy["threshold_pic50"] == 7.0
    assert policy["threshold_source"] == "synthetic_test_policy"
    assert policy["assay_context"] == CONTEXT
    assert policy["intended_target_role"] == "intended_target"
    assert observation.canonical_payload["reason_codes"]


# ── the stage status ──────────────────────────────────────────────────────

def test_a_toxicity_stage_that_did_not_run_says_so_with_a_reason():
    screening, snapshot, decision = _run(predicted=6.2, interval=(5.9, 6.5))
    assert screening.status is ScreeningStatus.LOW_PRIORITY
    screening = screening.complete_without_toxicity(
        reason="activity interval below target-specific threshold", now=NOW
    )
    observation = screening_state_observation(
        screening, run_id=screening.run_id, now=NOW
    )

    toxicity = observation.canonical_payload["toxicity"]
    assert toxicity["run"] is False
    assert toxicity["not_run_reason"]
    assert toxicity["analysis_id"] is None
    assert TOXICITY_NOT_RUN_LIMITATION in observation.required_limitations
    assert observation.producer is Producer.WORKFLOW


def test_a_completed_toxicity_stage_is_not_flagged_as_not_run():
    screening, _, _ = _run()
    screening = screening.start_toxicity(now=NOW)
    screening = screening.complete_toxicity(new_id("ana"), now=NOW)
    observation = screening_state_observation(
        screening, run_id=screening.run_id, now=NOW
    )
    assert observation.canonical_payload["toxicity"]["run"] is True
    assert TOXICITY_NOT_RUN_LIMITATION not in observation.required_limitations


def test_a_failed_toxicity_stage_is_distinguished_from_one_never_started():
    screening, _, _ = _run()
    failed = screening.start_toxicity(now=NOW).fail_toxicity(
        failure_code="predictor_unavailable", now=NOW
    )
    payload = screening_state_observation(
        failed, run_id=failed.run_id, now=NOW
    ).canonical_payload
    assert payload["status"] == ScreeningStatus.TOXICITY_FAILED.value
    assert payload["toxicity"]["run"] is False
    assert payload["toxicity"]["failure_code"] == "predictor_unavailable"
    assert payload["toxicity"]["not_run_reason"] is None


# ── replay ────────────────────────────────────────────────────────────────

def test_a_frozen_screening_is_identical_across_runs():
    """An observation id is minted per run; a frozen record has to compare
    equal across runs or a manifest hash means nothing."""
    screening, snapshot, decision = _run()
    first = freeze_screening(
        screening=screening,
        snapshot=snapshot,
        decision=decision,
        model_admitted=False,
        run_id=screening.run_id,
        now=NOW,
    )
    second = freeze_screening(
        screening=screening,
        snapshot=snapshot,
        decision=decision,
        model_admitted=False,
        run_id=screening.run_id,
        now=datetime(2026, 11, 1, tzinfo=timezone.utc),
    )
    assert first["content_sha256"] == second["content_sha256"]
    assert "observation_id" not in str(first)


def test_a_frozen_screening_states_the_stages_that_did_not_happen():
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(session_id=session_id, run_id=run_id).start_activity(now=NOW)
    screening = screening.fail_activity(failure_code="activity_model_unavailable", now=NOW)

    record = freeze_screening(
        screening=screening,
        snapshot=None,
        decision=None,
        model_admitted=False,
        run_id=run_id,
        now=NOW,
    )
    assert record["activity_estimate"] is None
    assert record["prioritisation_decision"] is None
    assert record["screening_state"]["activity_failure_code"] == "activity_model_unavailable"
    assert TOXICITY_NOT_RUN_LIMITATION in record["required_limitations"]


def test_a_frozen_screening_collects_every_required_limitation():
    screening, snapshot, decision = _run()
    record = freeze_screening(
        screening=screening,
        snapshot=snapshot,
        decision=decision,
        model_admitted=False,
        run_id=screening.run_id,
        now=NOW,
    )
    for code in (*ACTIVITY_LIMITATIONS, NOT_ADMITTED_LIMITATION, POLICY_LIMITATION):
        assert code in record["required_limitations"], code
