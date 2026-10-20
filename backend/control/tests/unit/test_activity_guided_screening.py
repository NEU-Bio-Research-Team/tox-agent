from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from toxagent.domain.ids import ANALYSIS, RUN, SESSION, new_id
from toxagent.domain.screening import (
    ActivityApplicability,
    ActivityEstimate,
    ActivityGuidedScreening,
    ActivityIdentityMismatch,
    ActivityModelProvenance,
    ActivitySnapshot,
    ActivitySubject,
    IllegalScreeningTransition,
    PolicyNotApplicable,
    PriorityStatus,
    ScreeningStatus,
    TargetIdentity,
    TargetPriorityPolicy,
    TargetRole,
    UnsupportedTargetRole,
)
from toxagent.predictor.activity_schemas import (
    ActivityPredictionRequest,
    ActivityPredictionResponse,
)


NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)
CONTEXT = "IC50|wild_type"
SEQUENCE_SHA256 = "a" * 64
OTHER_SEQUENCE_SHA256 = "b" * 64


def _snapshot(
    *,
    predicted: float = 8.1,
    uncertainty: float | None = 0.2,
    interval: tuple[float, float] | None = (7.8, 8.4),
    applicability: ActivityApplicability = ActivityApplicability.IN_DOMAIN,
    session_id: str | None = None,
    run_id: str | None = None,
    canonical_smiles: str = "CCO",
    target: TargetIdentity | None = None,
    assay_context: str = CONTEXT,
) -> ActivitySnapshot:
    return ActivitySnapshot.create(
        session_id=session_id or new_id(SESSION),
        run_id=run_id or new_id(RUN),
        input_smiles=canonical_smiles,
        canonical_smiles=canonical_smiles,
        target=target or TargetIdentity(target_id="EGFR", uniprot_id="P00533"),
        assay_context=assay_context,
        estimate=ActivityEstimate(
            predicted_pic50=predicted,
            uncertainty=uncertainty,
            interval_low=interval[0] if interval else None,
            interval_high=interval[1] if interval else None,
        ),
        ligand_applicability=applicability,
        target_applicability=applicability,
        joint_applicability=applicability,
        provenance=ActivityModelProvenance(
            model_id="sair-ligand-protein-v1",
            dataset_id="sair-unavailable-fixture",
            split_id="synthetic-test-only",
        ),
        now=NOW,
    )


def _subject(
    *,
    canonical_smiles: str = "CCO",
    target: TargetIdentity | None = None,
    assay_context: str = CONTEXT,
) -> ActivitySubject:
    return ActivitySubject(
        canonical_smiles=canonical_smiles,
        target=target or TargetIdentity(target_id="EGFR", uniprot_id="P00533"),
        assay_context=assay_context,
    )


def _screening(
    *,
    session_id: str,
    run_id: str,
    subject: ActivitySubject | None = None,
    role: TargetRole = TargetRole.INTENDED_TARGET,
) -> ActivityGuidedScreening:
    return ActivityGuidedScreening.create(
        session_id=session_id,
        run_id=run_id,
        input_smiles="CCO",
        subject=subject or _subject(),
        intended_target_role=role,
        now=NOW,
    )


def _policy(
    *,
    assay_context: str = CONTEXT,
    role: TargetRole = TargetRole.INTENDED_TARGET,
) -> TargetPriorityPolicy:
    return TargetPriorityPolicy(
        policy_id="activity-priority-egfr-v1",
        target_id="EGFR",
        assay_context=assay_context,
        threshold_pic50=7.0,
        threshold_source="synthetic_test_policy",
        max_uncertainty=0.5,
        intended_target_role=role,
    )


def test_policy_prioritises_only_when_the_whole_interval_clears_the_target_threshold():
    decision = _policy().decide(_snapshot(), now=NOW)
    assert decision.status is PriorityStatus.PRIORITISE
    assert "interval_above_target_threshold" in decision.reason_codes


def test_policy_marks_an_interval_wholly_below_threshold_as_low_priority():
    decision = _policy().decide(
        _snapshot(predicted=6.2, interval=(5.9, 6.5)), now=NOW
    )
    assert decision.status is PriorityStatus.LOW_PRIORITY


@pytest.mark.parametrize(
    ("snapshot", "reason"),
    [
        (_snapshot(predicted=7.0, interval=(6.6, 7.4)), "prediction_interval_crosses"),
        (_snapshot(uncertainty=0.8), "uncertainty_above"),
        (_snapshot(uncertainty=None, interval=None), "uncertainty_unavailable"),
        (
            _snapshot(applicability=ActivityApplicability.OUT_OF_DOMAIN),
            "joint_applicability_out_of_domain",
        ),
    ],
)
def test_uncertain_or_out_of_domain_activity_requires_review(snapshot, reason):
    decision = _policy().decide(snapshot, now=NOW)
    assert decision.status is PriorityStatus.REVIEW
    assert any(code.startswith(reason) for code in decision.reason_codes)


def test_policy_cannot_be_reused_for_a_different_target():
    other = _snapshot(target=TargetIdentity(target_id="DRD2"))
    with pytest.raises(PolicyNotApplicable, match="cannot decide snapshot"):
        _policy().decide(other, now=NOW)


def test_policy_cannot_apply_its_threshold_to_another_assay_context():
    """A threshold is calibrated per context; reusing it across contexts
    compares two different quantities."""
    measured_elsewhere = _snapshot(assay_context="Ki|binding")
    with pytest.raises(PolicyNotApplicable, match="calibrated for"):
        _policy().decide(measured_elsewhere, now=NOW)


def test_no_prioritisation_rule_exists_for_an_off_target_role():
    """For an anti-target, "high pIC50 deserves more screening" argues the
    opposite way, so the policy refuses instead of inverting silently."""
    with pytest.raises(UnsupportedTargetRole, match="intended target"):
        _policy(role=TargetRole.OFF_TARGET).decide(_snapshot(), now=NOW)


def test_prioritised_screening_runs_toxicity_and_links_the_existing_analysis():
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(session_id=session_id, run_id=run_id).start_activity(now=NOW)
    snapshot = _snapshot(session_id=session_id, run_id=run_id)
    screening = screening.record_activity(snapshot, now=NOW)
    screening = screening.record_decision(_policy().decide(snapshot, now=NOW), now=NOW)
    assert screening.status is ScreeningStatus.PRIORITISED

    screening = screening.start_toxicity(now=NOW)
    analysis_id = new_id(ANALYSIS)
    screening = screening.complete_toxicity(analysis_id, now=NOW)
    assert screening.status is ScreeningStatus.COMPLETED
    assert screening.toxicity_analysis_id == analysis_id
    assert screening.is_terminal


def test_low_priority_cannot_silently_start_toxicity_or_look_negative():
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(session_id=session_id, run_id=run_id).start_activity(now=NOW)
    snapshot = _snapshot(
        predicted=6.2,
        interval=(5.9, 6.5),
        session_id=session_id,
        run_id=run_id,
    )
    screening = screening.record_activity(snapshot, now=NOW)
    screening = screening.record_decision(_policy().decide(snapshot, now=NOW), now=NOW)
    assert screening.status is ScreeningStatus.LOW_PRIORITY
    with pytest.raises(ValueError, match="requires an actor"):
        screening.start_toxicity(now=NOW)

    screening = screening.complete_without_toxicity(
        reason="activity interval below target-specific threshold", now=NOW
    )
    assert screening.status is ScreeningStatus.COMPLETED_WITH_TOXICITY_NOT_RUN
    assert screening.toxicity_analysis_id is None
    assert screening.toxicity_not_run_reason


def test_review_and_low_priority_need_explicit_authorisation_to_continue():
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(session_id=session_id, run_id=run_id).start_activity(now=NOW)
    snapshot = _snapshot(
        predicted=7.0,
        interval=(6.6, 7.4),
        session_id=session_id,
        run_id=run_id,
    )
    screening = screening.record_activity(snapshot, now=NOW)
    screening = screening.record_decision(_policy().decide(snapshot, now=NOW), now=NOW)
    with pytest.raises(ValueError, match="requires an actor"):
        screening.start_toxicity(now=NOW)
    screening = screening.start_toxicity(
        now=NOW,
        authorised_by="reviewer@example.org",
        authorisation_reason="manual review accepted the uncertain estimate",
    )
    assert screening.status is ScreeningStatus.TOXICITY_RUNNING
    assert screening.toxicity_authorised_by == "reviewer@example.org"
    assert screening.toxicity_authorisation_reason


def test_manual_toxicity_authorisation_requires_both_actor_and_reason():
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(session_id=session_id, run_id=run_id).start_activity(now=NOW)
    snapshot = _snapshot(
        predicted=6.2,
        interval=(5.9, 6.5),
        session_id=session_id,
        run_id=run_id,
    )
    screening = screening.record_activity(snapshot, now=NOW)
    screening = screening.record_decision(_policy().decide(snapshot, now=NOW), now=NOW)

    with pytest.raises(ValueError, match="requires a reason"):
        screening.start_toxicity(now=NOW, authorised_by="reviewer@example.org")


def test_activity_failure_is_terminal_and_never_becomes_low_priority():
    screening = _screening(
        session_id=new_id(SESSION), run_id=new_id(RUN)
    ).start_activity(now=NOW)
    screening = screening.fail_activity(failure_code="activity_model_unavailable", now=NOW)
    assert screening.status is ScreeningStatus.ACTIVITY_FAILED
    assert screening.is_terminal
    with pytest.raises(IllegalScreeningTransition):
        screening.start_toxicity(now=NOW)


def test_activity_wire_contract_is_strict_and_keeps_regression_semantics():
    payload = {
        "input_smiles": "CCO",
        "canonical_smiles": "CCO",
        "target": {"target_id": "EGFR", "uniprot_id": "P00533"},
        "assay_context": "IC50|wild_type",
        "prediction": {
            "predicted_pic50": 8.1,
            "uncertainty": 0.2,
            "prediction_interval": [7.8, 8.4],
            "model_id": "sair-ligand-protein-v1",
        },
        "applicability": {
            "ligand": "in_domain",
            "target": "in_domain",
            "joint": "in_domain",
            "reasons": [],
        },
        "provenance": {
            "model_id": "sair-ligand-protein-v1",
            "dataset_id": "sair-v1",
            "artifact_hashes": [],
        },
    }
    parsed = ActivityPredictionResponse.model_validate(payload)
    assert parsed.prediction.predicted_pic50 == 8.1

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ActivityPredictionResponse.model_validate({**payload, "toxicity_score": 0.1})


def test_activity_wire_contract_rejects_mismatched_model_provenance():
    with pytest.raises(ValidationError, match="model_id must match"):
        ActivityPredictionResponse.model_validate(
            {
                "input_smiles": "CCO",
                "canonical_smiles": "CCO",
                "target": {"target_id": "EGFR"},
                "assay_context": "IC50|wild_type",
                "prediction": {
                    "predicted_pic50": 8.1,
                    "model_id": "model-a",
                },
                "applicability": {
                    "ligand": "unknown",
                    "target": "unknown",
                    "joint": "unknown",
                },
                "provenance": {"model_id": "model-b", "dataset_id": "fixture"},
            }
        )


def test_record_activity_rejects_a_snapshot_for_another_ligand():
    """The probe from the review: session, run and target all line up, and the
    snapshot is still about a different molecule."""
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(session_id=session_id, run_id=run_id).start_activity(now=NOW)
    other_ligand = _snapshot(
        session_id=session_id, run_id=run_id, canonical_smiles="CCN"
    )

    with pytest.raises(ActivityIdentityMismatch) as raised:
        screening.record_activity(other_ligand, now=NOW)

    assert raised.value.mismatches == ("canonical_smiles",)
    assert screening.status is ScreeningStatus.ACTIVITY_RUNNING
    assert screening.activity_snapshot_id is None


def test_record_activity_rejects_a_snapshot_for_another_target_sequence():
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(
        session_id=session_id,
        run_id=run_id,
        subject=_subject(
            target=TargetIdentity(
                target_id="EGFR",
                uniprot_id="P00533",
                sequence_sha256=SEQUENCE_SHA256,
            )
        ),
    ).start_activity(now=NOW)
    variant = _snapshot(
        session_id=session_id,
        run_id=run_id,
        target=TargetIdentity(
            target_id="EGFR",
            uniprot_id="P00533",
            sequence_sha256=OTHER_SEQUENCE_SHA256,
        ),
    )

    with pytest.raises(ActivityIdentityMismatch) as raised:
        screening.record_activity(variant, now=NOW)

    assert raised.value.mismatches == ("sequence_sha256",)


def test_record_activity_rejects_a_snapshot_measured_in_another_assay_context():
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(session_id=session_id, run_id=run_id).start_activity(now=NOW)
    elsewhere = _snapshot(
        session_id=session_id, run_id=run_id, assay_context="Ki|binding"
    )

    with pytest.raises(ActivityIdentityMismatch) as raised:
        screening.record_activity(elsewhere, now=NOW)

    assert raised.value.mismatches == ("assay_context",)


def test_record_activity_accepts_a_sequence_the_caller_never_declared():
    """A caller may open a screening knowing only the target id and have the
    predictor resolve the sequence.  That is enrichment, not a mismatch."""
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(
        session_id=session_id,
        run_id=run_id,
        subject=_subject(target=TargetIdentity(target_id="EGFR")),
    ).start_activity(now=NOW)
    resolved = _snapshot(
        session_id=session_id,
        run_id=run_id,
        target=TargetIdentity(
            target_id="EGFR", uniprot_id="P00533", sequence_sha256=SEQUENCE_SHA256
        ),
    )

    screening = screening.record_activity(resolved, now=NOW)
    assert screening.status is ScreeningStatus.ACTIVITY_COMPLETED


def test_record_decision_rejects_a_threshold_from_another_assay_context():
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(session_id=session_id, run_id=run_id).start_activity(now=NOW)
    snapshot = _snapshot(session_id=session_id, run_id=run_id)
    screening = screening.record_activity(snapshot, now=NOW)

    foreign = replace(
        _policy().decide(snapshot, now=NOW), assay_context="Ki|binding"
    )
    with pytest.raises(ActivityIdentityMismatch, match="assay_context"):
        screening.record_decision(foreign, now=NOW)


def test_record_decision_rejects_a_decision_that_assumes_another_target_role():
    session_id, run_id = new_id(SESSION), new_id(RUN)
    screening = _screening(session_id=session_id, run_id=run_id).start_activity(now=NOW)
    snapshot = _snapshot(session_id=session_id, run_id=run_id)
    screening = screening.record_activity(snapshot, now=NOW)

    foreign = replace(
        _policy().decide(snapshot, now=NOW),
        intended_target_role=TargetRole.OFF_TARGET,
    )
    with pytest.raises(ActivityIdentityMismatch, match="intended_target_role"):
        screening.record_decision(foreign, now=NOW)


def test_a_decision_records_the_context_and_role_it_assumed():
    decision = _policy().decide(_snapshot(), now=NOW)
    assert decision.assay_context == CONTEXT
    assert decision.intended_target_role is TargetRole.INTENDED_TARGET


def test_activity_request_must_name_the_assay_context():
    """The response requires a context, so the request has to choose one
    instead of letting the server pick unseen."""
    with pytest.raises(ValidationError, match="assay_context"):
        ActivityPredictionRequest.model_validate(
            {"smiles": "CCO", "target": {"target_id": "EGFR"}}
        )

    parsed = ActivityPredictionRequest.model_validate(
        {"smiles": "CCO", "target": {"target_id": "EGFR"}, "assay_context": CONTEXT}
    )
    assert parsed.assay_context == CONTEXT
