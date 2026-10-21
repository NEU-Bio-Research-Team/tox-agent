"""Project the activity workflow's artifacts into observations a claim can cite.

The grounded-answer resolver only lets a claim point at an :class:`Observation`.
Activity has three artifacts worth citing -- the estimate, the policy decision
over it, and the stage status -- and each one needs a different projection:

* A **pIC50 is not a probability.** It has no endpoint enum and no [0, 1]
  bound, so it gets its own observation kind rather than being pushed through
  the toxicity prediction shape (ADR 0013).
* A **policy decision is the output of a rule**, not a second measurement, so
  its producer is ``POLICY`` and its projection names the threshold, the
  threshold's source and the policy version that produced it.
* **A stage that did not run is a fact**, not a missing field. Reading
  ``toxicity_not_run`` as a negative result is on the critical-error list, so
  the projection states ``toxicity_run: false`` with a reason and carries a
  required limitation saying what it does not mean.

Every projection keeps the domain ids (``act_*``, ``pri_*``, ``scr_*``) in its
provenance, so a cited number is traceable back to the artifact that produced
it, and is a pure function of its input: the same frozen snapshot projects to
the same canonical payload and the same content hash, which is what lets the
agent be developed on replayed snapshots before any model is admitted.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ...domain.activity import ActivitySnapshot
from ...domain.analysis import AnalysisSnapshot
from ...domain.answer import LimitationCode
from ...domain.observation import Observation, ObservationKind, Producer
from ...domain.prioritisation import PrioritisationDecision
from ...domain.provenance import content_sha256
from ...domain.screening import ActivityGuidedScreening, ScreeningStatus

ACTIVITY_SCHEMA_VERSION = "activity-estimate-v1"
PRIORITISATION_SCHEMA_VERSION = "prioritisation-decision-v1"
SCREENING_SCHEMA_VERSION = "screening-state-v1"

#: A marginal prediction interval is not a per-candidate guarantee (L7), and a
#: pIC50 is not a statement about therapeutic effect (L9). The codes are the
#: answer vocabulary's own, so an observation cannot require a limitation an
#: answer has no way to declare.
ACTIVITY_LIMITATIONS: tuple[str, ...] = (
    LimitationCode.ACTIVITY_INTERVAL_IS_MARGINAL_COVERAGE.value,
    LimitationCode.PIC50_NOT_THERAPEUTIC_EFFICACY.value,
)
#: Stated whenever the artifact behind the estimate has not passed admission.
NOT_ADMITTED_LIMITATION = LimitationCode.ACTIVITY_MODEL_NOT_ADMITTED.value
#: A rule's output is not an independent measurement of anything.
POLICY_LIMITATION = LimitationCode.POLICY_DECISION_NOT_A_MEASUREMENT.value
#: The one the critical-error list exists for.
TOXICITY_NOT_RUN_LIMITATION = LimitationCode.TOXICITY_NOT_RUN_NOT_NEGATIVE.value

#: Statuses in which the toxicity stage has not produced a result.
_TOXICITY_NOT_RUN = frozenset(
    {
        ScreeningStatus.CREATED,
        ScreeningStatus.ACTIVITY_RUNNING,
        ScreeningStatus.ACTIVITY_COMPLETED,
        ScreeningStatus.ACTIVITY_FAILED,
        ScreeningStatus.PRIORITISED,
        ScreeningStatus.REVIEW_REQUIRED,
        ScreeningStatus.LOW_PRIORITY,
        ScreeningStatus.TOXICITY_RUNNING,
        ScreeningStatus.TOXICITY_FAILED,
        ScreeningStatus.COMPLETED_WITH_TOXICITY_NOT_RUN,
    }
)


TOXICITY_SNAPSHOT_SCHEMA_VERSION = "frozen-toxicity-snapshot-v1"


def toxicity_snapshot_payload(analysis: AnalysisSnapshot) -> dict[str, Any]:
    """The toxicity analysis a screening's toxicity stage produced, frozen
    with the screening so a replay can read it without calling the predictor
    (master plan §5.2). The analysis id is the frozen run's: it does not exist
    in a replaying session, which gets a session-local copy linked to it."""
    return {
        "schema_version": TOXICITY_SNAPSHOT_SCHEMA_VERSION,
        "origin_analysis_id": analysis.id,
        "input_smiles": analysis.input_smiles,
        "canonical_smiles": analysis.canonical_smiles,
        "requested_endpoints": list(analysis.requested_endpoints),
        "predictor_response": analysis.predictor_response,
        "provenance": {
            "base_url_id": analysis.provenance.base_url_id,
            "service_version": analysis.provenance.service_version,
            "git_commit": analysis.provenance.git_commit,
            "artifact_hashes": list(analysis.provenance.artifact_hashes),
        },
        "policy_snapshot": analysis.policy_snapshot,
        "content_sha256": analysis.content_sha256,
    }


def freeze_screening(
    *,
    screening: ActivityGuidedScreening,
    snapshot: ActivitySnapshot | None,
    decision: PrioritisationDecision | None,
    model_admitted: bool,
    run_id: str,
    now: datetime,
    toxicity: AnalysisSnapshot | None = None,
    ligand_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """One screening as the citable artifacts a frozen eval run replays from.

    Only the canonical payloads and their content hashes go in, never the
    observation ids: an id is minted per run, while a frozen record has to
    compare equal across runs for a manifest to mean anything. A stage that
    did not happen is ``None`` here and says so, rather than being left out.

    ``toxicity`` is the analysis the toxicity stage produced, when it ran. It
    is frozen with the screening only when given, so a record frozen without
    it keeps the hash it always had.
    """
    if toxicity is not None:
        if screening.toxicity_analysis_id != toxicity.id:
            raise ValueError("the toxicity analysis is not the one this screening completed with")
        if toxicity.canonical_smiles != screening.subject.canonical_smiles:
            raise ValueError("the toxicity analysis is of a different molecule than the screening")
    activity = (
        activity_canonical_payload(
            snapshot, screening_id=screening.id, model_admitted=model_admitted
        )
        if snapshot is not None
        else None
    )
    prioritisation = (
        prioritisation_canonical_payload(decision, screening_id=screening.id)
        if decision is not None
        else None
    )
    state = screening_state_canonical_payload(screening)
    record = {
        "schema_version": "frozen-screening-v1",
        "run_id": run_id,
        "screening_state": state,
        "activity_estimate": activity,
        "prioritisation_decision": prioritisation,
        "required_limitations": sorted(
            set(
                (ACTIVITY_LIMITATIONS if activity is not None else ())
                + ((NOT_ADMITTED_LIMITATION,) if activity is not None and not model_admitted else ())
                + ((POLICY_LIMITATION,) if prioritisation is not None else ())
                + ((TOXICITY_NOT_RUN_LIMITATION,) if not state["toxicity"]["run"] else ())
            )
        ),
        "frozen_at": now.isoformat(),
    }
    if toxicity is not None:
        record["toxicity_snapshot"] = toxicity_snapshot_payload(toxicity)
    if ligand_names:
        # AG-EVIDENCE: what the ligand is called, as the screening's owner
        # recorded it -- the server-held names a literature search about this
        # subject is judged against. Absent, only identifiers can match.
        record["subject_names"] = [n.strip() for n in ligand_names if n.strip()]
    record["content_sha256"] = content_sha256(
        {k: v for k, v in record.items() if k != "frozen_at"}
    )
    return record


def _subject(snapshot: ActivitySnapshot) -> dict[str, Any]:
    """Canonical ligand, target identity and assay context, as one block.

    A ligand on two targets is two subjects; a ligand in two assay contexts is
    two subjects. Carrying them together is what keeps a cited estimate from
    being reused outside the scope it was measured in.
    """
    return {
        "canonical_smiles": snapshot.canonical_smiles,
        "input_smiles": snapshot.input_smiles,
        "target": snapshot.target.to_dict(),
        "assay_context": snapshot.assay_context,
    }


def activity_canonical_payload(
    snapshot: ActivitySnapshot, *, screening_id: str, model_admitted: bool
) -> dict[str, Any]:
    return {
        "activity_snapshot_id": snapshot.id,
        "screening_id": screening_id,
        "subject": _subject(snapshot),
        # pIC50 keeps its own name and its own scale. There is no endpoint key
        # and no probability here, by design.
        "estimate": {
            "predicted_pic50": snapshot.estimate.predicted_pic50,
            "unit": "pIC50",
            "uncertainty": snapshot.estimate.uncertainty,
            "prediction_interval": (
                [snapshot.estimate.interval_low, snapshot.estimate.interval_high]
                if snapshot.estimate.has_interval
                else None
            ),
            "interval_level": 0.9 if snapshot.estimate.has_interval else None,
            "interval_kind": (
                "marginal_coverage_on_validation_residuals"
                if snapshot.estimate.has_interval
                else None
            ),
        },
        "applicability": {
            "ligand": snapshot.ligand_applicability.value,
            "target": snapshot.target_applicability.value,
            "joint": snapshot.joint_applicability.value,
        },
        "model": {
            **snapshot.provenance.to_dict(),
            "admitted": model_admitted,
        },
        "content_sha256": snapshot.content_sha256,
    }


def activity_observation(
    snapshot: ActivitySnapshot,
    *,
    screening_id: str,
    model_admitted: bool,
    now: datetime,
) -> Observation:
    payload = activity_canonical_payload(
        snapshot, screening_id=screening_id, model_admitted=model_admitted
    )
    return activity_observation_from_payload(
        payload, session_id=snapshot.session_id, run_id=snapshot.run_id, now=now,
    )


def activity_observation_from_payload(
    payload: dict[str, Any],
    *,
    session_id: str,
    run_id: str,
    now: datetime,
    provenance_extra: dict[str, Any] | None = None,
) -> Observation:
    """The citable observation of one activity estimate's canonical payload --
    built from a live snapshot or replayed from a frozen record, the same way."""
    model = payload["model"]
    limitations = ACTIVITY_LIMITATIONS
    if not model["admitted"]:
        limitations = limitations + (NOT_ADMITTED_LIMITATION,)
    return Observation.create(
        session_id=session_id,
        run_id=run_id,
        producer=Producer.PREDICTOR,
        kind=ObservationKind.ACTIVITY_ESTIMATE,
        schema_version=ACTIVITY_SCHEMA_VERSION,
        canonical_payload=payload,
        model_projection={
            "activity_snapshot_id": payload["activity_snapshot_id"],
            "subject": payload["subject"],
            "estimate": payload["estimate"],
            "applicability": payload["applicability"],
            "model_id": model["model_id"],
            "admitted": model["admitted"],
        },
        provenance={
            "activity_snapshot_id": payload["activity_snapshot_id"],
            "screening_id": payload["screening_id"],
            "model_id": model["model_id"],
            "dataset_id": model["dataset_id"],
            "split_id": model["split_id"],
            "artifact_hashes": list(model["artifact_hashes"]),
            "assay_context": payload["subject"]["assay_context"],
            **(provenance_extra or {}),
        },
        now=now,
        required_limitations=limitations,
    )


def prioritisation_canonical_payload(
    decision: PrioritisationDecision, *, screening_id: str
) -> dict[str, Any]:
    return {
        "prioritisation_decision_id": decision.id,
        "screening_id": screening_id,
        "activity_snapshot_id": decision.activity_snapshot_id,
        "status": decision.status.value,
        "reason_codes": list(decision.reason_codes),
        "policy": {
            "policy_id": decision.policy_id,
            "target_id": decision.target_id,
            "assay_context": decision.assay_context,
            "intended_target_role": decision.intended_target_role.value,
            "threshold_pic50": decision.threshold_pic50,
            "threshold_source": decision.threshold_source,
        },
    }


def prioritisation_observation(
    decision: PrioritisationDecision,
    *,
    session_id: str,
    run_id: str,
    screening_id: str,
    now: datetime,
) -> Observation:
    payload = prioritisation_canonical_payload(decision, screening_id=screening_id)
    return prioritisation_observation_from_payload(
        payload, session_id=session_id, run_id=run_id, now=now,
    )


def prioritisation_observation_from_payload(
    payload: dict[str, Any],
    *,
    session_id: str,
    run_id: str,
    now: datetime,
    provenance_extra: dict[str, Any] | None = None,
) -> Observation:
    return Observation.create(
        session_id=session_id,
        run_id=run_id,
        producer=Producer.POLICY,
        kind=ObservationKind.PRIORITISATION_DECISION,
        schema_version=PRIORITISATION_SCHEMA_VERSION,
        canonical_payload=payload,
        model_projection={
            "prioritisation_decision_id": payload["prioritisation_decision_id"],
            "status": payload["status"],
            "reason_codes": list(payload["reason_codes"]),
            "policy": payload["policy"],
        },
        provenance={
            "prioritisation_decision_id": payload["prioritisation_decision_id"],
            "activity_snapshot_id": payload["activity_snapshot_id"],
            "screening_id": payload["screening_id"],
            "policy_id": payload["policy"]["policy_id"],
            "threshold_source": payload["policy"]["threshold_source"],
            **(provenance_extra or {}),
        },
        now=now,
        required_limitations=(POLICY_LIMITATION,),
    )


def screening_state_canonical_payload(
    screening: ActivityGuidedScreening,
) -> dict[str, Any]:
    toxicity_run = screening.status not in _TOXICITY_NOT_RUN
    return {
        "screening_id": screening.id,
        "status": screening.status.value,
        "subject": {
            "canonical_smiles": screening.subject.canonical_smiles,
            "input_smiles": screening.input_smiles,
            "target": screening.subject.target.to_dict(),
            "assay_context": screening.subject.assay_context,
        },
        "intended_target_role": screening.intended_target_role.value,
        "activity_snapshot_id": screening.activity_snapshot_id,
        "prioritisation_decision_id": screening.prioritisation_decision_id,
        "toxicity": {
            # Stated, never absent: a reader must not have to infer this.
            "run": toxicity_run,
            "analysis_id": screening.toxicity_analysis_id,
            "not_run_reason": screening.toxicity_not_run_reason,
            "failure_code": (
                screening.failure_code
                if screening.status is ScreeningStatus.TOXICITY_FAILED
                else None
            ),
            "authorised_by": screening.toxicity_authorised_by,
            "authorisation_basis": screening.toxicity_authorisation_basis,
        },
        "activity_failure_code": (
            screening.failure_code
            if screening.status is ScreeningStatus.ACTIVITY_FAILED
            else None
        ),
        "terminal": screening.is_terminal,
        "version": screening.version,
    }


def screening_state_observation(
    screening: ActivityGuidedScreening, *, run_id: str, now: datetime
) -> Observation:
    payload = screening_state_canonical_payload(screening)
    return screening_state_observation_from_payload(
        payload, session_id=screening.session_id, run_id=run_id, now=now,
        provenance_extra={"session_id": screening.session_id, "run_id": screening.run_id},
    )


def screening_state_observation_from_payload(
    payload: dict[str, Any],
    *,
    session_id: str,
    run_id: str,
    now: datetime,
    provenance_extra: dict[str, Any] | None = None,
) -> Observation:
    limitations: tuple[str, ...] = ()
    if not payload["toxicity"]["run"]:
        limitations = (TOXICITY_NOT_RUN_LIMITATION,)
    return Observation.create(
        session_id=session_id,
        run_id=run_id,
        producer=Producer.WORKFLOW,
        kind=ObservationKind.SCREENING_STATE,
        schema_version=SCREENING_SCHEMA_VERSION,
        canonical_payload=payload,
        model_projection={
            "screening_id": payload["screening_id"],
            "status": payload["status"],
            "subject": payload["subject"],
            "toxicity": payload["toxicity"],
        },
        provenance={"screening_id": payload["screening_id"], **(provenance_extra or {})},
        now=now,
        required_limitations=limitations,
    )
