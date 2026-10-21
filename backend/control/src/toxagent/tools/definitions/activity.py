"""Activity read tools over attached, frozen screenings (#44).

Three reads, one per artifact the activity workflow produces -- the stage
status, the estimate, the policy decision -- each returning a citable
observation the grounded-answer resolver reads values from. All three take a
``screening_id`` and nothing else: a screening is one ligand on one target in
one assay context, so there is no way to ask for "the activity of this
molecule" and get another target's number.

A stage that did not happen is an answer, not an error: reading the estimate
of a screening whose activity stage failed says so, with the screening's
status, so "toxicity not run" or "no estimate" is never read as a negative.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from ...application.activity import replay
from ...application.conversation.subject_resolution import binding_of
from ...domain.errors import ScreeningNotFound
from ..registry import ToolContext, ToolDefinition, ToolOutput


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ScreeningInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    screening_id: str = Field(
        description="A screening attached to this session, as get_artifact_inventory lists "
                    "it (scr_...). One ligand on one target in one assay context.",
    )


#: The field paths a claim can cite, per artifact, shown beside each read so
#: the model never has to guess a path.
_CITABLE = {
    "screening_state": (
        "status", "toxicity.run", "toxicity.not_run_reason", "toxicity.failure_code",
        "activity_failure_code", "intended_target_role",
    ),
    "activity_estimate": (
        "estimate.predicted_pic50", "estimate.prediction_interval.0",
        "estimate.prediction_interval.1", "estimate.interval_level", "estimate.uncertainty",
        "applicability.ligand", "applicability.target", "applicability.joint",
        "model.model_id", "model.admitted",
    ),
    "prioritisation_decision": (
        "status", "reason_codes", "policy.threshold_pic50", "policy.threshold_source",
        "policy.policy_id", "policy.intended_target_role",
    ),
}


def build(database) -> list[ToolDefinition]:
    def reader(artifact: str):
        async def read(context: ToolContext, payload: ScreeningInput) -> ToolOutput:
            async with database.unit_of_work() as uow:
                attachment = await uow.screenings.get(
                    payload.screening_id, session_id=context.session_id
                )
            if attachment is None:
                raise ScreeningNotFound(
                    "no such screening attached to this session",
                    screening_id=payload.screening_id,
                )
            record = attachment.record
            observation = replay.observation_for(
                record, artifact, session_id=context.session_id, run_id=context.run_id,
                now=_now(),
            )
            state = record.get("screening_state") or {}
            if observation is None:
                view = {
                    "screening_id": payload.screening_id,
                    "artifact": artifact,
                    "produced": False,
                    "screening_status": state.get("status"),
                    "activity_failure_code": state.get("activity_failure_code"),
                    "note": (
                        "This stage produced no artifact for this screening. That is not a "
                        "negative result; cite the screening state for what happened."
                    ),
                }
                return ToolOutput(
                    canonical=view, model_view=view, ui_view=view,
                    provenance={"screening_id": payload.screening_id, "artifact": artifact,
                                "record_sha256": attachment.record_sha256},
                )
            async with database.unit_of_work() as uow:
                await uow.observations.add(observation)
                await uow.commit()
            view = {
                **observation.model_projection,
                "artifact": artifact,
                "produced": True,
                "mode": "replay",
                "required_limitations": list(observation.required_limitations),
                "citable_field_paths": list(_CITABLE[artifact]),
            }
            binding = binding_of(context)
            if artifact == "screening_state" and binding is not None \
                    and binding.screening_id == payload.screening_id:
                # #44: the toxicity analysis id in the record is the frozen
                # run's; this session reads the copy the server linked to it.
                toxicity = binding.artifact("toxicity_snapshot")
                view["linked_toxicity_analysis_id"] = (
                    toxicity.local_id if toxicity is not None else None
                )
            return ToolOutput(
                canonical=observation.canonical_payload, model_view=view, ui_view=view,
                observation_ids=(observation.id,),
                provenance={"screening_id": payload.screening_id, "artifact": artifact,
                            "record_sha256": attachment.record_sha256, "mode": "replay"},
            )

        return read

    common = dict(
        input_model=ScreeningInput,
        profiles=frozenset({"decision_support"}),
        soft_timeout_s=5.0,
        hard_timeout_s=15.0,
        max_retries=1,
    )
    return [
        ToolDefinition(
            name="get_screening_state",
            title="Read a screening's stage status",
            description=(
                "Where one activity-guided screening stands: its status (e.g. review_required, "
                "low_priority, completed_with_toxicity_not_run), whether the toxicity stage "
                "ran, and why not. A stage that did not run is not a negative result. Returns "
                "a citable observation_id; its required_limitations must be declared."
            ),
            handler=reader("screening_state"),
            **common,
        ),
        ToolDefinition(
            name="get_activity_snapshot",
            title="Read a screening's activity estimate",
            description=(
                "The predicted pIC50 of one screening's ligand on its target, in its assay "
                "context: interval and its coverage kind, uncertainty, ligand/target/joint "
                "applicability and model provenance. A pIC50 is an in-vitro potency estimate, "
                "not efficacy, and holds for this target and context only. Returns a citable "
                "observation_id; its required_limitations must be declared."
            ),
            handler=reader("activity_estimate"),
            **common,
        ),
        ToolDefinition(
            name="get_prioritisation_decision",
            title="Read a screening's prioritisation decision",
            description=(
                "The policy decision over one screening's estimate: status, reason codes, the "
                "threshold and its source, and the policy version. It is the output of a fixed "
                "rule, not a measurement, and cannot be overridden here. Returns a citable "
                "observation_id; its required_limitations must be declared."
            ),
            handler=reader("prioritisation_decision"),
            **common,
        ),
    ]
