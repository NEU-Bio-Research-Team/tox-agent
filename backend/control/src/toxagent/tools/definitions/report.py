"""Report-builder tools: the build manifest, and the one way a report is born.

``get_report_context`` is the first call a report-builder run makes. It answers
"what am I writing, about what, and what has already been produced" from server
state — the selected endpoints, which of them the analysis actually served, the
required section ids, the explanations and figures that exist, and how many
correction attempts remain. None of it is in the prompt, because a manifest
carried in a prompt is one a resumed run no longer has.

The stable path checkpoints one full candidate, repairs it with small versioned
patches, and submits the saved version. The older stateless check/submit tools
remain for compatibility, but a new run should never retransmit a full draft.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...application.conversation import projections
from ...application.investigation import scientific_case_service
from ...application.report.submit_draft import (
    WORKING_DRAFT_KEY,
    WORKING_DRAFT_SHA_KEY,
    WORKING_DRAFT_VERSION_KEY,
    SubmitReportDraft,
)
from ...domain.errors import Conflict
from ...flags import is_enabled
from ...domain.report import REQUIRED_SECTION_IDS
from ...predictor.contract import TOX21_TASKS
from ...validation.report.draft_wire import ReportDraftCandidate
from ..registry import ToolContext, ToolDefinition, ToolOutput


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportContextInput(_Input):
    report_build_id: str = Field(
        description="The build this run was started for. It is checked against the run's own "
                    "build; naming another session's build resolves to nothing."
    )


class SavedDraftInput(_Input):
    report_build_id: str
    expected_version: int = Field(ge=1)


class CheckSavedDraftInput(_Input):
    report_build_id: str


class DraftPatchOperation(_Input):
    op: Literal["add", "replace", "remove"]
    path: str = Field(
        min_length=2, max_length=500,
        description="JSON pointer into the saved ReportDraftCandidate, for example /limitations/-",
    )
    value: Any = None


class PatchSavedDraftInput(SavedDraftInput):
    operations: list[DraftPatchOperation] = Field(min_length=1, max_length=64)
    submit_if_valid: bool = Field(
        default=True,
        description=(
            "When the patch clears every violation, immediately perform final submission in "
            "this same tool call, avoiding another model turn near the deadline."
        ),
    )


def build(database, object_store=None, predictor=None) -> list[ToolDefinition]:
    submit = SubmitReportDraft(database, object_store, predictor)

    def checkpoint_view(outcome) -> dict[str, Any]:
        violations = [v.to_dict() for v in outcome.violations]
        return {
            "report_build_id": outcome.draft.report_build_id,
            "draft_version": outcome.version,
            "content_sha256": outcome.content_sha256,
            "ok": not violations,
            "violation_count": len(violations),
            "violations": violations,
            "next_action": (
                "Call submit_saved_report_draft with this draft_version."
                if not violations else
                "Call patch_saved_report_draft with only the changes named by these violations."
            ),
        }

    def submitted_output(outcome) -> ToolOutput:
        artifact = outcome.artifact
        return ToolOutput(
            canonical=artifact.to_dict(),
            model_view={
                "report_id": artifact.id,
                "accepted": True,
                "status": artifact.status.value,
                "gap_count": len(artifact.gaps),
                "content_sha256": artifact.content_sha256,
                "renderings": [r.format for r in outcome.renderings],
                "rendering_failures": [dict(f) for f in outcome.rendering_failures],
            },
            ui_view=artifact.to_dict(),
            observation_ids=tuple(artifact.cited_observation_ids),
            provenance={"report_build_id": artifact.report_build_id},
        )

    async def report_context(context: ToolContext, payload: ReportContextInput) -> ToolOutput:
        async with database.unit_of_work() as uow:
            state = await uow.reports.get_build(
                payload.report_build_id, session_id=context.session_id
            )
            if state is None:
                raise Conflict(
                    "no such report build in this session", build_id=payload.report_build_id
                )
            snapshot = await uow.analyses.get(state.analysis_id, session_id=context.session_id)
            observations = (
                await uow.observations.list_for_analysis(state.analysis_id) if snapshot else []
            )
            dossier = (
                await scientific_case_service.dossier_for_analysis(
                    uow, session_id=context.session_id, analysis_id=state.analysis_id,
                )
                if is_enabled("scientific_case_v1") else None
            )

        from ...application.explanation.service import (
            is_explanation_schema,
            package_from_observation,
        )

        # Every explanation schema version, not just the one written today: a
        # report built now may legitimately cite an explanation computed before
        # XAI-01 unified the two pipelines, and filtering on one version string
        # is how the builder used to conclude a finished explanation did not
        # exist.
        packages = [
            package_from_observation(observation)
            for observation in observations
            if is_explanation_schema(observation.schema_version)
        ]
        served = tuple(snapshot.served_endpoints) if snapshot else ()
        request = state.request
        # Named separately rather than left for the model to subtract: an
        # endpoint that was asked for and is not served is a gap the report
        # must record, and the easiest way to lose one is to never compute the
        # difference (SCI-06).
        unavailable = [e for e in request.selected_endpoints if e not in served]

        view: dict[str, Any] = {
            "report_build_id": state.id,
            "analysis_id": state.analysis_id,
            "stage": state.stage.value,
            "canonical_smiles": snapshot.canonical_smiles if snapshot else None,
            "report_language": request.report_language,
            "audience": request.audience,
            "selected_endpoints": list(request.selected_endpoints),
            "served_endpoints": list(served),
            "unavailable_endpoints": unavailable,
            "selected_tox21_tasks": list(request.selected_tox21_tasks),
            "include_explanations": request.include_explanations,
            "include_external_evidence": request.include_external_evidence,
            "output_formats": list(request.output_formats),
            "required_section_ids": list(REQUIRED_SECTION_IDS),
            # The field catalogue every `get_analysis_slice` argument has to
            # come from. Without it the only way to learn a section's field
            # names is to guess one, be refused, and read the allowed list off
            # the error — which a live build did twelve times, spending twelve
            # of its forty tool calls on lessons it could have been handed here
            # (run_947bfb9b, 2026-09-09).
            "analysis_slice_fields": {
                section: list(fields)
                for section, fields in projections.SECTION_FIELDS.items()
                # Only what this analysis actually served: offering the fields
                # of an endpoint with no data is offering a call that fails.
                if section not in ("clintox", "herg", "tox21") or section in served
            },
            "tox21_assay_fields": list(projections.TOX21_ASSAY_FIELDS),
            "tox21_assays_available": (
                list(request.selected_tox21_tasks) or (list(TOX21_TASKS) if "tox21" in served else [])
            ),
            "explanations": [
                {
                    "explanation_id": package.explanation_id,
                    "observation_id": package.observation_id,
                    "endpoint": package.endpoint,
                    "task": package.task,
                    "status": package.status.value,
                    "figure_id": package.figure.figure_id if package.figure else None,
                    "unmapped_importance": package.highlights.unmapped_importance,
                }
                for package in packages
            ],
            "correction_attempts_used": state.correction_attempts,
            "correction_attempts_remaining": max(
                0, state.MAX_CORRECTION_ATTEMPTS - state.correction_attempts
            ),
            "working_draft": (
                {
                    "saved": True,
                    "draft_version": int(state.stage_state.get(WORKING_DRAFT_VERSION_KEY, 1)),
                    "content_sha256": state.stage_state.get(WORKING_DRAFT_SHA_KEY),
                    "next_action": "Call check_saved_report_draft; do not regenerate the draft.",
                }
                if WORKING_DRAFT_KEY in state.stage_state else {"saved": False}
            ),
            "deadline_at": state.deadline_at.isoformat() if state.deadline_at else None,
        }
        if dossier is not None:
            # W9-05: the report is a view of the investigation, not a second one.
            view["case_dossier"] = scientific_case_service.report_dossier_view(dossier)
        return ToolOutput(
            canonical=view, model_view=view, ui_view=view,
            provenance={"report_build_id": state.id, "analysis_id": state.analysis_id},
        )

    async def check_draft(context: ToolContext, payload: ReportDraftCandidate) -> ToolOutput:
        violations = await submit.check(
            session_id=context.session_id,
            run_id=context.run_id,
            owner_id=context.actor.subject_id,
            draft=payload,
        )
        view = {
            "ok": not violations,
            "violation_count": len(violations),
            "violations": [v.to_dict() for v in violations],
            # Said explicitly, because the whole value of this tool is that a
            # model believes it: asking cost nothing that submitting would have
            # cost.
            "correction_attempts_consumed": 0,
            "note": (
                "Nothing was stored. This is the same validator submit_report_draft runs, "
                "against the same context. Fix every violation and check again; submit only "
                "when this returns ok."
                if violations
                else "This draft passes every deterministic gate. Submit it."
            ),
        }
        return ToolOutput(
            canonical=view, model_view=view, ui_view=view,
            provenance={"report_build_id": payload.report_build_id, "dry_run": True},
        )

    async def save_draft(context: ToolContext, payload: ReportDraftCandidate) -> ToolOutput:
        outcome = await submit.save_checkpoint(
            session_id=context.session_id, run_id=context.run_id,
            owner_id=context.actor.subject_id, draft=payload,
        )
        view = checkpoint_view(outcome)
        view["deadline_remaining_s"] = max(
            0, int((context.deadline_at - datetime.now(timezone.utc)).total_seconds())
        )
        return ToolOutput(
            canonical=view, model_view=view, ui_view=view,
            provenance={
                "report_build_id": payload.report_build_id,
                "draft_version": outcome.version,
                "checkpointed": True,
            },
        )

    async def check_saved_draft(context: ToolContext, payload: CheckSavedDraftInput) -> ToolOutput:
        outcome = await submit.check_checkpoint(
            session_id=context.session_id, run_id=context.run_id,
            owner_id=context.actor.subject_id, report_build_id=payload.report_build_id,
        )
        view = checkpoint_view(outcome)
        return ToolOutput(
            canonical=view, model_view=view, ui_view=view,
            provenance={
                "report_build_id": payload.report_build_id,
                "draft_version": outcome.version,
                "checkpointed": True,
            },
        )

    async def patch_saved_draft(context: ToolContext, payload: PatchSavedDraftInput) -> ToolOutput:
        outcome = await submit.patch_checkpoint(
            session_id=context.session_id, run_id=context.run_id,
            owner_id=context.actor.subject_id, report_build_id=payload.report_build_id,
            expected_version=payload.expected_version,
            operations=[operation.model_dump(mode="json") for operation in payload.operations],
        )
        if not outcome.violations and payload.submit_if_valid:
            submitted = await submit.execute_checkpoint(
                session_id=context.session_id, run_id=context.run_id,
                owner_id=context.actor.subject_id, report_build_id=payload.report_build_id,
                expected_version=outcome.version, language=context.language,
            )
            return submitted_output(submitted)
        view = checkpoint_view(outcome)
        view["deadline_remaining_s"] = max(
            0, int((context.deadline_at - datetime.now(timezone.utc)).total_seconds())
        )
        return ToolOutput(
            canonical=view, model_view=view, ui_view=view,
            provenance={
                "report_build_id": payload.report_build_id,
                "draft_version": outcome.version,
                "checkpointed": True,
            },
        )

    async def submit_saved_draft(context: ToolContext, payload: SavedDraftInput) -> ToolOutput:
        outcome = await submit.execute_checkpoint(
            session_id=context.session_id, run_id=context.run_id,
            owner_id=context.actor.subject_id, report_build_id=payload.report_build_id,
            expected_version=payload.expected_version, language=context.language,
        )
        return submitted_output(outcome)

    async def submit_draft(context: ToolContext, payload: ReportDraftCandidate) -> ToolOutput:
        outcome = await submit.execute(
            session_id=context.session_id,
            run_id=context.run_id,
            owner_id=context.actor.subject_id,
            draft=payload,
            language=context.language,
        )
        return submitted_output(outcome)

    return [
        ToolDefinition(
            name="get_report_context",
            title="Read this report build's manifest",
            description=(
                "Return what this run is building: the analysis, the endpoints that were "
                "selected and which of them the analysis actually served, the Tox21 assays "
                "chosen, whether explanations and external evidence were requested, the "
                "required section ids, the explanations that already exist, and how many "
                "correction attempts remain. Call it first. Every endpoint listed under "
                "unavailable_endpoints must appear in the report as a recorded gap — it was "
                "asked for and could not be answered, which is a result, not an omission."
            ),
            input_model=ReportContextInput,
            handler=report_context,
            profiles=frozenset({"report_build"}),
            soft_timeout_s=3.0,
            hard_timeout_s=8.0,
        ),
        ToolDefinition(
            name="check_report_draft",
            title="Legacy: validate a retransmitted full draft",
            description=(
                "Compatibility-only; new runs must use save_report_draft and versioned patches. "
                "Run the deterministic report validator against a full draft and return the same "
                "typed violations submit_report_draft would return — while storing nothing, "
                "changing no build state, and consuming none of the correction attempts.\n\n"
                "Because this tool stores nothing, using it after each fix forces the model to "
                "regenerate the complete report and risks a deadline. There is exactly one "
                "correction attempt and no fallback report, so a "
                "draft submitted without checking risks losing the whole report to a "
                "bookkeeping slip — a claim no section references, a duplicate claim id, a "
                "derived limitation code left undeclared.\n\n"
                "Passing here is not acceptance: submit_report_draft remains the boundary, "
                "and it re-validates. But a draft that fails here will fail there."
            ),
            input_model=ReportDraftCandidate,
            handler=check_draft,
            profiles=frozenset({"report_build"}),
            soft_timeout_s=8.0,
            hard_timeout_s=20.0,
        ),
        ToolDefinition(
            name="save_report_draft",
            title="Checkpoint and validate one complete report draft",
            description=(
                "Preferred first validation step. Send the complete ReportDraftCandidate once. "
                "The server stores it durably, validates it, and returns a draft_version plus "
                "typed violations. After this call, never resend the full draft: repair it with "
                "patch_saved_report_draft and finish with submit_saved_report_draft."
            ),
            input_model=ReportDraftCandidate,
            handler=save_draft,
            profiles=frozenset({"report_build"}),
            soft_timeout_s=8.0,
            hard_timeout_s=20.0,
        ),
        ToolDefinition(
            name="check_saved_report_draft",
            title="Revalidate the durable working draft by build id",
            description=(
                "Validate the server-side working draft without retransmitting it. Returns its "
                "current version and typed violations. Prefer patch_saved_report_draft because "
                "that tool patches and validates in one call."
            ),
            input_model=CheckSavedDraftInput,
            handler=check_saved_draft,
            profiles=frozenset({"report_build"}),
            soft_timeout_s=8.0,
            hard_timeout_s=20.0,
        ),
        ToolDefinition(
            name="patch_saved_report_draft",
            title="Patch and validate the durable working draft",
            description=(
                "Apply only the validator-requested add/replace/remove operations using JSON "
                "pointer paths. expected_version prevents stale repairs. The patched draft is "
                "stored and validated atomically. By default, a patch that clears every "
                "violation is submitted immediately in this same call, avoiding another slow "
                "model turn near the deadline."
            ),
            input_model=PatchSavedDraftInput,
            handler=patch_saved_draft,
            profiles=frozenset({"report_build"}),
            soft_timeout_s=8.0,
            hard_timeout_s=20.0,
        ),
        ToolDefinition(
            name="submit_saved_report_draft",
            title="Submit the validated durable draft by id and version",
            description=(
                "Preferred final action. Submit the already checkpointed draft using only its "
                "report_build_id and latest draft_version. This avoids regenerating the full "
                "report at the deadline. Call immediately when the saved validator returns ok."
            ),
            input_model=SavedDraftInput,
            handler=submit_saved_draft,
            profiles=frozenset({"report_build"}),
            soft_timeout_s=20.0,
            hard_timeout_s=120.0,
        ),
        ToolDefinition(
            name="submit_report_draft",
            title="Legacy: submit a retransmitted complete report draft",
            description=(
                "Compatibility-only; prefer submit_saved_report_draft. Submit the finished "
                "report. This is the final action of the run; free-text "
                "output is not a report and nothing else stores one.\n\n"
                "The draft must contain all eleven required section ids, every selected and "
                "served endpoint reported through claims that cite an observation_id and "
                "field_path, an explanation reference for each explained target, a research "
                "outcome (a synthesis, or a recorded gap saying none was found), conclusions "
                "that name an endpoint or are marked is_integrated, recommendations with "
                "basis_claim_ids, the required limitation codes, references, and provenance.\n\n"
                "Rules the validator enforces and will reject the draft for: a numeric or "
                "classification claim's source_value must equal the canonical field exactly; "
                "an evidence id may only be cited after get_evidence_record actually opened "
                "it; a figure may only be referenced by the section whose endpoint it depicts; "
                "a contradicting record must be visible in the external evidence or integrated "
                "interpretation text; partial or unmapped attribution must be stated; no "
                "section may be omitted, and one whose content is unavailable carries a gap "
                "instead; no aggregate 'safe'/'toxic' verdict; no recommendation that promises "
                "safety; no raw HTML or remote images.\n\n"
                "A rejected draft returns typed violations naming exactly what to correct. "
                "There is exactly one correction attempt, and there is no fallback report — a "
                "second failure ends the build with no document at all."
            ),
            input_model=ReportDraftCandidate,
            handler=submit_draft,
            profiles=frozenset({"report_build"}),
            soft_timeout_s=20.0,
            hard_timeout_s=45.0,
            idempotent=False,
        ),
    ]
