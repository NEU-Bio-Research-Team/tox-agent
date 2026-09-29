"""``SubmitReportDraft``: the entry points and the dispatch; the rest is in the mixins."""
from __future__ import annotations

from typing import Any, Mapping

from ....domain.errors import Conflict, Violation
from ....domain.events import EventType
from ....domain.evidence import EvidenceRecord
from ....domain.observation import Observation
from ....domain.report import (
    BuildStage,
    ExplanationPackage,
    ReportFigure,
)
from ....report.draft_compiler import claim_links, compile_report, evidence_links
from ....validation.report.draft_wire import ReportDraftCandidate
from ....validation.report.validator import (
    ReportValidationContext,
    validate_report_draft,
)
from ._common import ReportValidationFailed, SubmitReportOutcome, _now
from .assembly import ReportAssemblyMixin
from .checkpoints import DraftCheckpointMixin


class SubmitReportDraft(DraftCheckpointMixin, ReportAssemblyMixin):
    """``predictor`` is optional and used for exactly one thing: the neutral
    structure drawing (REP-02). A deployment without it produces a report whose
    substance profile has no picture — a visible absence, not a heat map
    standing in for an identity figure."""

    def __init__(self, database, object_store=None, predictor=None) -> None:
        self._db = database
        self._objects = object_store
        self._predictor = predictor

    async def _admit(self, uow, *, session_id: str, draft: ReportDraftCandidate):
        """The build and analysis this draft is about, or a typed conflict."""
        build = await uow.reports.get_build(draft.report_build_id, session_id=session_id)
        if build is None:
            # Scoped to the session, so a foreign build id is
            # indistinguishable from a nonexistent one.
            raise Conflict(
                "no such report build in this session", build_id=draft.report_build_id
            )
        if build.report_id is not None:
            raise Conflict(
                "this build has already produced a report; an artifact is immutable",
                build_id=build.id,
            )
        if build.is_terminal:
            raise Conflict(
                f"this report build is {build.stage.value} and accepts no further drafts",
                build_id=build.id,
            )
        snapshot = await uow.analyses.get(build.analysis_id, session_id=session_id)
        if snapshot is None:
            raise Conflict(
                "the analysis this build was created for no longer resolves",
                analysis_id=build.analysis_id,
            )
        return build, snapshot

    async def _validation_context(
        self, uow, *, build, snapshot, draft: ReportDraftCandidate,
        session_id: str, run_id: str, owner_id: str,
    ) -> tuple[ReportValidationContext, dict[str, Observation], Mapping[str, EvidenceRecord],
               dict[str, ExplanationPackage], dict[str, ReportFigure]]:
        """Everything the validator judges a draft against.

        Extracted so that ``check`` and ``execute`` cannot drift: a dry run that
        validated against a different context would be worse than no dry run,
        because it would report a pass the real submission then refused.
        """
        observations = await uow.observations.list_for_analysis(snapshot.id)
        observations_by_id = {o.id: o for o in observations}
        evidence_by_id = await self._resolve_evidence(uow, session_id, draft)
        read_evidence_ids = await self._resolve_read_evidence_ids(uow, run_id)
        explanations, figures = self._resolve_explanations(build, observations_by_id)
        figure_errors = await self._figure_errors(
            uow, figures, owner_id=owner_id, session_id=session_id
        )
        context = ReportValidationContext(
            session_id=session_id,
            report_build_id=build.id,
            analysis_id=snapshot.id,
            selected_endpoints=build.request.selected_endpoints,
            served_endpoints=snapshot.served_endpoints,
            observations_by_id=observations_by_id,
            evidence_by_id=evidence_by_id,
            explanations_by_id=explanations,
            figures_by_id=figures,
            figure_errors=figure_errors,
            read_evidence_ids=read_evidence_ids,
            include_explanations=build.request.include_explanations,
            include_external_evidence=build.request.include_external_evidence,
            selected_tox21_tasks=build.request.selected_tox21_tasks,
            language=build.request.report_language,
        )
        return context, observations_by_id, evidence_by_id, explanations, figures

    async def check(
        self,
        *,
        session_id: str,
        run_id: str,
        owner_id: str,
        draft: ReportDraftCandidate,
    ) -> list[Violation]:
        """Validate a draft without submitting it. Changes nothing.

        The acceptance boundary is unmoved: this runs the *same*
        ``validate_report_draft`` against the *same* context, and a draft that
        passes here still has to pass there. What it removes is the need to
        spend the single correction attempt discovering a typo.

        That attempt exists to bound how many times a model may *disagree* with
        the validator, and there is no reason for it to also bound how many
        times a model may *ask*. The validator is deterministic and touches no
        provider: it reads rows this run already produced. Before this existed,
        the only way to run it was to submit — so a build that had assembled
        eleven correct sections and forgotten one derived limitation code lost
        the whole report to a one-line omission (run_947bfb9b, 2026-09-09).
        """
        async with self._db.unit_of_work() as uow:
            build, snapshot = await self._admit(uow, session_id=session_id, draft=draft)
            context, *_ = await self._validation_context(
                uow, build=build, snapshot=snapshot, draft=draft,
                session_id=session_id, run_id=run_id, owner_id=owner_id,
            )
            result = validate_report_draft(draft, context=context, now=_now())
        # No `save_build`, no stage transition, no correction_attempts, no
        # event: a question that changed the build's state would not be a
        # question.
        return list(result.violations)

    async def execute(
        self,
        *,
        session_id: str,
        run_id: str,
        owner_id: str,
        draft: ReportDraftCandidate,
        language: str = "en",
    ) -> SubmitReportOutcome:
        async with self._db.unit_of_work() as uow:
            build, snapshot = await self._admit(uow, session_id=session_id, draft=draft)
            (
                context, observations_by_id, evidence_by_id, explanations, figures
            ) = await self._validation_context(
                uow, build=build, snapshot=snapshot, draft=draft,
                session_id=session_id, run_id=run_id, owner_id=owner_id,
            )
            result = validate_report_draft(draft, context=context, now=_now())

            if not result.ok:
                return await self._reject(uow, build, result.violations, session_id, run_id)

            structure = await self._structure_figure(
                uow, snapshot, owner_id=owner_id, session_id=session_id,
                existing_figure_id=build.stage_state.get("structure_figure_id"),
            )
            subject = self._subject_profile(
                build, snapshot, observations_by_id,
                structure_figure_id=structure.figure_id if structure else None,
            )
            if structure is not None:
                figures[structure.figure_id] = structure
            previous_version = await uow.reports.latest_version_for_analysis(
                snapshot.id, session_id=session_id
            )
            run = await uow.runs.get(run_id)
            binding = (
                await uow.runtime_bindings.get(run.runtime_binding_id)
                if run is not None and run.runtime_binding_id else None
            )
            artifact = compile_report(
                draft,
                session_id=session_id,
                analysis_id=snapshot.id,
                report_build_id=build.id,
                subject=subject,
                observations_by_id=observations_by_id,
                explanations_by_id=explanations,
                figures_by_id=figures,
                # The resolved records, so the artifact carries its own
                # immutable snapshot of every source rather than an id that
                # depends on the session's evidence table still saying the same
                # thing when somebody opens the report (REP-01).
                evidence_by_id=evidence_by_id,
                provenance=self._provenance(
                    snapshot, build, run_id,
                    runtime_manifest=binding.manifest() if binding else None,
                ),
                report_language=build.request.report_language,
                supersedes_report_id=build.stage_state.get("supersedes_report_id"),
                version=previous_version + 1,
                now=_now(),
            )

            # Figure bytes, resolved once and shared by every format. Before
            # REP-03 the renderers were called with no figure map at all, so
            # HTML and PDF printed "[figure unavailable]" for pictures whose
            # bytes were sitting in the object store.
            figure_svgs = await self._figure_svgs(
                uow, artifact, owner_id=owner_id, session_id=session_id
            )
            renderings, failures = await self._render(artifact, build, figure_svgs)
            artifact = artifact.with_renderings(renderings)

            await uow.reports.add_artifact(
                artifact.to_dict(),
                claim_links=claim_links(artifact),
                evidence_links=evidence_links(artifact),
            )
            for figure in artifact.figures:
                if await uow.reports.get_figure(figure.figure_id, session_id=session_id) is None:
                    await uow.reports.add_figure(
                        figure, session_id=session_id, created_at=artifact.created_at
                    )
            for rendering in renderings:
                await uow.reports.add_rendering(artifact.id, rendering)

            stage = (
                BuildStage.COMPLETED_WITH_GAPS if artifact.gaps else BuildStage.COMPLETED
            )
            # The build machine's own path is queued -> ... -> rendering ->
            # completed. A submission arriving while the build is still marked
            # synthesizing has legitimately done validating and rendering
            # inside this call, so it is walked through those states rather
            # than jumped past them: the transition table stays the only
            # description of what order things happen in.
            build = self._walk_to(build, stage, report_id=artifact.id)
            await uow.reports.save_build(build)

            for changed_stage in (BuildStage.VALIDATING, BuildStage.RENDERING):
                uow.emit(
                    session_id=session_id, type=EventType.REPORT_STAGE_CHANGED,
                    entity_type="report_build", entity_id=build.id, run_id=run_id,
                    payload={"stage": changed_stage.value},
                )

            uow.emit(
                session_id=session_id,
                type=EventType.REPORT_COMPLETED_WITH_GAPS
                if artifact.gaps else EventType.REPORT_COMPLETED,
                entity_type="report", entity_id=artifact.id, run_id=run_id,
                payload={
                    "report_build_id": build.id,
                    "status": artifact.status.value,
                    "gap_count": len(artifact.gaps),
                    "formats": [r.format for r in renderings],
                    "rendering_failures": [f["format"] for f in failures],
                },
            )
            await uow.commit()

        return SubmitReportOutcome(
            artifact=artifact, renderings=renderings, rendering_failures=tuple(failures)
        )

    async def _reject(
        self, uow, build, violations, session_id: str, run_id: str
    ) -> SubmitReportOutcome:
        exhausted = build.corrections_exhausted
        if exhausted:
            build = self._walk_to(
                build, BuildStage.FAILED,
                failure_code="report_validation_failed",
                failure_detail=(
                    f"{len(violations)} violation(s) survived the one permitted correction "
                    "attempt"
                ),
            )
        else:
            build = self._walk_to(build, BuildStage.SYNTHESIZING)
        await uow.reports.save_build(build)
        uow.emit(
            session_id=session_id, type=EventType.REPORT_VALIDATION_FAILED,
            entity_type="report_build", entity_id=build.id, run_id=run_id,
            payload={
                "correction_attempts": build.correction_attempts,
                "violations": [v.to_dict() for v in violations],
            },
        )
        if exhausted:
            uow.emit(
                session_id=session_id, type=EventType.REPORT_FAILED,
                entity_type="report_build", entity_id=build.id, run_id=run_id,
                payload={
                    "failure_code": "report_validation_failed",
                    "violation_count": len(violations),
                },
            )
        await uow.commit()

        remaining = 0 if exhausted else 1
        raise ReportValidationFailed(
            f"the report draft did not pass validation ({len(violations)} violation(s), "
            f"listed in details.violations). "
            + (
                "This build has now used its one correction attempt and has failed; no report "
                "was produced."
                if exhausted
                else "Correct exactly those and call submit_report_draft once more — one "
                "attempt remains, and there is no fallback report."
            ),
            violations=list(violations),
            attempts_remaining=remaining,
        )

    @staticmethod
    def _walk_to(build, stage: BuildStage, **changes: Any):
        """Move a build to ``stage`` through the transition table.

        A submission can arrive at several legal stages, and the alternative to
        walking is a ``replace()`` that sets ``stage`` directly — which is how
        a state machine quietly stops being one.
        """
        now = _now()
        route = {
            BuildStage.COMPLETED: (BuildStage.VALIDATING, BuildStage.RENDERING),
            BuildStage.COMPLETED_WITH_GAPS: (BuildStage.VALIDATING, BuildStage.RENDERING),
            BuildStage.SYNTHESIZING: (BuildStage.VALIDATING,),
            BuildStage.FAILED: (),
        }.get(stage, ())
        for intermediate in route:
            if build.stage is intermediate:
                continue
            try:
                build = build.advance(intermediate, now=now)
            except Exception:
                # Already past it (a resumed build), which is not an error:
                # the destination transition below is still checked.
                pass
        return build.advance(stage, now=now, **changes)
