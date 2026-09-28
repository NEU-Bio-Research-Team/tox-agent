"""``submit_report_draft``'s workflow (spec section 10, stages 4 and 5).

Correction policy, in one sentence and deliberately narrower than the
conversational one: a draft is validated, an invalid first draft returns typed
violations for exactly one correction attempt, and a second invalid draft fails
the build. There is no fallback report.

That asymmetry with ``submit_answer`` is the point. A fallback *answer* is an
honest, server-authored sentence saying what could not be established — a
person reads it in a chat and moves on. A fallback *report* would be a
downloadable, citable, eleven-section document produced by nobody, and there is
no wording that makes that safe (spec section 10: "Never accept invalid agent
output").

What happens on the accepted path: the draft is compiled into the immutable
artifact, the renderings are produced and stored, the claim and evidence link
rows are written, and the build reaches ``completed`` or, when gaps exist,
``completed_with_gaps`` — all in one transaction, so a report can never exist
whose citations were not recorded.
"""
from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import ValidationError

from ...domain.errors import Conflict, ToxAgentError, Violation
from ...domain.events import EventType
from ...domain.evidence import EvidenceRecord
from ...domain.ids import RENDERING, new_id
from ...domain.observation import Observation
from ...persistence.object_store import ObjectNotFound, ObjectRef
from ...domain.report import (
    BuildStage,
    ExplanationPackage,
    ReportArtifact,
    ReportFigure,
    ReportRendering,
    SubstanceProfile,
)
from ...report.draft_compiler import claim_links, compile_report, evidence_links
from ...report.figures import (
    FIGURE_RENDERER_VERSION,
    FigureRejected,
    sanitize_svg,
    store_structure_figure,
)
from ...validation.answer.citations import cited_in_prose
from ...report.renderers import (
    HTML_RENDERER_VERSION,
    MARKDOWN_BUNDLE_RENDERER_VERSION,
    MARKDOWN_RENDERER_VERSION,
    PDF_RENDERER_VERSION,
    RendererUnavailable,
    render_html,
    render_markdown,
    render_markdown_bundle,
    render_pdf,
)
from ...validation.report.validator import (
    ReportValidationContext,
    validate_report_draft,
)
from ...validation.report.draft_wire import ReportDraftCandidate

log = logging.getLogger("toxagent.report")

SUBMIT_TOOL_NAME = "submit_report_draft"
SUBMIT_SAVED_TOOL_NAME = "submit_saved_report_draft"

WORKING_DRAFT_KEY = "working_report_draft"
WORKING_DRAFT_VERSION_KEY = "working_report_draft_version"
WORKING_DRAFT_SHA_KEY = "working_report_draft_sha256"

#: Formats this deployment can actually produce. A requested format outside
#: this set becomes a recorded rendering failure, never a silently missing file.
RENDERERS: Mapping[str, tuple[str, str, Any]] = {
    "markdown": ("text/markdown; charset=utf-8", MARKDOWN_RENDERER_VERSION, render_markdown),
    "markdown_bundle": (
        "application/zip", MARKDOWN_BUNDLE_RENDERER_VERSION, render_markdown_bundle,
    ),
    "html": ("text/html; charset=utf-8", HTML_RENDERER_VERSION, render_html),
    "pdf": ("application/pdf", PDF_RENDERER_VERSION, render_pdf),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ReportValidationFailed(ToxAgentError):
    """A correctable rejection. Carries the typed violations the next attempt
    must fix, and how many attempts remain."""

    code = "report_validation_failed"
    http_status = 422

    def __init__(
        self, message: str, *, violations: list[Violation], attempts_remaining: int
    ) -> None:
        super().__init__(
            message,
            violations=[v.to_dict() for v in violations],
            attempts_remaining=attempts_remaining,
        )
        self.violations = violations
        self.attempts_remaining = attempts_remaining


@dataclass(frozen=True)
class SubmitReportOutcome:
    artifact: ReportArtifact
    renderings: tuple[ReportRendering, ...]
    rendering_failures: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class DraftCheckpointOutcome:
    """A durable working draft and the validator's current assessment."""

    draft: ReportDraftCandidate
    version: int
    content_sha256: str
    violations: tuple[Violation, ...]


def _draft_sha256(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pointer_parts(path: str) -> list[str]:
    if not path.startswith("/") or path == "/":
        raise Conflict("a draft patch path must be a non-root JSON pointer", path=path)
    return [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]


def _apply_draft_patch(document: dict[str, Any], operations: list[Mapping[str, Any]]) -> None:
    """Apply the small RFC-6902 subset exposed to the report agent.

    The patched document is parsed as ``ReportDraftCandidate`` before it is
    stored, so a patch can never checkpoint a half-shaped report. Supporting
    add/replace/remove is enough to repair validator paths without making the
    model resend an otherwise unchanged 10k-token candidate.
    """
    for operation in operations:
        op = operation.get("op")
        path = str(operation.get("path", ""))
        parts = _pointer_parts(path)
        parent: Any = document
        for part in parts[:-1]:
            if isinstance(parent, list):
                try:
                    parent = parent[int(part)]
                except (ValueError, IndexError) as exc:
                    raise Conflict("a draft patch path does not resolve", path=path) from exc
            elif isinstance(parent, dict) and part in parent:
                parent = parent[part]
            else:
                raise Conflict("a draft patch path does not resolve", path=path)

        leaf = parts[-1]
        if isinstance(parent, list):
            if op == "add" and leaf == "-":
                parent.append(deepcopy(operation.get("value")))
                continue
            try:
                index = int(leaf)
            except ValueError as exc:
                raise Conflict("a list patch path requires an integer index", path=path) from exc
            if op == "add":
                if index < 0 or index > len(parent):
                    raise Conflict("a draft patch index is out of range", path=path)
                parent.insert(index, deepcopy(operation.get("value")))
            elif op == "replace":
                if index < 0 or index >= len(parent):
                    raise Conflict("a draft patch index is out of range", path=path)
                parent[index] = deepcopy(operation.get("value"))
            elif op == "remove":
                if index < 0 or index >= len(parent):
                    raise Conflict("a draft patch index is out of range", path=path)
                parent.pop(index)
            else:
                raise Conflict("unsupported draft patch operation", operation=op)
        elif isinstance(parent, dict):
            if op == "remove":
                if leaf not in parent:
                    raise Conflict("a draft patch path does not resolve", path=path)
                del parent[leaf]
            elif op in {"add", "replace"}:
                if op == "replace" and leaf not in parent:
                    raise Conflict("a draft patch path does not resolve", path=path)
                parent[leaf] = deepcopy(operation.get("value"))
            else:
                raise Conflict("unsupported draft patch operation", operation=op)
        else:
            raise Conflict("a draft patch parent is not a container", path=path)


class SubmitReportDraft:
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

    async def save_checkpoint(
        self,
        *,
        session_id: str,
        run_id: str,
        owner_id: str,
        draft: ReportDraftCandidate,
    ) -> DraftCheckpointOutcome:
        """Persist one full candidate once, and validate the stored version.

        Subsequent repair turns operate on JSON-pointer patches. This keeps a
        long report out of repeated model output and leaves a resumable product
        checkpoint if the runtime or provider disappears between validation
        and final submission.
        """
        document = draft.model_dump(mode="json")
        async with self._db.unit_of_work() as uow:
            build, snapshot = await self._admit(uow, session_id=session_id, draft=draft)
            context, *_ = await self._validation_context(
                uow, build=build, snapshot=snapshot, draft=draft,
                session_id=session_id, run_id=run_id, owner_id=owner_id,
            )
            result = validate_report_draft(draft, context=context, now=_now())
            version = int(build.stage_state.get(WORKING_DRAFT_VERSION_KEY, 0)) + 1
            digest = _draft_sha256(document)
            state = dict(build.stage_state)
            state.update({
                WORKING_DRAFT_KEY: document,
                WORKING_DRAFT_VERSION_KEY: version,
                WORKING_DRAFT_SHA_KEY: digest,
            })
            await uow.reports.save_build(replace(build, stage_state=state, updated_at=_now()))
            uow.emit(
                session_id=session_id, type=EventType.REPORT_DRAFT_SAVED,
                entity_type="report_build", entity_id=build.id, run_id=run_id,
                payload={
                    "draft_version": version,
                    "content_sha256": digest,
                    "violation_count": len(result.violations),
                    "violation_codes": sorted({v.code for v in result.violations}),
                },
            )
            await uow.commit()
        return DraftCheckpointOutcome(
            draft=draft, version=version, content_sha256=digest,
            violations=tuple(result.violations),
        )

    async def check_checkpoint(
        self,
        *,
        session_id: str,
        run_id: str,
        owner_id: str,
        report_build_id: str,
    ) -> DraftCheckpointOutcome:
        async with self._db.unit_of_work() as uow:
            build, snapshot, draft, version, digest = await self._load_checkpoint(
                uow, session_id=session_id, report_build_id=report_build_id
            )
            context, *_ = await self._validation_context(
                uow, build=build, snapshot=snapshot, draft=draft,
                session_id=session_id, run_id=run_id, owner_id=owner_id,
            )
            result = validate_report_draft(draft, context=context, now=_now())
        return DraftCheckpointOutcome(
            draft=draft, version=version, content_sha256=digest,
            violations=tuple(result.violations),
        )

    async def patch_checkpoint(
        self,
        *,
        session_id: str,
        run_id: str,
        owner_id: str,
        report_build_id: str,
        expected_version: int,
        operations: list[Mapping[str, Any]],
    ) -> DraftCheckpointOutcome:
        async with self._db.unit_of_work() as uow:
            build, snapshot, _, version, _ = await self._load_checkpoint(
                uow, session_id=session_id, report_build_id=report_build_id
            )
            if version != expected_version:
                raise Conflict(
                    "the saved report draft changed; re-read it before patching",
                    expected_version=expected_version, current_version=version,
                )
            document = deepcopy(build.stage_state[WORKING_DRAFT_KEY])
            _apply_draft_patch(document, operations)
            try:
                draft = ReportDraftCandidate.model_validate(document)
            except ValidationError as exc:
                raise Conflict(
                    "the patch would make the saved report draft structurally invalid",
                    errors=exc.errors(include_url=False, include_context=False)[:10],
                ) from exc
            if draft.report_build_id != report_build_id:
                raise Conflict("a patch cannot move a draft to another report build")

            context, *_ = await self._validation_context(
                uow, build=build, snapshot=snapshot, draft=draft,
                session_id=session_id, run_id=run_id, owner_id=owner_id,
            )
            result = validate_report_draft(draft, context=context, now=_now())
            next_version = version + 1
            canonical = draft.model_dump(mode="json")
            digest = _draft_sha256(canonical)
            state = dict(build.stage_state)
            state.update({
                WORKING_DRAFT_KEY: canonical,
                WORKING_DRAFT_VERSION_KEY: next_version,
                WORKING_DRAFT_SHA_KEY: digest,
            })
            await uow.reports.save_build(replace(build, stage_state=state, updated_at=_now()))
            uow.emit(
                session_id=session_id, type=EventType.REPORT_DRAFT_PATCHED,
                entity_type="report_build", entity_id=build.id, run_id=run_id,
                payload={
                    "draft_version": next_version,
                    "content_sha256": digest,
                    "operation_count": len(operations),
                    "violation_count": len(result.violations),
                    "violation_codes": sorted({v.code for v in result.violations}),
                },
            )
            await uow.commit()
        return DraftCheckpointOutcome(
            draft=draft, version=next_version, content_sha256=digest,
            violations=tuple(result.violations),
        )

    async def execute_checkpoint(
        self,
        *,
        session_id: str,
        run_id: str,
        owner_id: str,
        report_build_id: str,
        expected_version: int,
        language: str = "en",
    ) -> SubmitReportOutcome:
        async with self._db.unit_of_work() as uow:
            _, _, draft, version, _ = await self._load_checkpoint(
                uow, session_id=session_id, report_build_id=report_build_id
            )
        if version != expected_version:
            raise Conflict(
                "the saved report draft changed; check the current version before submitting",
                expected_version=expected_version, current_version=version,
            )
        return await self.execute(
            session_id=session_id, run_id=run_id, owner_id=owner_id,
            draft=draft, language=language,
        )

    async def _load_checkpoint(self, uow, *, session_id: str, report_build_id: str):
        build = await uow.reports.get_build(report_build_id, session_id=session_id)
        if build is None:
            raise Conflict("no such report build in this session", build_id=report_build_id)
        raw = build.stage_state.get(WORKING_DRAFT_KEY)
        if raw is None:
            raise Conflict(
                "this report build has no saved working draft; call save_report_draft first",
                build_id=report_build_id,
            )
        try:
            draft = ReportDraftCandidate.model_validate(raw)
        except ValidationError as exc:  # defensive: old/corrupt checkpoint
            raise Conflict(
                "the saved working draft cannot be reconstructed",
                errors=exc.errors(include_url=False, include_context=False)[:10],
            ) from exc
        admitted, snapshot = await self._admit(uow, session_id=session_id, draft=draft)
        version = int(build.stage_state.get(WORKING_DRAFT_VERSION_KEY, 1))
        digest = str(build.stage_state.get(WORKING_DRAFT_SHA_KEY) or _draft_sha256(raw))
        return admitted, snapshot, draft, version, digest

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

    # --- rejection ---------------------------------------------------------

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

    # --- helpers -----------------------------------------------------------

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

    async def _resolve_evidence(
        self, uow, session_id: str, draft: ReportDraftCandidate
    ) -> Mapping[str, EvidenceRecord]:
        ids = {e for c in draft.claims for e in c.citation_ids}
        ids |= {e for s in draft.evidence_synthesis for e in s.evidence_ids}
        # Inline ``[@evd_...]`` markers too: a token in a sentence is a citation
        # that has to resolve, be validated and reach the reference snapshot,
        # exactly like one attached to a claim (REP-01).
        for section in draft.sections:
            ids |= cited_in_prose(section.body_markdown)
        resolved: dict[str, EvidenceRecord] = {}
        for evidence_id in ids:
            record = await uow.evidence.get(evidence_id, session_id=session_id)
            if record is not None:
                resolved[evidence_id] = record
        return resolved

    async def _resolve_read_evidence_ids(self, uow, run_id: str) -> frozenset[str]:
        """Which evidence this run actually opened, not merely saw in a search
        result — the same source of truth ``submit_answer`` uses."""
        calls = await uow.tool_calls.list_for_run(run_id)
        read: set[str] = set()
        for call in calls:
            if call["tool_name"] == "get_evidence_record" and call["status"] == "completed":
                read.update(call.get("observation_ids") or ())
        return frozenset(read)

    @staticmethod
    def _resolve_explanations(
        build, observations_by_id: Mapping[str, Observation]
    ) -> tuple[dict[str, ExplanationPackage], dict[str, ReportFigure]]:
        """Rebuild the packages this build produced, from the observations.

        Read from stored observations rather than from the build's stage state
        so that a figure/observation mismatch is checked against what was
        actually persisted, not against a summary written next to it.
        """
        from ..explanation.service import is_explanation_schema, package_from_observation

        packages: dict[str, ExplanationPackage] = {}
        figures: dict[str, ReportFigure] = {}
        for observation in observations_by_id.values():
            if not is_explanation_schema(observation.schema_version):
                continue
            package = package_from_observation(observation)
            packages[package.explanation_id] = package
            if package.figure is not None:
                figures[package.figure.figure_id] = package.figure
        return packages, figures

    async def _figure_errors(
        self, uow, figures: Mapping[str, ReportFigure], *, owner_id: str, session_id: str
    ) -> dict[str, str]:
        errors: dict[str, str] = {}
        for figure_id, figure in figures.items():
            attachment = await uow.attachments.get(figure.attachment_id, owner_id=owner_id)
            if attachment is None or attachment.session_id != session_id:
                errors[figure_id] = "the attachment does not exist in this session"
                continue
            if attachment.media_type != figure.media_type:
                errors[figure_id] = "the attachment MIME type differs from the figure"
                continue
            if attachment.sha256 != figure.content_sha256:
                errors[figure_id] = "the attachment hash differs from the figure"
                continue
            if self._objects is None:
                errors[figure_id] = "no object store is configured"
                continue
            try:
                data = await self._objects.get(ObjectRef(attachment.object_uri))
            except ObjectNotFound:
                errors[figure_id] = "the stored figure bytes are missing"
                continue
            if hashlib.sha256(data).hexdigest() != figure.content_sha256:
                errors[figure_id] = "the stored figure bytes do not match the content hash"
        return errors

    async def _structure_figure(
        self, uow, snapshot, *, owner_id: str, session_id: str,
        existing_figure_id: str | None,
    ) -> ReportFigure | None:
        """The compound's own picture, drawn once per session and reused.

        Nothing produced this before: ``SubstanceProfile.structure_figure_id``
        existed and was read from a ``stage_state`` key nothing ever wrote, so
        every report's substance profile was pictureless and the only image
        available was the explanation heat map — which is a statement about one
        model on one endpoint, not about what the compound is (REP-02).

        A failure here is never a failed report. The identity of the compound is
        already in the profile as canonical SMILES and, where resolved, a name
        and identifiers; the drawing is an aid, and losing it must not lose the
        document.
        """
        if existing_figure_id:
            row = await uow.reports.get_figure(existing_figure_id, session_id=session_id)
            # A figure written by an older sanitizer is redrawn, not reused:
            # v1 stripped RDKit's paint and stored black boxes.
            if row is not None and row["renderer_version"] == FIGURE_RENDERER_VERSION:
                return ReportFigure(
                    figure_id=row["figure_id"],
                    attachment_id=row["attachment_id"],
                    media_type=row["media_type"],
                    caption=row["caption"],
                    alt_text=row["alt_text"],
                    content_sha256=row["content_sha256"],
                    renderer_version=row["renderer_version"],
                    endpoint=row.get("endpoint"),
                    task=row.get("task"),
                    observation_id=row.get("observation_id"),
                )
        if self._predictor is None or self._objects is None:
            return None
        try:
            response = await self._predictor.depict(snapshot.canonical_smiles)
        except Exception:  # noqa: BLE001 — an absent drawing, never a failed report
            log.warning(
                "could not draw the structure figure for analysis %s", snapshot.id,
                exc_info=True,
            )
            return None
        if not response.depiction_svg:
            return None
        try:
            stored = await store_structure_figure(
                svg=response.depiction_svg,
                object_store=self._objects,
                owner_id=owner_id,
                session_id=session_id,
                canonical_smiles=snapshot.canonical_smiles,
                now=_now(),
            )
        except FigureRejected:
            log.warning("the structure depiction did not survive sanitization", exc_info=True)
            return None
        await uow.attachments.add(stored.attachment)
        await uow.reports.add_figure(
            stored.figure, session_id=session_id, created_at=_now()
        )
        return stored.figure

    @staticmethod
    def _subject_profile(
        build,
        snapshot,
        observations_by_id: Mapping[str, Observation],
        *,
        structure_figure_id: str | None = None,
    ) -> SubstanceProfile:
        """Identity from the compound record when one was resolved, structure
        from the snapshot always. An unresolved field stays ``None``."""
        record: dict[str, Any] | None = None
        for observation in observations_by_id.values():
            if observation.schema_version == "compound-record-v1" and observation.provenance.get(
                "resolved"
            ):
                record = observation.canonical_payload
        if record is None:
            return SubstanceProfile(
                canonical_smiles=snapshot.canonical_smiles,
                structure_figure_id=structure_figure_id,
            )
        provider = record.get("provider", "compound_provider")
        identifiers = dict(record.get("identifiers") or {})
        source_refs = {
            field: f"{provider}:{record.get('canonical_url') or 'record'}"
            for field in ("preferred_name", "synonyms", "identifiers", "properties")
            if record.get(field)
        }
        return SubstanceProfile(
            canonical_smiles=snapshot.canonical_smiles,
            structure_figure_id=structure_figure_id,
            preferred_name=record.get("preferred_name"),
            synonyms=tuple(record.get("synonyms") or ()),
            identifiers=identifiers,
            properties=tuple(record.get("properties") or ()),
            source_refs=source_refs,
        )

    @staticmethod
    def _provenance(
        snapshot, build, run_id: str, *, runtime_manifest: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        provenance = {
            "analysis_content_sha256": snapshot.content_sha256,
            "predictor_service_version": snapshot.provenance.service_version,
            "predictor_git_commit": snapshot.provenance.git_commit,
            "predictor_base_url_id": snapshot.provenance.base_url_id,
            "artifact_hashes": list(snapshot.provenance.artifact_hashes),
            "policy_snapshot": snapshot.policy_snapshot,
            "report_build_id": build.id,
            "run_id": run_id,
            "requested": build.request.to_dict(),
            "instruction_manifest": build.stage_state.get("instruction_manifest"),
        }
        if runtime_manifest is not None:
            provenance["runtime"] = runtime_manifest
        return provenance

    async def _figure_svgs(
        self, uow, artifact: ReportArtifact, *, owner_id: str, session_id: str
    ) -> dict[str, str]:
        """``figure_id -> sanitized SVG source`` for every figure the report shows.

        Re-sanitized on the way out even though the stored bytes were sanitized
        on the way in. The stored bytes are what a renderer inlines into HTML a
        reader opens, and "it was clean when we wrote it" is not a property this
        function can verify about a blob it just fetched — the hash check proves
        the bytes are the ones recorded, not that the recording was safe.

        A figure that cannot be resolved is simply absent from the map, and the
        renderers fall back to its alt text: the report still says what the
        picture would have said.
        """
        if self._objects is None:
            return {}
        resolved: dict[str, str] = {}
        for figure in artifact.figures:
            if figure.media_type != "image/svg+xml":
                continue
            attachment = await uow.attachments.get(figure.attachment_id, owner_id=owner_id)
            if attachment is None or attachment.session_id != session_id:
                continue
            try:
                data = await self._objects.get(ObjectRef(attachment.object_uri))
            except ObjectNotFound:
                continue
            if hashlib.sha256(data).hexdigest() != figure.content_sha256:
                continue
            try:
                resolved[figure.figure_id] = sanitize_svg(data.decode("utf-8"))
            except (FigureRejected, UnicodeDecodeError):
                continue
        return resolved

    async def _render(
        self, artifact: ReportArtifact, build, figure_svgs: Mapping[str, str] | None = None
    ) -> tuple[tuple[ReportRendering, ...], list[dict[str, str]]]:
        """Produce every requested format this deployment can produce.

        A format that fails is recorded as a failure and the report still
        exists: the artifact is the report, and losing a PDF must not lose the
        document (eval scenario 14).
        """
        renderings: list[ReportRendering] = []
        failures: list[dict[str, str]] = []
        for fmt in build.request.output_formats:
            entry = RENDERERS.get(fmt)
            if entry is None:
                failures.append(
                    {
                        "format": fmt,
                        "reason": f"no renderer is pinned for {fmt!r} in this deployment",
                    }
                )
                continue
            media_type, version, renderer = entry
            try:
                rendered = renderer(artifact, figure_svgs=figure_svgs or {})
            except RendererUnavailable as exc:
                failures.append({"format": fmt, "reason": str(exc)})
                continue
            data = rendered if isinstance(rendered, bytes) else rendered.encode("utf-8")
            digest = hashlib.sha256(data).hexdigest()
            key = f"reports/{artifact.id}/{fmt}-{digest[:16]}"
            if self._objects is None:
                failures.append(
                    {
                        "format": fmt,
                        "reason": "this deployment has no object store, so no file was written",
                    }
                )
                continue
            await self._objects.put(key, data, content_type=media_type)
            renderings.append(
                ReportRendering(
                    rendering_id=new_id(RENDERING),
                    format=fmt,
                    media_type=media_type,
                    object_uri=key,
                    content_sha256=digest,
                    size_bytes=len(data),
                    renderer_version=version,
                    created_at=_now(),
                )
            )
        return tuple(renderings), failures
