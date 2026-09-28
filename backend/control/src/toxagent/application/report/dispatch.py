"""Run a report build through the orchestrator instead of the model (PR-12).

PR-09 to PR-11 built every piece of the server-owned report path and wired none
of it: the loop, the checkpoints, the deterministic handlers, the fact bundle,
the synthesis schema, the compiler and its gates were all tested as units while
``BUILD_REPORT`` kept going to the generic gateway. This module is the dispatch.
Behind ``report_orchestrator_v2`` it is the scheduler handler for the report
intent; with the flag off nothing here runs.

What it adds is only what the units could not know about each other:

**Ports bound to real services.** The explanation stage calls the same
``GetOrCreateExplanation`` the tool does. Substance and evidence go through the
``ToolRunner`` under the ``report_build`` profile, so a server-initiated search
is authorised, budgeted and audited exactly like a model-initiated one — the
only difference is who chose the query.

**A store that does not overwrite the model's turn.** The orchestrator holds a
copy of the build while the synthesizing handler dispatches a runtime, and the
synthesis tool writes to the build during that turn. A save that replaced the
whole row with the orchestrator's copy would erase the accepted synthesis, so
the store merges those keys from the row.

**Stages that publish.** Validating re-judges the stored synthesis against a
freshly assembled bundle; rendering writes the artifact once (a resumed build
finds it rather than writing a second); completion then advances the build and
the run together and posts the report reference.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from ... import metrics
from ...domain.errors import (
    Conflict,
    RuntimeProtocolError,
    RuntimeUnavailable,
    ToxAgentError,
)
from ...domain.events import EventType
from ...domain.message import Message, PartType, Role
from ...domain.report import BuildStage, ReportBuild
from ...domain.run import RunStatus
from ...report.synthesis_artifact import to_artifact
from ...report.draft_compiler import evidence_links
from ..explanation.service import GetOrCreateExplanation
from .inputs import load_report_inputs
from .orchestrator import ReportOrchestrator, StageContext
from .stage_handlers import DeterministicHandlers
from .stages import StageStatus, read_checkpoints
from .synthesis import (
    MAX_SYNTHESIS_SUBMISSIONS,
    SYNTHESIS_ATTEMPTS_KEY,
    SYNTHESIS_KEY,
    SYNTHESIS_SHA_KEY,
    judge,
    stored_synthesis,
)
from ..runs.transitions import advance
from .submit_draft import SubmitReportDraft
from ..tool_context import ToolContext

log = logging.getLogger("toxagent.report.dispatch")

#: Keys the synthesis turn writes while the orchestrator holds a copy of the
#: build. The row's value always wins for these.
TURN_OWNED_STATE_KEYS: tuple[str, ...] = (
    SYNTHESIS_KEY,
    SYNTHESIS_SHA_KEY,
    SYNTHESIS_ATTEMPTS_KEY,
    "instruction_manifest",
)

#: Search wording per endpoint. The server chooses the query from the resolved
#: identity and this vocabulary; a model never does (P1-2).
ENDPOINT_QUERY_TERMS: Mapping[str, str] = {
    "herg": "hERG potassium channel inhibition cardiotoxicity",
    "clintox": "clinical toxicity",
    "tox21": "Tox21 toxicity assay",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ReportStageFailed(ToxAgentError):
    code = "report_stage_failed"
    http_status = 500


class ReportSynthesisRefused(ToxAgentError):
    code = "report_validation_failed"
    http_status = 422


# --- persistence seams -----------------------------------------------------


class DatabaseBuildStore:
    """``BuildStore`` over the report repository, merging turn-owned keys."""

    def __init__(self, database, *, session_id: str) -> None:
        self._db = database
        self._session_id = session_id

    async def load(self, build_id: str) -> ReportBuild | None:
        async with self._db.unit_of_work() as uow:
            return await uow.reports.get_build(build_id, session_id=self._session_id)

    async def save(self, build: ReportBuild) -> None:
        async with self._db.unit_of_work() as uow:
            current = await uow.reports.get_build(build.id, session_id=self._session_id)
            if current is not None:
                state = {**current.stage_state, **build.stage_state}
                for key in TURN_OWNED_STATE_KEYS:
                    if key in current.stage_state:
                        state[key] = current.stage_state[key]
                build = replace(
                    build,
                    stage_state=state,
                    report_id=current.report_id or build.report_id,
                )
            await uow.reports.save_build(build)
            await uow.commit()


class OutboxStageEvents:
    """``EventSink`` writing ``report.stage_changed`` to the session outbox."""

    def __init__(self, database, *, session_id: str, run_id: str) -> None:
        self._db = database
        self._session_id = session_id
        self._run_id = run_id
        self._started: dict[str, float] = {}

    async def emit(self, build: ReportBuild, event: str, payload: dict) -> None:
        stage, status = str(payload.get("stage") or ""), str(payload.get("status") or "")
        clock = time.monotonic()
        if status == StageStatus.RUNNING.value:
            self._started[stage] = clock
        elif status in ("completed", "skipped", "failed"):
            begun = self._started.pop(stage, clock)
            metrics.observe("toxagent_report_stage_seconds", clock - begun, stage=stage, status=status)
        async with self._db.unit_of_work() as uow:
            uow.emit(
                session_id=self._session_id,
                type=EventType(event),
                entity_type="report_build",
                entity_id=build.id,
                run_id=self._run_id,
                payload=dict(payload),
            )
            await uow.commit()


# --- the handler -----------------------------------------------------------


class OrchestratedReportBuild:
    """The ``BUILD_REPORT`` scheduler handler when the orchestrator is on."""

    def __init__(
        self,
        database,
        *,
        gateway,
        runner,
        registry,
        predictor,
        object_store=None,
        tool_timeout_s: float = 600.0,
    ) -> None:
        self._db = database
        self._gateway = gateway
        self._runner = runner
        self._registry = registry
        self._explanations = GetOrCreateExplanation(database, predictor, object_store)
        # Only its renderer table and figure resolution are used: the artifact
        # is published from the same code the old path publishes from, so a v3
        # report and a v2 report cannot render differently by accident.
        self._publisher = SubmitReportDraft(database, object_store, predictor)
        self._tool_timeout_s = tool_timeout_s

    async def execute(self, context) -> None:
        if context.needs_snapshot_first:
            await self._gateway.snapshot_before_runtime(context)
        build_id = context.report_build_id or await self._gateway.ensure_report_build(context)
        context = replace(context, report_build_id=build_id)
        await self._start_run(context)

        orchestrator = ReportOrchestrator(
            store=DatabaseBuildStore(self._db, session_id=context.session_id),
            handlers=self._handlers(context),
            events=OutboxStageEvents(
                self._db, session_id=context.session_id, run_id=context.run_id
            ),
        )
        result = await orchestrator.run(build_id)
        checkpoints = read_checkpoints(result.build.stage_state)
        failed = next(
            (cp for cp in checkpoints.values() if cp.status is StageStatus.FAILED), None
        )
        if failed is not None:
            raise await self._failure(context, failed)
        await self._complete(context, build_id)

    # --- stage handlers ------------------------------------------------------

    def _handlers(self, context) -> dict[BuildStage, Any]:
        deterministic = DeterministicHandlers(
            analyses=self._analysis_port(),
            explanations=self._explanation_port(context),
            substances=(
                self._substance_port(context)
                if self._registry.get("resolve_compound_record") is not None
                else None
            ),
            evidence=(
                self._evidence_port(context)
                if self._registry.get("search_toxicology_evidence") is not None
                else None
            ),
        ).as_mapping()
        return {
            **deterministic,
            BuildStage.SYNTHESIZING: self._synthesize(context),
            BuildStage.VALIDATING: self._validate(context),
            BuildStage.RENDERING: self._render(context),
        }

    def _analysis_port(self):
        async def port(analysis_id: str, *, session_id: str):
            async with self._db.unit_of_work() as uow:
                return await uow.analyses.get(analysis_id, session_id=session_id)

        return port

    def _explanation_port(self, context):
        async def port(*, analysis_id: str, endpoint: str, task: str | None):
            result = await self._explanations.execute(
                owner_id=context.actor.subject_id,
                session_id=context.session_id,
                run_id=context.run_id,
                analysis_id=analysis_id,
                endpoint=endpoint,
                task=task,
            )
            return result.package

        return port

    def _tool_context(self, context) -> ToolContext:
        return ToolContext(
            session_id=context.session_id,
            run_id=context.run_id,
            actor=context.actor,
            profile="report_build",
            deadline_at=_now() + timedelta(seconds=self._tool_timeout_s),
            language=context.language,
            intent=context.intent.value,
            model_selection=context.model_selection,
        )

    async def _call_tool(self, context, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        envelope = await self._runner.call(self._tool_context(context), name, arguments)
        if envelope.get("status") != "completed":
            error = envelope.get("error") or {}
            raise RuntimeError(f"{name} {error.get('code') or envelope.get('status')}")
        return envelope

    def _substance_port(self, context):
        async def resolve(*, canonical_smiles: str) -> Mapping[str, Any] | None:
            build = await self._load_build(context)
            envelope = await self._call_tool(
                context, "resolve_compound_record", {"analysis_id": build.analysis_id}
            )
            canonical = dict(envelope.get("canonical") or {})
            if canonical.get("resolved") is False:
                return None
            canonical.pop("raw", None)
            return canonical

        return resolve

    def _evidence_port(self, context):
        async def port(
            *,
            analysis_id: str,
            endpoint: str,
            compound_names: Sequence[str],
            limit: int,
        ) -> Sequence[Mapping[str, Any]]:
            name = next((n for n in compound_names if n), None)
            if name is None:
                build = await self._load_build(context)
                async with self._db.unit_of_work() as uow:
                    snapshot = await uow.analyses.get(
                        build.analysis_id, session_id=context.session_id
                    )
                name = snapshot.canonical_smiles if snapshot else ""
            envelope = await self._call_tool(
                context,
                "search_toxicology_evidence",
                {
                    "analysis_id": analysis_id,
                    "query": f"{name} {ENDPOINT_QUERY_TERMS.get(endpoint, endpoint)}"[:500],
                    "limit": limit,
                    "endpoint": endpoint,
                },
            )
            view = envelope.get("model_view") or {}
            return [
                item
                for item in view.get("results") or ()
                if item.get("evidence_id") and item.get("status") == "accepted"
            ]

        return port

    def _synthesize(self, context):
        async def handler(stage: StageContext) -> dict[str, Any]:
            build = await self._load_build(context)
            if SYNTHESIS_KEY in build.stage_state:
                # Accepted before a restart. The prose is already paid for.
                return {
                    "synthesis_sha256": build.stage_state.get(SYNTHESIS_SHA_KEY),
                    "attempts": build.stage_state.get(SYNTHESIS_ATTEMPTS_KEY),
                    "reused": True,
                }
            attempts = int(build.stage_state.get(SYNTHESIS_ATTEMPTS_KEY) or 0)
            if attempts >= MAX_SYNTHESIS_SUBMISSIONS:
                raise ReportSynthesisRefused(
                    "every permitted synthesis submission was refused",
                    attempts=attempts,
                )
            inputs = await load_report_inputs(
                self._db, build.id, session_id=context.session_id
            )

            async def has_synthesis(_context) -> bool:
                current = await self._load_build(context)
                return SYNTHESIS_KEY in current.stage_state

            fact_view = inputs.bundle.to_model_view()
            dossier = await self._case_dossier(context, build)
            if dossier is not None:
                # W9-05: the synthesis reads the investigation it reports on.
                # The facts stay the only source of values; the dossier says
                # what was concluded and what is open, and names its sources.
                fact_view = {**fact_view, "case_dossier": dossier}
            await self._gateway.run_report_synthesis(
                context,
                report_build_id=build.id,
                fact_view=fact_view,
                deadline_at=build.deadline_at,
                has_synthesis=has_synthesis,
            )
            build = await self._load_build(context)
            if SYNTHESIS_KEY not in build.stage_state:
                attempts = int(build.stage_state.get(SYNTHESIS_ATTEMPTS_KEY) or 0)
                if attempts >= MAX_SYNTHESIS_SUBMISSIONS:
                    raise ReportSynthesisRefused(
                        f"the synthesis was refused {attempts} time(s); no report was produced",
                        attempts=attempts,
                    )
                raise RuntimeProtocolError(
                    "the runtime turn ended without an accepted submit_report_synthesis "
                    f"({attempts} submission(s) made)"
                )
            return {
                "synthesis_sha256": build.stage_state.get(SYNTHESIS_SHA_KEY),
                "attempts": build.stage_state.get(SYNTHESIS_ATTEMPTS_KEY),
            }

        return handler

    async def _case_dossier(self, context, build: ReportBuild) -> dict[str, Any] | None:
        """The analysis's latest dossier as the report reads it, recorded on the
        build (``case_dossier_ref``) so the report says which one it used."""
        from ...flags import is_enabled
        from ..investigation import scientific_case_service

        if not is_enabled("scientific_case_v1"):
            return None
        async with self._db.unit_of_work() as uow:
            dossier = await scientific_case_service.dossier_for_analysis(
                uow, session_id=context.session_id, analysis_id=build.analysis_id,
            )
        if dossier is None:
            return None
        view = scientific_case_service.report_dossier_view(dossier)
        ref = {key: view[key] for key in ("case_id", "run_id", "case_revision")}
        await DatabaseBuildStore(self._db, session_id=context.session_id).save(
            replace(build, stage_state={**build.stage_state, "case_dossier_ref": ref})
        )
        return view

    def _validate(self, context):
        async def handler(stage: StageContext) -> dict[str, Any]:
            inputs = await load_report_inputs(
                self._db, stage.build.id, session_id=context.session_id
            )
            synthesis = stored_synthesis(inputs.build.stage_state)
            if synthesis is None:
                raise LookupError("no accepted synthesis is stored on this build")
            judged = judge(inputs, synthesis)
            if not judged.ok:
                # Accepted at submission and refused now means an input moved
                # between the two — a promoted record was rejected, say. The
                # report is not published over that.
                raise ReportSynthesisRefused(
                    "the stored synthesis no longer passes against the current facts",
                    violations=[v.to_dict() for v in judged.violations],
                )
            return {
                "synthesis_sha256": inputs.build.stage_state.get(SYNTHESIS_SHA_KEY),
                "content_sha256": judged.report.content_sha256(),
                "gap_count": len(judged.report.gaps),
            }

        return handler

    def _render(self, context):
        async def handler(stage: StageContext) -> dict[str, Any]:
            build_id = stage.build.id
            async with self._db.unit_of_work() as uow:
                existing = next(
                    (
                        item
                        for item in await uow.reports.list_artifacts_for_session(
                            context.session_id, limit=200
                        )
                        if item.get("report_build_id") == build_id
                    ),
                    None,
                )
            if existing is not None:
                # Written before a crash, checkpoint lost. Publishing again
                # would give one build two immutable reports.
                return {"report_id": existing["report_id"], "reused": True}

            inputs = await load_report_inputs(self._db, build_id, session_id=context.session_id)
            synthesis = stored_synthesis(inputs.build.stage_state)
            judged = judge(inputs, synthesis) if synthesis is not None else None
            if judged is None or not judged.ok:
                raise LookupError("rendering reached a build with no passing synthesis")

            async with self._db.unit_of_work() as uow:
                version = await uow.reports.latest_version_for_analysis(
                    inputs.build.analysis_id, session_id=context.session_id
                ) + 1
            artifact = to_artifact(
                judged.report,
                session_id=context.session_id,
                subject=inputs.subject,
                evidence=inputs.evidence,
                explanations=inputs.explanations,
                synthesis_sha256=str(inputs.build.stage_state.get(SYNTHESIS_SHA_KEY) or ""),
                version=version,
                now=_now(),
            )
            async with self._db.unit_of_work() as uow:
                figure_svgs = await self._publisher._figure_svgs(
                    uow, artifact, owner_id=context.actor.subject_id,
                    session_id=context.session_id,
                )
            renderings, failures = await self._publisher._render(
                artifact, inputs.build, figure_svgs
            )
            artifact = artifact.with_renderings(renderings)
            async with self._db.unit_of_work() as uow:
                await uow.reports.add_artifact(
                    artifact.to_dict(), evidence_links=evidence_links(artifact)
                )
                for figure in artifact.figures:
                    if await uow.reports.get_figure(
                        figure.figure_id, session_id=context.session_id
                    ) is None:
                        await uow.reports.add_figure(
                            figure, session_id=context.session_id,
                            created_at=artifact.created_at,
                        )
                for rendering in renderings:
                    await uow.reports.add_rendering(artifact.id, rendering)
                await uow.commit()
            return {
                "report_id": artifact.id,
                "formats": [r.format for r in renderings],
                "rendering_failures": [dict(f) for f in failures],
            }

        return handler

    # --- run lifecycle -------------------------------------------------------

    async def _load_build(self, context) -> ReportBuild:
        async with self._db.unit_of_work() as uow:
            build = await uow.reports.get_build(
                context.report_build_id, session_id=context.session_id
            )
        if build is None:
            raise Conflict("the report build this run was started for no longer resolves")
        return build

    async def _start_run(self, context) -> None:
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            if run is None or run.session_id != context.session_id:
                raise RuntimeProtocolError("the report run does not exist in this session")
            if run.status is RunStatus.QUEUED:
                await advance(
                    uow, run, RunStatus.RUNNING,
                    payload={"report_build_id": context.report_build_id, "orchestrated": True},
                )
                await uow.commit()

    async def _failure(self, context, checkpoint) -> ToxAgentError:
        detail = checkpoint.detail or f"{checkpoint.stage} failed"
        if checkpoint.stage == BuildStage.SYNTHESIZING.value:
            if detail.startswith("RuntimeUnavailable"):
                # The scheduler's recovery policy applies to this one: the run
                # had a runtime binding, and a recovery resumes from the
                # checkpoints rather than repeating the assembly stages.
                return RuntimeUnavailable(detail, report_build_id=context.report_build_id)
            if detail.startswith("ReportSynthesisRefused"):
                return ReportSynthesisRefused(detail, report_build_id=context.report_build_id)
        return ReportStageFailed(
            f"report stage {checkpoint.stage} failed: {detail}",
            stage=checkpoint.stage,
            report_build_id=context.report_build_id,
        )

    async def _complete(self, context, build_id: str) -> None:
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            build = await uow.reports.get_build(build_id, session_id=context.session_id)
            if run is None or build is None:
                raise RuntimeProtocolError("the report run or build disappeared before completion")
            rendered = read_checkpoints(build.stage_state).get(BuildStage.RENDERING.value)
            report_id = (rendered.output_refs.get("report_id") if rendered else None) or build.report_id
            artifact = (
                await uow.reports.get_artifact(report_id, session_id=context.session_id)
                if report_id else None
            )
            if artifact is None:
                raise RuntimeProtocolError("the rendered report artifact cannot be read back")
            if not build.is_terminal:
                terminal = (
                    BuildStage.COMPLETED_WITH_GAPS
                    if artifact["status"] == BuildStage.COMPLETED_WITH_GAPS.value
                    else BuildStage.COMPLETED
                )
                build = SubmitReportDraft._walk_to(build, terminal, report_id=report_id)
                await uow.reports.save_build(build)
                uow.emit(
                    session_id=context.session_id,
                    type=(
                        EventType.REPORT_COMPLETED_WITH_GAPS
                        if terminal is BuildStage.COMPLETED_WITH_GAPS
                        else EventType.REPORT_COMPLETED
                    ),
                    entity_type="report", entity_id=report_id, run_id=context.run_id,
                    payload={
                        "report_build_id": build.id,
                        "status": artifact["status"],
                        "gap_count": len(artifact.get("gaps") or ()),
                        "formats": [r.get("format") for r in artifact.get("renderings") or ()],
                        "schema_version": artifact.get("schema_version"),
                    },
                )
            if run.is_terminal:
                await uow.commit()
                return
            sequence = await uow.messages.next_sequence(context.session_id)
            reply = Message.create(
                context.session_id, Role.ASSISTANT, sequence, now=_now(),
                parts=(
                    (PartType.TEXT, {"text": f"Report completed: {artifact['title']}"}),
                    (PartType.REPORT_REF, {
                        "report_id": report_id,
                        "report_build_id": build.id,
                        "status": artifact["status"],
                    }),
                ),
            )
            await uow.messages.add(reply)
            uow.emit(
                session_id=context.session_id, type=EventType.MESSAGE_CREATED,
                entity_type="message", entity_id=reply.id, run_id=context.run_id,
                payload={"role": "assistant", "report_id": report_id},
            )
            await advance(
                uow, run, RunStatus.COMPLETED,
                payload={"report_id": report_id, "report_build_id": build.id, "orchestrated": True},
            )
            await uow.commit()
