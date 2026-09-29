"""Report builds driven through the gateway, and committing a report turn."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Awaitable, Callable

from ...application.runs.scheduler import RunContext
from ...application.runs.transitions import advance
from ...domain.errors import DeadlineExceeded, RuntimeProtocolError
from ...domain.events import EventType
from ...domain.message import Message, PartType, Role
from ...domain.report import BuildStage, ReportBuild, ReportBuildRequest
from ...domain.run import RunStatus
from ..synthesis_profile import PROFILE_NAME as SYNTHESIS_PROFILE
from ..synthesis_profile import compose_synthesis_profile
from ._common import _no_commit, _now


class ReportsMixin:
    """Report builds and report completion."""

    async def ensure_report_build(self, context: RunContext) -> str:
        """Create the build manifest without walking any stage (WS05).

        The orchestrator emits a stage event when a stage actually starts; the
        old path's walk through five stages at one timestamp is exactly what it
        replaces, so it must not happen on the way in either.
        """
        return await self._ensure_report_build(context, walk_stages=False)

    async def run_report_synthesis(
        self,
        context: RunContext,
        *,
        report_build_id: str,
        fact_view: dict,
        deadline_at: datetime | None,
        has_synthesis: Callable[[RunContext], Awaitable[bool]],
    ) -> None:
        """Dispatch the one LLM turn of an orchestrated report build (PR-12).

        The prompt is the synthesis brief and the fact bundle, nothing else: no
        answer-format guide, no required-limitations guide and no transcript,
        because none of them describe work this turn can do. The capability
        token names ``report_synthesis``, so the runtime sees one tool.

        Completion is not this method's business. The turn ends, and the
        orchestrator's synthesizing handler decides from the stored build
        whether a synthesis was accepted.
        """
        if self._profiles_dir is None:
            raise RuntimeProtocolError("the report synthesis profile directory is not configured")
        import json
        from dataclasses import replace

        composed = compose_synthesis_profile(self._profiles_dir)
        profile = SYNTHESIS_PROFILE
        system_prompt = "\n\n".join(
            (
                composed.instructions,
                f"Capability profile for this turn: {profile}. Only the tools listed by "
                "the MCP server for this connection exist.",
                "Pinned references:\n"
                f"- report_build {report_build_id}: the fact bundle below is the only "
                "source of values for this report.\n"
                "```json\n"
                + json.dumps(fact_view, sort_keys=True, ensure_ascii=False, default=str)
                + "\n```",
            )
        )
        now = _now()
        deadline = now + timedelta(seconds=self._settings.report_turn_deadline_s)
        if deadline_at is not None:
            deadline = min(deadline, deadline_at)
        if deadline <= now:
            raise DeadlineExceeded("the report build deadline elapsed before synthesis")
        await self._dispatch(
            replace(
                context,
                report_build_id=report_build_id,
                text=f"Write and submit the synthesis for report build {report_build_id}.",
            ),
            system_prompt=system_prompt,
            profile=profile,
            deadline=deadline,
            instructions_hash=composed.content_sha256,
            has_product=has_synthesis,
            commit=_no_commit,
        )

    async def _ensure_report_build(self, context: RunContext, *, walk_stages: bool = True) -> str:
        """Create the durable build manifest after an optional new snapshot exists.

        ``walk_stages`` is the old path's behaviour and stays its default: the
        model-driven build expects to find the pointer at synthesizing.
        """
        now = _now()
        async with self._db.unit_of_work() as uow:
            existing = await uow.reports.list_builds_for_session(context.session_id, limit=50)
            current = (
                next((item for item in existing if item.id == context.report_build_id), None)
                if context.report_build_id
                else next((item for item in existing if item.run_id == context.run_id), None)
            )
            if current is not None:
                return current.id
            session = await uow.sessions.get_unscoped(context.session_id)
            analysis_id = context.analysis_id or (session.active_analysis_id if session else None)
            snapshot = (
                await uow.analyses.get(analysis_id, session_id=context.session_id)
                if analysis_id else None
            )
            if snapshot is None:
                raise RuntimeProtocolError("a report build requires an immutable analysis snapshot")
            selected = tuple(context.endpoints or snapshot.served_endpoints)
            unavailable = sorted(set(selected) - set(snapshot.served_endpoints))
            if unavailable:
                raise RuntimeProtocolError(
                    "selected report endpoints are not served by this analysis",
                    unavailable_endpoints=unavailable,
                )
            request = ReportBuildRequest(
                session_id=context.session_id,
                analysis_id=snapshot.id,
                selected_endpoints=selected,
                selected_tox21_tasks=tuple(
                    task for endpoint, task in context.explanation_targets
                    if endpoint == "tox21" and task
                ),
                report_language=context.report_language,
                audience=context.report_audience,
                include_explanations=context.explanation_mode != "none",
                include_external_evidence=context.include_external_evidence,
                output_formats=context.report_output_formats,
            )
            build = ReportBuild.start(
                session_id=context.session_id, run_id=context.run_id, request=request,
                now=now,
                deadline_at=now + timedelta(seconds=self._settings.report_turn_deadline_s),
            )
            await uow.reports.add_build(build)
            uow.emit(
                session_id=context.session_id, type=EventType.REPORT_BUILD_STARTED,
                entity_type="report_build", entity_id=build.id, run_id=context.run_id,
                payload={"analysis_id": snapshot.id, "selected_endpoints": list(selected)},
            )
            if not walk_stages:
                await uow.commit()
                return build.id
            stages = [
                BuildStage.PREPARING_ANALYSIS, BuildStage.ASSEMBLING_SUBSTANCE,
                BuildStage.ASSEMBLING_PREDICTIONS,
            ]
            if request.include_explanations:
                stages.append(BuildStage.GENERATING_EXPLANATIONS)
            if request.include_external_evidence:
                stages.append(BuildStage.RESEARCHING_EVIDENCE)
            stages.append(BuildStage.SYNTHESIZING)
            for stage in stages:
                build = build.advance(stage, now=now)
            await uow.reports.save_build(build)
            uow.emit(
                session_id=context.session_id, type=EventType.REPORT_STAGE_CHANGED,
                entity_type="report_build", entity_id=build.id, run_id=context.run_id,
                payload={"stage": build.stage.value},
            )
            await uow.commit()
            return build.id

    async def _commit_report_and_complete(self, context: RunContext) -> None:
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            builds = await uow.reports.list_builds_for_session(context.session_id, limit=50)
            build = next((item for item in builds if item.id == context.report_build_id), None)
            if run is None or build is None or not build.report_id:
                # Say which of the three it was. "without submit_report_draft"
                # was emitted for all of them, including the common case where
                # the tool *was* called and the draft was refused — which sent
                # whoever read the failure looking for a runtime that skipped a
                # tool call, when the actual record showed two calls and two
                # rejections.
                raise RuntimeProtocolError(self._report_failure_detail(run, build))
            artifact = await uow.reports.get_artifact(
                build.report_id, session_id=context.session_id
            )
            if artifact is None:
                raise RuntimeProtocolError("the accepted report artifact cannot be reconstructed")
            if run.status is not RunStatus.RUNNING:
                raise RuntimeProtocolError(
                    "the run changed before its accepted report could be committed",
                    status=run.status.value,
                )
            sequence = await uow.messages.next_sequence(context.session_id)
            reply = Message.create(
                context.session_id, Role.ASSISTANT, sequence, now=_now(),
                parts=(
                    (PartType.TEXT, {
                        "text": f"Report completed: {artifact['title']}",
                    }),
                    (PartType.REPORT_REF, {
                        "report_id": artifact["report_id"],
                        "report_build_id": build.id,
                        "status": artifact["status"],
                    }),
                ),
            )
            await uow.messages.add(reply)
            uow.emit(
                session_id=context.session_id, type=EventType.MESSAGE_CREATED,
                entity_type="message", entity_id=reply.id, run_id=context.run_id,
                payload={"role": "assistant", "report_id": artifact["report_id"]},
            )
            await advance(
                uow, run, RunStatus.COMPLETED,
                payload={"report_id": artifact["report_id"], "report_build_id": build.id},
            )
            await uow.commit()

    @staticmethod
    def _report_failure_detail(run, build) -> str:
        """Why no report exists, in the terms the record actually supports."""
        if run is None:
            return "the runtime run disappeared before completion"
        if build is None:
            return "the report build this run was started for no longer resolves"
        if build.stage is BuildStage.FAILED:
            reason = build.failure_detail or build.failure_code or "no reason recorded"
            return (
                "submit_report_draft was called and the draft did not pass validation, so "
                f"the build failed and no report was produced: {reason}"
            )
        return (
            "the runtime reached a terminal event without a report: the build is "
            f"{build.stage.value} and submit_report_draft never produced an artifact"
        )
