"""Reports, their renderings and figures, and report builds."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from fastapi import Depends, Query, Request
from fastapi.responses import Response

from ...application.conversation.submit_message import MessageSubmission
from ...application.policy import Actor
from ...domain.errors import (
    NotFound,
)
from ...domain.events import EventType
from ...persistence.object_store import ObjectNotFound, ObjectRef
from ..responses import ReportArtifact, ReportBuildView, ReportListResponse
from ..schemas import (
    AcceptedResponse,
    CancelResponse,
    CreateReportRequest,
)
from ._common import _services, actor, router


@router.post("/sessions/{session_id}/reports", response_model=AcceptedResponse, status_code=202)
async def create_report(
    request: Request, session_id: str, body: CreateReportRequest,
    principal: Actor = Depends(actor),
):
    """Admit a report build through the same durable run path as chat."""
    molecule = body.molecule
    targets = tuple(("tox21", task) for task in body.selected_tox21_tasks)
    accepted = await _services(request).submit_message.execute(
        actor=principal,
        session_id=session_id,
        submission=MessageSubmission(
            text="Build a complete toxicity screening report.",
            intent_hint="build_report",
            smiles=molecule.smiles if molecule else None,
            endpoints=tuple(body.selected_endpoints) if body.selected_endpoints else None,
            explanation_mode="required" if body.include_explanations else "none",
            explanation_targets=targets,
            analysis_id=body.analysis_id,
            report_language=body.report_language,
            report_audience=body.audience,
            include_external_evidence=body.include_external_evidence,
            report_output_formats=tuple(body.output_formats),
        ),
    )
    return AcceptedResponse(**accepted.to_dict())


@router.get("/sessions/{session_id}/reports", responses={200: {"model": ReportListResponse}})
async def list_reports(
    request: Request, session_id: str, limit: int = Query(50, ge=1, le=200),
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        reports = await uow.reports.list_artifacts_for_session(session_id, limit=limit)
    return {"reports": reports, "count": len(reports)}


@router.get("/sessions/{session_id}/reports/{report_id}", responses={200: {"model": ReportArtifact}})
async def get_report(
    request: Request, session_id: str, report_id: str,
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        report = await uow.reports.get_artifact(report_id, session_id=session_id)
    if report is None:
        raise NotFound("no such report", report_id=report_id)
    return report


@router.get("/sessions/{session_id}/reports/{report_id}/renderings/{format}")
async def download_report_rendering(
    request: Request, session_id: str, report_id: str, format: str,
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        rendering = await uow.reports.get_rendering(
            report_id, format, session_id=session_id
        )
    if rendering is None:
        raise NotFound("no such report rendering", report_id=report_id, format=format)
    try:
        data = await services.object_store.get(ObjectRef(rendering["object_uri"]))
    except ObjectNotFound as exc:
        raise NotFound("the report rendering bytes are unavailable") from exc
    suffix = {"markdown": "md", "html": "html", "pdf": "pdf"}.get(format, format)
    return Response(
        data, media_type=rendering["media_type"],
        headers={"Content-Disposition": f'attachment; filename="{report_id}.{suffix}"'},
    )


@router.get("/sessions/{session_id}/reports/{report_id}/figures/{figure_id}")
async def get_report_figure(
    request: Request, session_id: str, report_id: str, figure_id: str,
    principal: Actor = Depends(actor),
):
    """One figure's bytes, scoped to the report that shows it (REP-02).

    Three checks, none of them optional. The session must belong to the caller;
    the figure must be one this report actually carries — otherwise a valid
    figure id from another report in the same session would serve an image the
    caller was never shown; and the bytes must hash to what the figure claims,
    because a report's integrity guarantee covers its pictures and an image is
    the one part of a document nobody proofreads.

    Cached immutably: the artifact is immutable and the object key is the content
    hash, so the same URL can only ever return the same bytes. ``private``
    because the URL is session-scoped and a shared cache serving it to a second
    caller would be serving them someone else's report.
    """
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        report = await uow.reports.get_artifact(report_id, session_id=session_id)
        if report is None:
            raise NotFound("no such report", report_id=report_id)
        carried = {f["figure_id"] for f in report.get("figures") or ()}
        if figure_id not in carried:
            # 404 rather than 403: whether a figure exists elsewhere is not
            # something this endpoint should be willing to confirm.
            raise NotFound(
                "this report does not carry that figure",
                report_id=report_id, figure_id=figure_id,
            )
        row = await uow.reports.get_figure(figure_id, session_id=session_id)
        if row is None:
            raise NotFound("no such figure", figure_id=figure_id)
        attachment = await uow.attachments.get(
            row["attachment_id"], owner_id=principal.subject_id
        )
    if attachment is None or attachment.session_id != session_id:
        raise NotFound("the figure's attachment is unavailable", figure_id=figure_id)
    if attachment.media_type != row["media_type"]:
        raise NotFound("the figure's stored media type does not match", figure_id=figure_id)
    try:
        data = await services.object_store.get(ObjectRef(attachment.object_uri))
    except ObjectNotFound as exc:
        raise NotFound("the figure bytes are unavailable", figure_id=figure_id) from exc
    if hashlib.sha256(data).hexdigest() != row["content_sha256"]:
        raise NotFound(
            "the stored figure bytes do not match the recorded content hash",
            figure_id=figure_id,
        )
    return Response(
        data,
        media_type=row["media_type"],
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            "ETag": f'"{row["content_sha256"]}"',
            # An SVG served as a document could script; served as an image it
            # cannot. Belt and braces over the sanitizer, which is the thing
            # actually relied on.
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'inline; filename="{figure_id}.svg"',
        },
    )


@router.get("/sessions/{session_id}/report-builds/{build_id}", responses={200: {"model": ReportBuildView}})
async def get_report_build(
    request: Request, session_id: str, build_id: str,
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        build = await uow.reports.get_build(build_id, session_id=session_id)
    if build is None:
        raise NotFound("no such report build", build_id=build_id)
    return build.to_dict()


@router.post("/sessions/{session_id}/report-builds/{build_id}:cancel", response_model=CancelResponse)
async def cancel_report_build(
    request: Request, session_id: str, build_id: str,
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        build = await uow.reports.get_build(build_id, session_id=session_id)
    if build is None:
        raise NotFound("no such report build", build_id=build_id)
    outcome = await services.scheduler.cancel(build.run_id)
    if outcome.requested:
        from ...domain.report import BuildStage
        async with services.database.unit_of_work() as uow:
            current = await uow.reports.get_build(build_id, session_id=session_id)
            if current is not None and not current.is_terminal:
                await uow.reports.save_build(current.advance(BuildStage.CANCELLED, now=datetime.now(timezone.utc)))
                uow.emit(
                    session_id=session_id, type=EventType.REPORT_CANCELLED,
                    entity_type="report_build", entity_id=build_id, run_id=build.run_id,
                    payload={"stage": "cancelled"},
                )
                await uow.commit()
    return CancelResponse(**outcome.to_dict())
