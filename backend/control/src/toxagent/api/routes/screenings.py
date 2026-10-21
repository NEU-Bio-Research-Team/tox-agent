"""Frozen activity screenings attached to a session (#44)."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, Request

from ...application.activity import replay
from ...application.policy import Actor
from ...domain.errors import NotFound
from ..responses import ScreeningList, ScreeningSummary
from ..schemas import AttachScreeningRequest
from ._common import _services, actor, router


def _replay_enabled() -> None:
    from ...platform.flags import is_enabled

    if not is_enabled("activity_replay_v1"):
        raise NotFound("activity screenings are not enabled on this deployment")


def _view(attachment) -> dict:
    return {
        **replay.summary(attachment.record),
        "source": attachment.source,
        "attached_at": attachment.attached_at.isoformat(),
    }


@router.post(
    "/sessions/{session_id}/screenings", status_code=201,
    responses={201: {"model": ScreeningSummary}},
)
async def attach_screening(
    request: Request, session_id: str, body: AttachScreeningRequest,
    principal: Actor = Depends(actor),
):
    _replay_enabled()
    services = _services(request)
    await services.sessions.get(principal, session_id)
    attached = await replay.attach(
        services.database, getattr(request.app.state, "activity_replay", None),
        session_id=session_id, screening_id=body.screening_id,
        now=datetime.now(timezone.utc),
    )
    return _view(attached)


@router.get("/sessions/{session_id}/screenings", responses={200: {"model": ScreeningList}})
async def list_screenings(request: Request, session_id: str, principal: Actor = Depends(actor)):
    _replay_enabled()
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        attachments = await uow.screenings.list_for_session(session_id)
    return {"screenings": [_view(a) for a in attachments], "mode": "replay"}
