"""Frozen activity screenings attached to a session (#44)."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, Request

from ...application.activity import replay
from ...application.policy import Actor
from ...domain.errors import NotFound, ScreeningNotFound
from ...domain.subject import activity_subject_key
from ..responses import ScreeningDetail, ScreeningList, ScreeningSummary
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


def detail_view(attachment) -> dict:
    """The screening read model (#44). Stage statuses are stated, never
    inferred: a stage that did not run says so, with its reason."""
    record = attachment.record
    state = record.get("screening_state") or {}
    toxicity = state.get("toxicity") or {}
    estimate = record.get("activity_estimate")
    decision = record.get("prioritisation_decision")
    activity_failed = state.get("activity_failure_code")
    link = replay.linked_toxicity(record)
    refs = [{"kind": "screening", "source_id": attachment.screening_id,
             "content_sha256": attachment.record_sha256}]
    if estimate is not None:
        refs.append({"kind": "activity_estimate", "source_id": estimate.get("activity_snapshot_id"),
                     "content_sha256": estimate.get("content_sha256")})
    if decision is not None:
        refs.append({"kind": "prioritisation_decision",
                     "source_id": decision.get("prioritisation_decision_id"),
                     "content_sha256": None})
    if link is not None:
        refs.append({"kind": "toxicity_snapshot", "source_id": link["origin_analysis_id"],
                     "content_sha256": link["content_sha256"]})
    if toxicity.get("run"):
        toxicity_status = "completed"
    elif toxicity.get("failure_code"):
        toxicity_status = "failed"
    else:
        toxicity_status = "not_run"
    subject = state.get("subject") or {}
    return {
        "screening_id": attachment.screening_id,
        "subject": subject,
        "subject_key": activity_subject_key(subject),
        "intended_target_role": state.get("intended_target_role"),
        "status": state.get("status"),
        "stages": {
            "activity": {
                "status": "failed" if activity_failed else (
                    "completed" if estimate is not None else "not_run"),
                "failure_code": activity_failed,
            },
            "prioritisation": {
                "status": "completed" if decision is not None else "not_run",
                "decision": (decision or {}).get("status"),
            },
            "toxicity": {
                "status": toxicity_status,
                "not_run_reason": toxicity.get("not_run_reason"),
                "failure_code": toxicity.get("failure_code"),
                "authorised_by": toxicity.get("authorised_by"),
                "frozen_analysis_available": link is not None,
            },
        },
        "activity_estimate": estimate,
        "prioritisation_decision": decision,
        "artifact_refs": refs,
        "required_limitations": list(record.get("required_limitations") or ()),
        "mode": "replay",
        "serving": False,
        "model_admitted": ((estimate or {}).get("model") or {}).get("admitted"),
        "record_sha256": attachment.record_sha256,
        "attached_at": attachment.attached_at.isoformat(),
    }


@router.get(
    "/sessions/{session_id}/screenings/{screening_id}",
    responses={200: {"model": ScreeningDetail}},
)
async def get_screening(request: Request, session_id: str, screening_id: str,
                        principal: Actor = Depends(actor)):
    _replay_enabled()
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        attachment = await uow.screenings.get(screening_id, session_id=session_id)
    if attachment is None:
        raise ScreeningNotFound("no such screening attached to this session",
                                screening_id=screening_id)
    return detail_view(attachment)
