"""The session event stream and its list form."""
from __future__ import annotations

from fastapi import Depends, Query, Request
from sse_starlette.sse import EventSourceResponse

from ...application.policy import Actor
from ...streaming.sse import event_stream
from ..responses import EventListResponse
from ._common import _services, actor, router


@router.get("/sessions/{session_id}/events")
async def stream_events(
    request: Request,
    session_id: str,
    after_sequence: int = Query(0, ge=0),
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    last_event_id = request.headers.get("last-event-id")
    cursor = after_sequence
    if last_event_id and last_event_id.isdigit():
        # Last-Event-ID wins: it is what the browser resends automatically, and
        # a stale query parameter would silently replay events the client has.
        cursor = int(last_event_id)
    return EventSourceResponse(
        event_stream(services.database.outbox(), services.notifier, session_id, after_sequence=cursor)
    )


@router.get("/sessions/{session_id}/events:list", responses={200: {"model": EventListResponse}})
async def list_events(
    request: Request,
    session_id: str,
    after_sequence: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=500),
    run_id: str | None = Query(None),
    principal: Actor = Depends(actor),
):
    """A non-streaming read of the same outbox the SSE feed serves.

    The stream never terminates, so it is the wrong tool for "replay
    everything that happened in this one run" (a Run Inspector opened after
    the fact, or a page that reconnected and needs to fill a gap). The outbox
    row is retained forever — nothing here is a delivery guarantee beyond what
    ``/events`` already gives; this just lets a client stop listening.
    """
    services = _services(request)
    session = await services.sessions.get(principal, session_id)
    events = await services.database.outbox().read_after(
        session_id, after_sequence, limit=limit, run_id=run_id
    )
    return {
        "events": [e.to_dict() for e in events],
        "count": len(events),
        "latest_sequence": session.event_sequence,
    }
