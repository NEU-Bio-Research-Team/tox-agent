"""Where a waiting run stands (WS08 step 6 / PR-15).

With external workers a run can sit in `queued` for a while, and "queued" alone
does not tell anyone whether it is next, behind forty others, or waiting for a
concurrency slot that frees in seconds. This is the projection that does.

``position`` is an estimate and is labelled as one: the count of unowned jobs
ahead in the same queue at the moment of the read. Priorities, deferrals and
other workers' claims all move it.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from ..application.policy import Actor
from ..application.runs.queues import queue_of_job
from ..domain.errors import NotFound
from .responses import RunQueueStatus
from .routes import _services, actor

router = APIRouter(prefix="/v1", tags=["toxagent"])


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@router.get("/sessions/{session_id}/runs/{run_id}/queue", responses={200: {"model": RunQueueStatus}})
async def get_run_queue(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    now = datetime.now(timezone.utc)
    async with services.database.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise NotFound("no such run", run_id=run_id)
        job = await uow.run_jobs.position(run_id, now=now)

    if job is None:
        return {"run_id": run_id, "state": "not_queued", "status": run.status.value}

    available_at = _as_utc(job.get("available_at"))
    code = job.get("last_error_code") or ""
    if job.get("worker_id") is not None:
        state = "claimed"
    elif available_at is not None and available_at > now:
        state = "deferred"
    else:
        state = "waiting"
    retry_after_s = (
        max(0, round((available_at - now).total_seconds())) if state == "deferred" else None
    )
    return {
        "run_id": run_id,
        "status": run.status.value,
        "queue_name": queue_of_job(job),
        "state": state,
        "position_estimate": job["ahead"] if state != "claimed" else 0,
        "retry_after_s": retry_after_s,
        # `quota_wait:<scope>` names which cap is holding it, never whose.
        "waiting_on": code.split(":", 1)[1] if code.startswith("quota_wait:") else None,
        "attempts": job.get("attempts", 0),
    }
