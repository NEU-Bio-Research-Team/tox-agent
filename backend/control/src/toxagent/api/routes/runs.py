"""Runs: status, decision state, evidence relations, cancellation."""
from __future__ import annotations

from fastapi import Depends, Request

from ...application.conversation.sessions import run_projection
from ...application.policy import Actor
from ...domain.errors import (
    NotFound,
)
from ..schemas import (
    CancelResponse,
)
from ._common import _isoformat, _services, actor, router


@router.get("/sessions/{session_id}/runs/{run_id}")
async def get_run(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise NotFound("no such run", run_id=run_id)
        binding = (
            await uow.runtime_bindings.get(run.runtime_binding_id)
            if run.runtime_binding_id else None
        )
        tool_calls = await uow.tool_calls.list_for_run(run_id)
        usage_events = await uow.runtime_usage.list_for_run(run_id)
        configuration_snapshot = await uow.run_configuration_snapshots.get(run_id)
    projection = run_projection(run)
    projection["runtime"] = binding.manifest() if binding else None
    projection["configuration_snapshot"] = configuration_snapshot
    projection["usage"] = {
        # No runtime event is different from an explicit event containing
        # input=0/output=0. Consumers must not turn unavailable into zero.
        "status": "reported" if usage_events else "unknown",
        "events": [event.to_dict() for event in usage_events],
    }
    projection["tool_calls"] = [
        {
            "call_id": c["id"], "tool_name": c["tool_name"], "status": c["status"],
            "error_code": c["error_code"], "duration_ms": c["duration_ms"],
            # A hash, never the arguments: enough for a trajectory grader to
            # see a repeated call without exposing what the model sent.
            "arguments_sha256": c.get("arguments_sha256"),
            "observation_ids": list(c.get("observation_ids") or ()),
            "started_at": _isoformat(c["started_at"]),
            "ended_at": _isoformat(c["ended_at"]),
        }
        for c in tool_calls
    ]
    return projection


@router.get("/sessions/{session_id}/runs/{run_id}/decision-state")
async def get_decision_state(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    """The run's DecisionSupportStateV1: goal, propositions, coverage, usage
    and stop reason. 404 for a run that keeps none (every non-decision_support
    run, and decision_support runs admitted before the state existed)."""
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise NotFound("no such run", run_id=run_id)
        state = await uow.decision_states.get(run_id)
    if state is None:
        raise NotFound("this run keeps no decision-support state", run_id=run_id)
    return state.to_dict()


@router.get("/sessions/{session_id}/runs/{run_id}/evidence-relations")
async def list_evidence_relations(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    """What the run judged each source to say about each proposition.

    The relations an accepted grounded-answer-v2 draft proposed and the server
    resolved (``domain/evidence_relation.py``) were stored and never readable:
    a client could see the answer and the sources it cited, but not the
    product's own ``supports``/``contradicts`` assessment. The proposition text
    is in the same run's ``decision-state``, keyed by ``proposition_id``.
    Empty for a run whose answer schema carries no relations (W7-04).
    """
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise NotFound("no such run", run_id=run_id)
        relations = await uow.evidence_relations.list_for_run(run_id)
    return {"evidence_relations": [relation.to_dict() for relation in relations]}


@router.post("/sessions/{session_id}/runs/{run_id}:cancel", response_model=CancelResponse)
async def cancel_run(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise NotFound("no such run", run_id=run_id)
        binding = (
            await uow.runtime_bindings.get(run.runtime_binding_id)
            if run.runtime_binding_id else None
        )
    supported = bool(binding and binding.capabilities.cancel_turn)
    outcome = await services.scheduler.cancel(run_id, runtime_cancel_supported=supported)
    return CancelResponse(**outcome.to_dict())
