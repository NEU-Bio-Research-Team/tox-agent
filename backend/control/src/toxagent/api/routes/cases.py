"""Scientific cases and decision dossiers."""
from __future__ import annotations

from typing import Any

from fastapi import Depends, Request

from ...application.policy import Actor
from ...domain.errors import (
    InvalidRequest,
    NotFound,
)
from ..responses import (
    DecisionDossier,
    ScientificCase,
    ScientificCaseEventsResponse,
    ScientificCaseListResponse,
)
from ..schemas import (
    CaseContextRequest,
    CaseQuestionRequest,
    CaseScopeRequest,
)
from ._common import _services, actor, router


def _case_summary(case) -> dict[str, Any]:
    return {
        "case_id": case.id, "question": case.question, "subject_key": case.subject_key,
        "subject_refs": list(case.subject_refs), "status": case.status,
        "revision": case.revision, "hypotheses": len(case.hypotheses),
        "evidence": len(case.evidence), "open_uncertainties": len(case.open_uncertainties),
        "runs": len(case.runs), "coverage": case.coverage, "updated_at": case.updated_at,
        "requester": case.requester, "external_search": case.data_scope.external_search,
    }


async def _owned_case(request: Request, principal: Actor, session_id: str, case_id: str):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        case = await uow.scientific_cases.get(case_id, session_id=session_id)
    if case is None:
        raise NotFound("no such scientific case", case_id=case_id)
    return services, case


@router.get("/sessions/{session_id}/cases", responses={200: {"model": ScientificCaseListResponse}})
async def list_scientific_cases(
    request: Request, session_id: str, principal: Actor = Depends(actor)
):
    """The session's scientific cases, newest first. Empty on a deployment
    that never had ``scientific_case_v1`` on."""
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        cases = await uow.scientific_cases.list_for_session(session_id)
    return {"cases": [_case_summary(case) for case in cases]}


@router.get("/sessions/{session_id}/cases/{case_id}", responses={200: {"model": ScientificCase}})
async def get_scientific_case(
    request: Request, session_id: str, case_id: str, principal: Actor = Depends(actor)
):
    _, case = await _owned_case(request, principal, session_id, case_id)
    return case.to_dict()


@router.get("/sessions/{session_id}/cases/{case_id}/events", responses={200: {"model": ScientificCaseEventsResponse}})
async def get_scientific_case_events(
    request: Request, session_id: str, case_id: str, principal: Actor = Depends(actor)
):
    """The append-only log the case is the fold of: who changed what, in which run."""
    services, case = await _owned_case(request, principal, session_id, case_id)
    async with services.database.unit_of_work() as uow:
        events = await uow.scientific_cases.events(case.id, session_id=session_id)
    return {"case_id": case.id, "events": [event.to_dict() for event in events]}


@router.get("/sessions/{session_id}/cases/{case_id}/dossier", responses={200: {"model": DecisionDossier}})
async def get_latest_decision_dossier(
    request: Request, session_id: str, case_id: str, principal: Actor = Depends(actor)
):
    services, case = await _owned_case(request, principal, session_id, case_id)
    async with services.database.unit_of_work() as uow:
        dossier = await uow.scientific_cases.latest_dossier(case.id, session_id=session_id)
    if dossier is None:
        raise NotFound("no run of this case has finished yet", case_id=case_id)
    return dossier


@router.get("/sessions/{session_id}/runs/{run_id}/dossier", responses={200: {"model": DecisionDossier}})
async def get_run_decision_dossier(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    """The DecisionDossierV1 one run compiled. 404 for a run without a case."""
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        dossier = await uow.scientific_cases.get_dossier(run_id, session_id=session_id)
    if dossier is None:
        raise NotFound("this run compiled no decision dossier", run_id=run_id)
    return dossier


async def _user_case_update(request: Request, principal: Actor, session_id: str, case_id: str,
                            op: str, **payload: Any):
    from ...application.investigation import scientific_case_service
    from ...domain import scientific_case as sc
    from ...platform.flags import is_enabled

    if not is_enabled("scientific_case_v1"):
        raise NotFound("scientific cases are not enabled on this deployment")
    services, case = await _owned_case(request, principal, session_id, case_id)
    async with services.database.unit_of_work() as uow:
        try:
            updated = await scientific_case_service.apply_updates(
                uow, case_id=case.id, session_id=session_id,
                updates=[scientific_case_service.update(op, actor=sc.Actor.USER.value, **payload)],
            )
        except sc.InvalidCaseUpdate as exc:
            raise InvalidRequest(str(exc), case_id=case_id) from None
        await uow.commit()
    return updated.to_dict()


@router.post("/sessions/{session_id}/cases/{case_id}/context", responses={200: {"model": ScientificCase}})
async def add_scientific_case_context(
    request: Request, session_id: str, case_id: str, body: CaseContextRequest,
    principal: Actor = Depends(actor),
):
    """Add researcher-supplied context; the next turn sees it in the case."""
    return await _user_case_update(
        request, principal, session_id, case_id, "add_context",
        key=body.key, value=body.value, note=body.note or "",
    )


@router.post("/sessions/{session_id}/cases/{case_id}/question", responses={200: {"model": ScientificCase}})
async def set_scientific_case_question(
    request: Request, session_id: str, case_id: str, body: CaseQuestionRequest,
    principal: Actor = Depends(actor),
):
    payload: dict[str, Any] = {"question": body.question}
    if body.decision_context:
        payload["decision_context"] = body.decision_context
    return await _user_case_update(request, principal, session_id, case_id, "set_question", **payload)


@router.post("/sessions/{session_id}/cases/{case_id}/scope", responses={200: {"model": ScientificCase}})
async def set_scientific_case_scope(
    request: Request, session_id: str, case_id: str, body: CaseScopeRequest,
    principal: Actor = Depends(actor),
):
    """Set what the case may reach. With external search off, the evidence
    search tool refuses for every later turn of this case."""
    return await _user_case_update(
        request, principal, session_id, case_id, "set_scope",
        external_search=body.external_search, reason=body.reason or "",
    )


@router.post("/sessions/{session_id}/cases/{case_id}:close", responses={200: {"model": ScientificCase}})
async def close_scientific_case(
    request: Request, session_id: str, case_id: str, principal: Actor = Depends(actor)
):
    """Close the case; the next turn on this subject opens a new one."""
    return await _user_case_update(request, principal, session_id, case_id, "close")
