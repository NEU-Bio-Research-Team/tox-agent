"""Read access to a session's analyses, attributions, answers, observations and evidence."""
from __future__ import annotations

from typing import Any

from fastapi import Depends, Query, Request

from ...application.policy import Actor
from ...domain.errors import (
    AnalysisNotFound,
    NotFound,
)
from ...domain.evidence import EvidenceStatus
from ...domain.observation import ObservationKind
from ._common import _services, actor, router


@router.get("/sessions/{session_id}/analyses/{analysis_id}")
async def get_analysis(
    request: Request,
    session_id: str,
    analysis_id: str,
    include_raw: bool = Query(False),
    principal: Actor = Depends(actor),
):
    from ...application.conversation.projections import display_projection

    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        snapshot = await uow.analyses.get(analysis_id, session_id=session_id)
    if snapshot is None:
        raise AnalysisNotFound("no such analysis", analysis_id=analysis_id)
    projection = display_projection(snapshot)
    if include_raw and principal.has_role("auditor"):
        # The lossless payload is audit material, not a default response body.
        projection["predictor_response"] = snapshot.predictor_response
    return projection


@router.get("/sessions/{session_id}/analyses/{analysis_id}/attributions")
async def list_attributions(
    request: Request,
    session_id: str,
    analysis_id: str,
    principal: Actor = Depends(actor),
):
    """List bounded attribution observations for one immutable analysis.

    The projection is intentionally the same top-token view supplied to the
    model, rather than canonical/raw provider output. An attribution belongs
    to exactly one endpoint (and one Tox21 task when applicable), so the UI
    cannot construct an aggregate explanation from this endpoint.
    """
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        snapshot = await uow.analyses.get(analysis_id, session_id=session_id)
        if snapshot is None:
            raise AnalysisNotFound("no such analysis", analysis_id=analysis_id)
        observations = await uow.observations.list_for_analysis(analysis_id)
    return {
        "attributions": [
            {
                "observation_id": observation.id,
                "run_id": observation.run_id,
                "created_at": observation.created_at.isoformat(),
                "content_sha256": observation.content_sha256,
                "required_limitations": list(observation.required_limitations),
                **observation.model_projection,
            }
            for observation in observations
            if observation.kind is ObservationKind.ATTRIBUTION
        ]
    }


@router.get("/sessions/{session_id}/answers/{answer_id}")
async def get_answer(
    request: Request, session_id: str, answer_id: str, principal: Actor = Depends(actor)
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        answer = await uow.answers.get(answer_id, session_id=session_id)
    if answer is None:
        raise NotFound("no such answer", answer_id=answer_id)
    return answer.to_dict()


@router.get("/sessions/{session_id}/observations/{observation_id}")
async def get_observation(
    request: Request,
    session_id: str,
    observation_id: str,
    principal: Actor = Depends(actor),
):
    """The other end of every claim's ``observation_id`` (plan section 5.5).

    Without this, ``field_path`` and ``source_value`` on a claim are citations
    to nothing a client can open. The lossless ``canonical_payload`` stays
    audit-only, same gate as ``analyses`` ``include_raw``: a claim only ever
    needed the bounded ``model_projection`` to be valid.
    """
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        observation = await uow.observations.get(observation_id, session_id=session_id)
    if observation is None:
        raise NotFound("no such observation", observation_id=observation_id)
    body: dict[str, Any] = {
        "observation_id": observation.id,
        "run_id": observation.run_id,
        "producer": observation.producer.value,
        "kind": observation.kind.value,
        "schema_version": observation.schema_version,
        "model_projection": observation.model_projection,
        "provenance": observation.provenance,
        "required_limitations": list(observation.required_limitations),
        "content_sha256": observation.content_sha256,
        "created_at": observation.created_at.isoformat(),
    }
    if principal.has_role("auditor"):
        body["canonical_payload"] = observation.canonical_payload
    return body


@router.get("/sessions/{session_id}/evidence")
async def list_evidence(
    request: Request,
    session_id: str,
    status: str = Query("accepted"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        records = await uow.evidence.list_for_session(
            session_id,
            status=EvidenceStatus(status) if status != "all" else None,
            limit=limit, offset=offset,
        )
    return {
        "evidence": [
            {
                **record.model_view(),
                "status": record.status.value,
                "provider": record.provider,
                "retrieved_at": record.retrieved_at.isoformat(),
                "content_sha256": record.content_sha256,
            }
            for record in records
        ],
        "count": len(records),
    }


@router.get("/sessions/{session_id}/evidence/{evidence_id}")
async def get_evidence(
    request: Request,
    session_id: str,
    evidence_id: str,
    principal: Actor = Depends(actor),
):
    """Return the bounded, normalized evidence projection for its owner.

    This deliberately mirrors the model-visible view, plus audit-safe
    transport metadata. ``raw_payload_ref`` remains object-store/auditor
    material (W4-09) and is never a browser URL or a model capability.
    """
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        record = await uow.evidence.get(evidence_id, session_id=session_id)
    if record is None:
        raise NotFound("no such evidence", evidence_id=evidence_id)
    return {
        **record.model_view(),
        "status": record.status.value,
        "provider": record.provider,
        "retrieved_at": record.retrieved_at.isoformat(),
        "content_sha256": record.content_sha256,
    }
