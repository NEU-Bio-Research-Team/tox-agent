"""Sessions, their settings, and messages."""
from __future__ import annotations

from fastapi import Depends, Query, Request

from ...application.conversation.submit_message import MessageSubmission
from ...application.policy import Actor
from ..responses import MessageListResponse, SessionListResponse, SessionProjection, SessionSettings
from ..schemas import (
    AcceptedResponse,
    CreateSessionRequest,
    SendMessageRequest,
    SessionResponse,
    SessionSettingsRequest,
    UpdateSessionRequest,
)
from ._common import _decode_image, _services, actor, router


@router.get("/sessions", responses={200: {"model": SessionListResponse}})
async def list_sessions(
    request: Request,
    limit: int = Query(25, ge=1, le=50),
    offset: int = Query(0, ge=0),
    principal: Actor = Depends(actor),
):
    return await _services(request).sessions.list(principal, limit=limit, offset=offset)


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    request: Request, body: CreateSessionRequest, principal: Actor = Depends(actor)
):
    session = await _services(request).sessions.create(
        principal,
        preferred_language=body.preferred_language,
        title=body.title,
        client_session_id=body.client_session_id,
    )
    return SessionResponse(
        session_id=session.id,
        status=session.status.value,
        preferred_language=session.preferred_language.value,
        title=session.title,
        created_at=session.created_at.isoformat(),
        version=session.version,
        title_source=session.title_source.value if session.title_source else None,
        title_status=session.title_status,
    )


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(
    request: Request, session_id: str, body: UpdateSessionRequest, principal: Actor = Depends(actor)
):
    session = await _services(request).sessions.rename(
        principal, session_id, title=body.title, expected_version=body.expected_version
    )
    return SessionResponse(
        session_id=session.id, status=session.status.value,
        preferred_language=session.preferred_language.value, title=session.title,
        created_at=session.created_at.isoformat(), version=session.version,
        title_source=session.title_source.value if session.title_source else None,
        title_status=session.title_status,
    )


@router.get("/sessions/{session_id}", responses={200: {"model": SessionProjection}})
async def get_session(request: Request, session_id: str, principal: Actor = Depends(actor)):
    return await _services(request).sessions.projection(principal, session_id)


@router.get("/sessions/{session_id}/settings", responses={200: {"model": SessionSettings}})
async def get_session_settings(request: Request, session_id: str, principal: Actor = Depends(actor)):
    return await _services(request).sessions.settings(principal, session_id)


@router.patch("/sessions/{session_id}/settings", responses={200: {"model": SessionSettings}})
async def update_session_settings(request: Request, session_id: str, body: SessionSettingsRequest,
                                  principal: Actor = Depends(actor)):
    return await _services(request).sessions.update_settings(
        principal, session_id, ai_profile_id=body.ai_profile_id,
        predictor_bindings=dict(body.predictor_bindings),
    )


@router.get("/sessions/{session_id}/messages", responses={200: {"model": MessageListResponse}})
async def list_messages(
    request: Request,
    session_id: str,
    after_sequence: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    principal: Actor = Depends(actor),
):
    messages = await _services(request).sessions.messages(
        principal, session_id, after_sequence=after_sequence, limit=limit
    )
    return {"messages": messages, "count": len(messages)}


@router.post("/sessions/{session_id}/messages", response_model=AcceptedResponse, status_code=202)
async def send_message(
    request: Request, session_id: str, body: SendMessageRequest, principal: Actor = Depends(actor)
):
    options = body.analysis_options
    molecule = body.molecule
    image_mime_type, image_size_bytes, image_bytes = _decode_image(body.image)
    accepted = await _services(request).submit_message.execute(
        actor=principal,
        session_id=session_id,
        submission=MessageSubmission(
            text=body.text,
            client_message_id=body.client_message_id,
            intent_hint=body.intent_hint,
            smiles=molecule.smiles if molecule else None,
            batch_smiles=tuple(molecule.batch_smiles or ()) if molecule else (),
            endpoints=tuple(options.endpoints) if options and options.endpoints else None,
            model_selection=options.model_selection if options else None,
            threshold_overrides=options.threshold_overrides if options else None,
            include_attribution=options.include_attribution if options else False,
            explanation_mode=options.explanation_mode if options else "on_demand",
            explanation_targets=tuple((target.endpoint, target.task) for target in (options.explanation_targets if options else [])),
            analysis_id=body.analysis_id,
            image_mime_type=image_mime_type,
            image_size_bytes=image_size_bytes,
            image_bytes=image_bytes,
        ),
    )
    return AcceptedResponse(**accepted.to_dict())
