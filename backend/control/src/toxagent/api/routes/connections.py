"""Model-provider connections: create, list, probe, change model, delete — and
ChatGPT Plus/Pro sign-in.

Every route acts for the authenticated owner only. A credential goes in once
(create, or a model listing that is not stored) and is never returned: not in
a response, not in an error, not in a log.
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends, Request

from ...application.policy import Actor
from ...connections import providers
from ...connections.chatgpt import ChatGPTUnavailable, EnrollmentError
from ...connections.models import ModelListError
from ...connections.probe import ProbeError
from ...connections.service import ConnectionNotFound
from ...domain.errors import (
    CapabilityUnavailable,
    Conflict,
    InvalidRequest,
    NotFound,
)
from ...domain.runtime import AuthMode
from ..responses import (
    ChatGPTEnrollment,
    ModelConnection,
    ModelConnectionList,
    ModelOptionList,
    SupportedProviderList,
)
from ..schemas import (
    CompleteChatGPTRequest,
    CreateModelConnectionRequest,
    ListModelsRequest,
    UpdateModelConnectionRequest,
)
from ._common import _services, actor, router


def _chatgpt(request: Request):
    return request.app.state.chatgpt


@router.get("/model-connections:providers", response_model=None, responses={200: {"model": SupportedProviderList}})
async def list_supported_providers(request: Request) -> dict[str, Any]:
    """Providers this control plane has an adapter for, and their defaults.

    The UI used to carry its own list, which included providers whose wire
    format nothing here speaks and whose base URL default was blank — so the
    form could be filled in correctly and still fail every probe (I13). One
    list, served by the code that decides. ChatGPT sign-in is listed only
    where this deployment can actually run it.
    """
    return {"providers": providers.catalogue(chatgpt=_chatgpt(request).availability.to_dict())}


@router.post("/model-connections:models", responses={200: {"model": ModelOptionList}})
async def list_models_for_key(
    request: Request, body: ListModelsRequest, principal: Actor = Depends(actor)
):
    """The models a key can use, before it is saved. The key is not stored."""
    del principal
    try:
        models = await _services(request).connections.list_models(
            provider_id=body.provider_id, base_url=body.base_url, credential=body.credential
        )
    except providers.UnsupportedProvider as exc:
        raise InvalidRequest(exc.reason, provider_id=exc.provider_id) from exc
    except ModelListError as exc:
        raise InvalidRequest(str(exc), model_list_failure=exc.kind) from exc
    return {"models": [model.to_dict() for model in models]}


@router.post("/model-connections", status_code=201, responses={201: {"model": ModelConnection}})
async def create_model_connection(
    request: Request, body: CreateModelConnectionRequest, principal: Actor = Depends(actor)
):
    try:
        item = await _services(request).connections.create(
            owner_id=principal.subject_id, provider_id=body.provider_id, model_id=body.model_id,
            auth_mode=AuthMode(body.auth_mode), base_url=body.base_url, credential=body.credential,
            display_name=body.display_name,
        )
    except providers.UnsupportedProvider as exc:
        # The reason is the useful part: it says what to pick instead.
        raise InvalidRequest(exc.reason, provider_id=exc.provider_id) from exc
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc
    return item.public_dict()


@router.get("/model-connections", responses={200: {"model": ModelConnectionList}})
async def list_model_connections(request: Request, principal: Actor = Depends(actor)):
    items = await _services(request).connections.list(owner_id=principal.subject_id)
    return {"connections": [item.public_dict() for item in items]}


# -- ChatGPT Plus/Pro sign-in (connections/chatgpt.py) ---------------------------
# Declared before the `{connection_id}` routes so a literal path segment is
# never read as a connection id.


def _enrollment_error(exc: EnrollmentError):
    if exc.kind in {"not_connected", "expired"}:
        return Conflict(str(exc), enrollment_failure=exc.kind)
    if exc.kind == "invalid_model":
        return InvalidRequest(str(exc), enrollment_failure=exc.kind)
    return CapabilityUnavailable(str(exc), enrollment_failure=exc.kind)


@router.post("/model-connections:chatgpt-start", responses={200: {"model": ChatGPTEnrollment}})
async def start_chatgpt_sign_in(request: Request, principal: Actor = Depends(actor)):
    """Begin a device-code sign-in; returns the OpenAI page and the code to enter."""
    try:
        return await _chatgpt(request).start(principal.subject_id)
    except ChatGPTUnavailable as exc:
        raise CapabilityUnavailable(str(exc), capability="chatgpt_oauth") from exc
    except EnrollmentError as exc:
        raise _enrollment_error(exc) from exc


@router.get("/model-connections:chatgpt", responses={200: {"model": ChatGPTEnrollment}})
async def chatgpt_sign_in_status(request: Request, principal: Actor = Depends(actor)):
    try:
        return await _chatgpt(request).status(principal.subject_id)
    except ChatGPTUnavailable as exc:
        raise CapabilityUnavailable(str(exc), capability="chatgpt_oauth") from exc


@router.post("/model-connections:chatgpt-complete", status_code=201, responses={201: {"model": ModelConnection}})
async def complete_chatgpt_sign_in(
    request: Request, body: CompleteChatGPTRequest, principal: Actor = Depends(actor)
):
    """Save a finished sign-in as a connection for the chosen model."""
    try:
        item = await _chatgpt(request).complete(
            principal.subject_id, model_id=body.model_id, display_name=body.display_name
        )
    except ChatGPTUnavailable as exc:
        raise CapabilityUnavailable(str(exc), capability="chatgpt_oauth") from exc
    except EnrollmentError as exc:
        raise _enrollment_error(exc) from exc
    return item.public_dict()


@router.delete("/model-connections:chatgpt", status_code=204)
async def cancel_chatgpt_sign_in(request: Request, principal: Actor = Depends(actor)):
    await _chatgpt(request).cancel(principal.subject_id)
    return None


@router.get("/model-connections/{connection_id}", responses={200: {"model": ModelConnection}})
async def get_model_connection(
    request: Request, connection_id: str, principal: Actor = Depends(actor)
):
    try:
        item = await _services(request).connections.get(connection_id, owner_id=principal.subject_id)
    except ConnectionNotFound as exc:
        raise NotFound("model connection not found") from exc
    return item.public_dict()


@router.patch("/model-connections/{connection_id}", responses={200: {"model": ModelConnection}})
async def change_connection_model(
    request: Request, connection_id: str, body: UpdateModelConnectionRequest,
    principal: Actor = Depends(actor),
):
    """Switch a connection to another model of the same account; test it again."""
    try:
        item = await _services(request).connections.change_model(
            connection_id, owner_id=principal.subject_id,
            model_id=body.model_id, display_name=body.display_name,
        )
    except ConnectionNotFound as exc:
        raise NotFound("model connection not found") from exc
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc
    return item.public_dict()


@router.get("/model-connections/{connection_id}/models", responses={200: {"model": ModelOptionList}})
async def list_connection_models(
    request: Request, connection_id: str, principal: Actor = Depends(actor)
):
    """The models this connection's own credential can use."""
    try:
        models = await _services(request).connections.models_for(
            connection_id, owner_id=principal.subject_id
        )
    except ConnectionNotFound as exc:
        raise NotFound("model connection not found") from exc
    except ModelListError as exc:
        raise InvalidRequest(str(exc), model_list_failure=exc.kind) from exc
    except ProbeError as exc:
        raise InvalidRequest(str(exc), probe_failure=exc.kind.value) from exc
    return {"models": [model.to_dict() for model in models]}


@router.post("/model-connections/{connection_id}:test", responses={200: {"model": ModelConnection}})
async def test_model_connection(
    request: Request, connection_id: str, principal: Actor = Depends(actor)
):
    try:
        item = await _services(request).connections.test(connection_id, owner_id=principal.subject_id)
    except ConnectionNotFound as exc:
        raise NotFound("model connection not found") from exc
    except ProbeError as exc:
        # Typed and already redacted (connections/probe.py): unreachable,
        # unauthorized, blocked and protocol_error are different problems with
        # different fixes, and the connection has been marked failed.
        raise InvalidRequest(str(exc), probe_failure=exc.kind.value) from exc
    return item.public_dict()


@router.delete("/model-connections/{connection_id}", status_code=204)
async def delete_model_connection(
    request: Request, connection_id: str, principal: Actor = Depends(actor)
):
    try:
        await _services(request).connections.delete(connection_id, owner_id=principal.subject_id)
    except ConnectionNotFound as exc:
        raise NotFound("model connection not found") from exc
    return None
