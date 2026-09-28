"""Model-provider connections: create, list, probe, delete."""
from __future__ import annotations

from typing import Any

from fastapi import Depends, Request

from ...application.policy import Actor
from ...connections import providers
from ...connections.probe import ProbeError
from ...connections.service import ConnectionNotFound
from ...domain.errors import (
    InvalidRequest,
    NotFound,
)
from ...domain.runtime import AuthMode
from ..schemas import (
    CreateModelConnectionRequest,
)
from ._common import _services, actor, router


@router.get("/model-connections:providers")
async def list_supported_providers() -> dict[str, Any]:
    """Providers this control plane has an adapter for, and their defaults.

    The UI used to carry its own list, which included providers whose wire
    format nothing here speaks and whose base URL default was blank — so the
    form could be filled in correctly and still fail every probe (I13). One
    list, served by the code that decides.
    """
    return {"providers": providers.catalogue()}


@router.post("/model-connections", status_code=201)
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


@router.get("/model-connections")
async def list_model_connections(request: Request, principal: Actor = Depends(actor)):
    items = await _services(request).connections.list(owner_id=principal.subject_id)
    return {"connections": [item.public_dict() for item in items]}


@router.get("/model-connections/{connection_id}")
async def get_model_connection(
    request: Request, connection_id: str, principal: Actor = Depends(actor)
):
    try:
        item = await _services(request).connections.get(connection_id, owner_id=principal.subject_id)
    except ConnectionNotFound as exc:
        raise NotFound("model connection not found") from exc
    return item.public_dict()


@router.post("/model-connections/{connection_id}:test")
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
