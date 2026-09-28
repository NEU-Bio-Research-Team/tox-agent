"""The two routers, the auth dependency and helpers every resource module shares."""
from __future__ import annotations

import base64
import binascii
from typing import Any

from fastapi import APIRouter, Request

from ...application.policy import Actor
from ...domain.errors import (
    InvalidRequest,
)
from .._image import matches_declared_image_type

router = APIRouter(prefix="/v1", tags=["toxagent"])


health = APIRouter(tags=["health"])


async def actor(request: Request) -> Actor:
    return await request.app.state.auth.authenticate(request)


def _services(request: Request):
    return request.app.state


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _decode_image(image) -> tuple[str | None, int, bytes | None]:
    """Decode the upload here, at the transport boundary. A malformed
    ``data_base64`` or a MIME/signature mismatch is a client mistake, not a
    500. The decoded bytes then pass once to ``MessageSubmission`` so it can
    persist them before accepting an OCR run (W4-07/08)."""
    if image is None:
        return None, 0, None
    try:
        decoded = base64.b64decode(image.data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequest("image.data_base64 is not valid base64") from exc
    if not matches_declared_image_type(image.mime_type, decoded):
        raise InvalidRequest("image bytes do not match the declared mime_type")
    return image.mime_type, len(decoded), decoded
