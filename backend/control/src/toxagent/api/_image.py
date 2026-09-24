"""Byte-boundary image checks, shared by the session upload route and the
stateless ``POST /v1/predict/recognize`` proxy.

Deliberately not an image decoder — ``toxocr`` owns parsing an image as its
separate boundary (ADR 0001). This only stops the product from retaining and
forwarding arbitrary bytes merely labelled ``image/png``, and keeps the accepted
media types aligned in one place.
"""
from __future__ import annotations

import base64
import binascii

from ..domain.errors import InvalidRequest


def matches_declared_image_type(mime_type: str, data: bytes) -> bool:
    """Cheap, dependency-free signature check."""
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def decode_declared_image(mime_type: str, data_base64: str, *, max_bytes: int) -> bytes:
    """Decode a base64 upload and check it at the transport boundary.

    A malformed ``data_base64``, an oversize payload, or a MIME/signature
    mismatch is a client mistake (``invalid_request``), never a 500 and never a
    silently-forwarded blob.
    """
    try:
        decoded = base64.b64decode(data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequest("image data_base64 is not valid base64") from exc
    if not decoded:
        raise InvalidRequest("image payload is empty")
    if len(decoded) > max_bytes:
        raise InvalidRequest(
            f"image is {len(decoded)} bytes, over the {max_bytes}-byte limit",
            max_bytes=max_bytes,
            size_bytes=len(decoded),
        )
    if not matches_declared_image_type(mime_type, decoded):
        raise InvalidRequest("image bytes do not match the declared mime_type")
    return decoded
