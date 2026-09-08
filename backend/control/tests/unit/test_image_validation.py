"""Upload validation at the API byte boundary (remaining-plan W4-08)."""
from __future__ import annotations

import base64

import pytest

from toxagent.api.routes import _decode_image
from toxagent.api.schemas import ImageInput
from toxagent.domain.errors import InvalidRequest


@pytest.mark.parametrize(
    ("mime_type", "data"),
    [
        ("image/png", b"\x89PNG\r\n\x1a\npayload"),
        ("image/jpeg", b"\xff\xd8\xff\xe0payload"),
        ("image/webp", b"RIFF\x00\x00\x00\x00WEBPpayload"),
    ],
)
def test_decode_image_accepts_each_supported_magic_signature(mime_type: str, data: bytes):
    mime, size, decoded = _decode_image(
        ImageInput(mime_type=mime_type, data_base64=base64.b64encode(data).decode())
    )

    assert (mime, size, decoded) == (mime_type, len(data), data)


@pytest.mark.parametrize(
    ("declared", "actual"),
    [
        ("image/png", b"not a raster"),
        ("image/png", b"\xff\xd8\xff\xe0jpeg"),
        ("image/jpeg", b"\x89PNG\r\n\x1a\npng"),
        ("image/webp", b"RIFF\x00\x00\x00\x00NOPEpayload"),
    ],
)
def test_decode_image_rejects_a_declared_type_that_does_not_match_bytes(
    declared: str, actual: bytes
):
    image = ImageInput(mime_type=declared, data_base64=base64.b64encode(actual).decode())

    with pytest.raises(InvalidRequest, match="do not match"):
        _decode_image(image)
