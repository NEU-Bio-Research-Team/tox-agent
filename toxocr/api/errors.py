"""One error envelope shape, matching toxagent-control's own convention
closely enough that OcrClient (toxagent-control/toxagent/predictor/ocr_client.py)
can read it without a second parser."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from ..scientific.molscribe_predictor import ImageDecodeError, StructureNotDetected


def _envelope(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


async def image_decode_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ImageDecodeError)
    return JSONResponse(status_code=415, content=_envelope("unsupported_image_format", str(exc)))


async def structure_not_detected_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StructureNotDetected)
    return JSONResponse(status_code=422, content=_envelope("smiles_not_detected", str(exc)))
