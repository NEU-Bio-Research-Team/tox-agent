"""Typed errors and their HTTP mapping.

Every failure is a named error code, never a prediction-shaped placeholder. The
code this replaces answered an unparseable molecule with
``{"label": "PARSE_ERROR", "p_toxic": 0.0}`` — a payload a caller can read as
"predicted non-toxic".
"""
from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ..domain.molecule import InvalidSmilesError
from ..scientific.artifacts import ArtifactError


def error_body(code: str, message: str, **detail: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"error": code, "message": message}
    if detail:
        body["detail"] = detail
    return body


async def invalid_smiles_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, InvalidSmilesError)
    return JSONResponse(
        status_code=400,
        content=error_body("invalid_smiles", str(exc), smiles=exc.smiles, reason=exc.reason),
    )


async def artifact_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """A model that is missing, corrupt or unloaded is 503, never a fallback."""
    return JSONResponse(status_code=503, content=error_body("model_not_ready", str(exc)))


async def value_error_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content=error_body("invalid_request", str(exc)))
