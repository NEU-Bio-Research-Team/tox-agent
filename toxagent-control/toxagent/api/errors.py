"""The error envelope (plan section 6.6).

One shape for every failure, and a code that is either in the documented set or
becomes ``internal_error`` with no message. Leaking an unexpected exception's
text is how a database URL or a provider key ends up in a client's logs.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..domain.errors import PUBLIC_ERROR_CODES, ToxAgentError

log = logging.getLogger("toxagent.api")


def envelope(code: str, message: str, *, retryable: bool = False, **details: Any) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": details,
        }
    }


async def toxagent_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ToxAgentError)
    code = exc.code if exc.code in PUBLIC_ERROR_CODES else "internal_error"
    body = exc.envelope(run_id=getattr(request.state, "run_id", None))
    body["error"]["code"] = code
    return JSONResponse(status_code=exc.http_status, content=body)


async def validation_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return JSONResponse(
        status_code=400,
        content=envelope(
            "invalid_request",
            "the request body does not match the API contract",
            fields=[
                {"loc": [str(p) for p in e["loc"]], "msg": e["msg"]} for e in exc.errors()[:10]
            ],
        ),
    )


async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Log the detail, return none of it."""
    log.exception("unhandled error in the control plane", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=envelope("internal_error", "the control plane failed to handle this request"),
    )


def install(app: FastAPI) -> None:
    app.add_exception_handler(ToxAgentError, toxagent_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
