from __future__ import annotations

from typing import Any, Awaitable, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException


_MODEL_SERVER_TITLE = "ToxAgent Model Server"
_MODEL_SERVER_DESCRIPTION = "SMILESGNN toxicity prediction API for ToxAgent agentic system"
_MODEL_SERVER_VERSION = "0.0.6"

HTTPExceptionHandler = Callable[..., Awaitable[Any]]


def create_model_server_app(
    *,
    lifespan: Any,
    http_exception_handler: HTTPExceptionHandler,
) -> FastAPI:
    app = FastAPI(
        title=_MODEL_SERVER_TITLE,
        description=_MODEL_SERVER_DESCRIPTION,
        version=_MODEL_SERVER_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    return app