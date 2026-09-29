"""Hold every JSON response the suite produces to the model its route declares.

Routes document their response shape with ``responses={200: {"model": X}}``
(``toxagent/api/responses.py``) and return plain dicts, so production never
turns a shape mismatch into a 500. The mismatch is caught here instead: this
wraps every route handler, validates each JSON body whose status code has a
declared model, and records a violation that ``tests/conftest.py`` turns into a
test failure. The frontend's types are generated from those models, so a route
that drifts from its model is a route the browser misreads.
"""
from __future__ import annotations

import json

import fastapi.routing
from pydantic import BaseModel, ValidationError

#: (route, status, error summary) for every response that did not match.
VIOLATIONS: list[tuple[str, int, str]] = []

_original_get_route_handler = fastapi.routing.APIRoute.get_route_handler


def _declared_model(route: fastapi.routing.APIRoute, status: int) -> type[BaseModel] | None:
    spec = route.responses.get(status) or route.responses.get(str(status)) or {}
    model = spec.get("model")
    return model if isinstance(model, type) and issubclass(model, BaseModel) else None


def _checked_route_handler(route: fastapi.routing.APIRoute):
    handler = _original_get_route_handler(route)

    async def checked(request):
        response = await handler(request)
        model = _declared_model(route, response.status_code)
        body = getattr(response, "body", None)
        if model is not None and body and "json" in (response.media_type or ""):
            try:
                model.model_validate(json.loads(body))
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5]
                )
                VIOLATIONS.append((f"{sorted(route.methods)[0]} {route.path}", response.status_code, problems))
        return response

    return checked


def install() -> None:
    fastapi.routing.APIRoute.get_route_handler = _checked_route_handler
