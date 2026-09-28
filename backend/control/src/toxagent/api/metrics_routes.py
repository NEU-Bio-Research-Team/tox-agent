"""``GET /metrics``: the process registry in Prometheus text format (PR-18).

Unauthenticated unless ``TOXAGENT_METRICS_TOKEN`` is set, because the usual
deployment scrapes it from inside a private network and every value in it has
already passed the cardinality guard in ``toxagent.platform.metrics`` — there is no id,
molecule or prose in it to protect. A deployment whose control plane is reachable
from outside sets the token, and the scraper sends it as a bearer credential.
"""
from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from ..platform import metrics

router = APIRouter(tags=["health"])


@router.get("/metrics", response_class=PlainTextResponse)
async def get_metrics(request: Request) -> PlainTextResponse:
    expected = (os.getenv("TOXAGENT_METRICS_TOKEN") or "").strip()
    if expected:
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {expected}"):
            return PlainTextResponse("unauthorized\n", status_code=401)
    return PlainTextResponse(
        metrics.REGISTRY.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )
