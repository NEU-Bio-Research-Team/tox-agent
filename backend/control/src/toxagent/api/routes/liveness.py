"""Liveness and readiness."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ...application.capabilities import CapabilityResolver
from ..responses import HealthReady, LiveStatus, ServiceInfo
from ._common import _services, health


@health.get("/", response_model=None, responses={200: {"model": ServiceInfo}})
async def root(request: Request) -> dict[str, Any]:
    """Enough to tell a browser that landed on the bare host this is an API,
    not a dead service. The product UI is a separate deployment; this control
    plane has never served one at ``/`` and Phase 6 does not change that."""
    from ... import __version__

    return {"name": "toxagent-control", "version": __version__, "docs": "/docs"}


@health.get("/health/live", response_model=None, responses={200: {"model": LiveStatus}})
async def live() -> dict[str, str]:
    """Process liveness. Says nothing about the predictor or a runtime."""
    return {"status": "alive"}


@health.get("/health/ready", responses={200: {"model": HealthReady}})
async def ready(request: Request) -> JSONResponse:
    """Can this deployment serve what it was assembled to serve?

    Three separate questions, deliberately not merged into one boolean (I05):

    - **Core readiness** decides the HTTP status. It covers the dependencies
      every mode needs: the database and the predictor. Nothing else can make
      this endpoint return 503.
    - **Capability availability** reports each intent with a reason. A
      predictor-only stack reports `report_qa` unavailable *and stays ready*:
      the runtime was deliberately not installed, and a monitor that pages on
      an absent optional feature is a monitor nobody keeps.
    - **`mode`** says which of those two products this is, derived from what
      is actually wired rather than from `TOXAGENT_RUNTIME_KIND` — which
      Compose set to a value the app never constructs a provider for (I01).

    The database was previously not probed at all, so a control plane whose
    database was unreachable could answer `ready: true` and take traffic.
    """
    services = _services(request)
    resolver: CapabilityResolver = request.app.state.capabilities
    dependencies: dict[str, Any] = {}
    checked_at = datetime.now(timezone.utc).isoformat()
    ok = True

    try:
        await services.database.check()
        dependencies["database"] = {"ready": True, "checked_at": checked_at}
    except Exception as exc:  # noqa: BLE001 — reported as a dependency state
        dependencies["database"] = {
            "ready": False, "reason": type(exc).__name__, "checked_at": checked_at
        }
        ok = False

    try:
        readiness = await services.predictor.ready()
        dependencies["predictor"] = {
            "ready": readiness.ready,
            "served_endpoints": readiness.served_endpoints,
            "checked_at": checked_at,
        }
        ok = ok and readiness.ready
    except Exception as exc:  # noqa: BLE001 — reported as a dependency state
        dependencies["predictor"] = {
            "ready": False, "reason": type(exc).__name__, "checked_at": checked_at
        }
        ok = False

    # `configured` is what this deployment declared; `available` is what a
    # request would actually meet. Keeping both is the point — they disagreed.
    capabilities = {
        name: capability.to_dict() for name, capability in resolver.snapshot().items()
    }

    runtime_info: dict[str, Any] = {
        "configured_kind": services.settings.runtime.kind,
        "bound": services.runtime_gateway is not None,
        "checked_at": checked_at,
    }
    gateway = services.runtime_gateway
    if gateway is not None:
        try:
            runtime_info["healthy"] = await gateway.health()
        except Exception as exc:  # noqa: BLE001 — reported as a dependency state
            runtime_info["healthy"] = False
            runtime_info["reason"] = type(exc).__name__
        # An agent-enabled deployment whose runtime is down is not ready: it
        # advertises intents it cannot currently serve.
        ok = ok and runtime_info["healthy"]

    # Declared-but-unservable is incoherent wiring, not a deployment mode, and
    # is the one capability condition that can fail readiness. A predictor-only
    # stack declares nothing conversational and stays ready.
    broken = resolver.misconfigured()
    if broken:
        runtime_info["healthy"] = False
        runtime_info["misconfigured"] = sorted(c.name for c in broken)
        ok = False

    # Explainability, per model, in the same reason codes an explanation gap
    # carries. Never part of `ok`: a deployment whose admitted models expose no
    # attribution is a deployment without XAI, not a broken one — but "why can
    # this stack not explain herg" should be answerable without building a
    # report to find out (XAI-01).
    from ...application.explanation.readiness import explainer_readiness

    dependencies["explainer"] = await explainer_readiness(services.predictor)
    dependencies["runtime"] = runtime_info
    dependencies["capabilities"] = capabilities
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"ready": ok, "mode": resolver.mode.value, **dependencies},
    )
