"""v1 routes.

Handlers stay thin: parse, delegate, serialise. There is no model-specific
branching here — which model answers which endpoint is the registry's decision,
not the HTTP layer's.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..application.attribution import AttributionService
from ..application.explain import ExplainService
from ..application.predictor import ToxicityPredictor
from ..scientific.artifacts import ArtifactError
from ..scientific.registry import ModelRegistry
from .schemas import (
    AttributionRequest,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplainRequest,
    ModelInfo,
    ModelsResponse,
    PredictionRequest,
    ReadinessResponse,
)

health_router = APIRouter(tags=["health"])
v1_router = APIRouter(prefix="/v1", tags=["prediction"])


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def _predictor(request: Request) -> ToxicityPredictor:
    return request.app.state.predictor


def _overrides(body: PredictionRequest | BatchPredictionRequest) -> dict[str, Any]:
    o = body.threshold_overrides
    if o is None:
        return {}
    return {
        "herg_threshold_override": o.herg,
        "clintox_threshold_override": o.clintox,
        "tox21_threshold_overrides": o.tox21,
    }


# --- health ----------------------------------------------------------------

@health_router.get("/health/live")
def live() -> dict[str, str]:
    """Process liveness only. Deliberately says nothing about the models."""
    return {"status": "alive"}


@health_router.get("/health/ready", response_model=ReadinessResponse)
def ready(request: Request):
    registry = _registry(request)
    served = sorted(registry.describe_capabilities())
    is_ready, reasons = registry.is_ready()
    return ReadinessResponse(ready=is_ready, reasons=reasons, served_endpoints=served, device=request.app.state.settings.device)


# --- inventory -------------------------------------------------------------

@v1_router.get("/models", response_model=ModelsResponse)
def models(request: Request):
    registry = _registry(request)
    infos: list[ModelInfo] = []
    for health in registry.health():
        try:
            spec = registry.spec(health.model_id)
            required, blocked = spec.required, (spec.blocked_reason or None)
            capabilities = sorted(spec.capabilities)
        except ArtifactError:
            required, blocked, capabilities = True, None, sorted(health.capabilities)
        infos.append(
            ModelInfo(
                model_id=health.model_id,
                capabilities=capabilities,
                loaded=health.loaded,
                required=required,
                detail=health.detail,
                blocked_reason=blocked,
            )
        )
    infos.sort(key=lambda m: m.model_id)
    return ModelsResponse(models=infos, served_endpoints=sorted(registry.describe_capabilities()))


# --- prediction ------------------------------------------------------------

@v1_router.post("/predictions")
def predict(request: Request, body: PredictionRequest) -> dict[str, Any]:
    result = _predictor(request).predict(
        body.smiles, body.endpoints, **_overrides(body)
    )
    return result.to_dict()


@v1_router.post("/predictions:batch", response_model=BatchPredictionResponse)
def predict_batch(request: Request, body: BatchPredictionRequest):
    results, errors = _predictor(request).predict_batch(
        body.smiles, body.endpoints, **_overrides(body)
    )
    return BatchPredictionResponse(
        results=[r.to_dict() for r in results],
        errors=[
            {
                "index": e.index,
                "input_smiles": e.input_smiles,
                "error": e.error,
                "detail": e.detail,
            }
            for e in errors
        ],
        count=len(body.smiles),
    )


# --- attribution -----------------------------------------------------------

@v1_router.post("/attributions")
def attributions(request: Request, body: AttributionRequest) -> dict[str, Any]:
    service: AttributionService = request.app.state.attribution
    return service.attribute(body.smiles, body.endpoint, body.task)


# --- explanation ---------------------------------------------------------------

@v1_router.post("/explanations")
def explanations(request: Request, body: ExplainRequest) -> dict[str, Any]:
    """Token attribution projected onto heavy-atom indices. ``/v1/attributions``
    stays as the token-only endpoint for backward compatibility."""
    service: ExplainService = request.app.state.explain
    return service.explain(body.smiles, body.endpoint, body.task)
