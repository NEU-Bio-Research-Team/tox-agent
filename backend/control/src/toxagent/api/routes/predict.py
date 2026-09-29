"""Quick predict, compare, explain and structure recognition, outside any session."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Depends, Request

from ...application.policy import Actor
from ...domain.errors import (
    CapabilityUnavailable,
    EndpointUnavailable,
    SmilesNotDetected,
    StructureRecognitionUnavailable,
)
from ...domain.run import Intent
from ...predictor.contract import ENDPOINTS, TOX21_TASKS
from ...predictor.ocr_client import OcrError, OcrUnavailable
from .._image import decode_declared_image
from ..responses import (
    AtomAttribution,
    EffectiveProduct,
    PredictCapabilities,
    QuickPredictBatchResult,
    QuickPredictCompareResult,
    QuickPredictResult,
)
from ..schemas import (
    ExplainRequest,
    PredictBatchRequest,
    PredictCompareRequest,
    PredictRequest,
    RecognizedStructure,
    RecognizeRequest,
)
from ._common import _services, actor, router


@router.post("/predict", responses={200: {"model": QuickPredictResult}})
async def quick_predict(
    request: Request, body: PredictRequest, principal: Actor = Depends(actor)
):
    """SMILES in, numbers out. No session, no run, no analysis row, no event.

    The response is the same ``AnalysisProjection`` shape the session path
    returns, with ``persisted=false`` and ``analysis_id=null``. A caller that
    needs a durable, provenance-stamped record uses the Lane D analysis flow;
    this returns the provenance in the body and the client keeps it if it wants.
    """
    services = _services(request)
    async with services.predict_limits.slot(principal.subject_id):
        result = await services.quick_predict.execute(
            actor=principal,
            smiles=body.smiles,
            endpoints=tuple(body.endpoints) if body.endpoints else None,
            model_selection=body.model_selection,
            threshold_overrides=body.threshold_overrides,
        )
        if body.include_attribution:
            result["attributions"] = await _quick_attributions(
                services, result["canonical_smiles"], result["served_endpoints"]
            )
    return result


async def _quick_attributions(
    services, canonical_smiles: str, served_endpoints: list[str]
) -> list[dict[str, Any]]:
    """Best-effort token attributions for the convenience flag. Tox21 is
    skipped — it needs a named assay, and a combined tox21 attribution is
    scientifically meaningless."""
    out: list[dict[str, Any]] = []
    for endpoint in served_endpoints:
        if endpoint == "tox21":
            continue
        attribution = await services.predictor.attribution(canonical_smiles, endpoint)
        out.append(attribution.model_dump(mode="json"))
    return out


@router.post("/predict:batch", responses={200: {"model": QuickPredictBatchResult}})
async def quick_predict_batch(
    request: Request, body: PredictBatchRequest, principal: Actor = Depends(actor)
):
    """Order-preserving batch predict. Per-item errors, nothing persisted."""
    services = _services(request)
    services.predict_limits.check_batch_size(len(body.smiles))
    async with services.predict_limits.slot(principal.subject_id):
        return await services.quick_predict.execute_batch(
            actor=principal,
            smiles=body.smiles,
            endpoints=tuple(body.endpoints) if body.endpoints else None,
            model_selection=body.model_selection,
            threshold_overrides=body.threshold_overrides,
        )


async def _refuse_unadmitted(services, selections) -> None:
    """Refuse a comparison naming a model this build will not run.

    Checked against the predictor's own catalogue rather than a list here, so
    a model blocked for a reason the control plane does not model — a
    tokenizer that cannot be proved, in ClinTox's case — is still refused, and
    refused with the predictor's stated reason instead of a guess.
    """
    catalogue = await services.predictor.models()
    admitted = {
        model.model_id: model for model in catalogue.models if model.loaded
    }
    known = {model.model_id: model for model in catalogue.models}
    problems = []
    for endpoint, model_id in selections:
        model = admitted.get(model_id)
        if model is None:
            declared = known.get(model_id)
            reason = (
                (declared.blocked_reason or declared.detail or "not admitted on this build")
                if declared is not None
                else "no such model in this deployment's catalogue"
            )
            problems.append({"endpoint": endpoint, "model_id": model_id, "reason": reason})
        elif endpoint not in model.capabilities:
            # A model that is admitted, but not for this endpoint. Running it
            # would answer a different question and label it with this one.
            problems.append({
                "endpoint": endpoint, "model_id": model_id,
                "reason": f"does not serve {endpoint}; serves {sorted(model.capabilities)}",
            })
    if problems:
        raise EndpointUnavailable(
            "comparison names "
            + ("a model" if len(problems) == 1 else f"{len(problems)} models")
            + " this deployment will not run",
            refused=problems,
            admitted=sorted(admitted),
        )


@router.post("/predict:compare", responses={200: {"model": QuickPredictCompareResult}})
async def quick_predict_compare(
    request: Request, body: PredictCompareRequest, principal: Actor = Depends(actor)
):
    """Run explicitly selected admitted models side-by-side.

    Each item is independently projected through ``QuickPredict``.  There is
    no aggregate verdict and no cross-model averaging: models can differ in
    calibration and their numbers must remain attributable to that model.

    Admission is checked here, before any of them runs. It used to be checked
    only by the predictor, per request, inside an ``asyncio.gather`` with no
    ``return_exceptions`` — so naming one blocked model (ClinTox on this
    build) discarded every other column and returned one error about the whole
    comparison. The user asked which models can answer, and the honest reply
    names the one that cannot and why, rather than failing all of them.
    """
    services = _services(request)
    selections = [
        (endpoint, model_id)
        for endpoint, model_ids in body.model_selection.items()
        for model_id in dict.fromkeys(model_ids)
    ]
    await _refuse_unadmitted(services, selections)
    async with services.predict_limits.slot(principal.subject_id):
        results = await asyncio.gather(
            *(
                services.quick_predict.execute(
                    actor=principal,
                    smiles=body.smiles,
                    endpoints=(endpoint,),
                    model_selection={endpoint: model_id},
                    threshold_overrides=body.threshold_overrides,
                )
                for endpoint, model_id in selections
            )
        )
    return {
        "persisted": False,
        "input_smiles": body.smiles,
        "comparisons": [
            {"endpoint": endpoint, "model_id": model_id, "result": result}
            for (endpoint, model_id), result in zip(selections, results)
        ],
    }


@router.get("/system/effective-product", responses={200: {"model": EffectiveProduct}})
async def effective_product(request: Request, principal: Actor = Depends(actor)):
    """The effective product configuration this deployment runs: flags, intent
    to lane/profile/tools, runtime binding, budgets, topology and hashes.

    Read by the eval runner so a live benchmark manifest records what was
    actually graded. Secret-free by construction (effective_product.py)."""
    from ..effective_product import describe_effective_product

    services = _services(request)
    resolver = getattr(services, "capabilities", None)
    capabilities = None
    if resolver is not None:
        capabilities = {
            "mode": resolver.mode.value,
            "intents": {
                intent.value: resolver.intent(intent).to_dict()["available"]
                for intent in Intent
            },
        }
    return describe_effective_product(
        services.settings,
        tool_registry=getattr(services, "tool_registry", None),
        capabilities=capabilities,
    )


@router.get("/predict/capabilities", responses={200: {"model": PredictCapabilities}})
async def predict_capabilities(request: Request, principal: Actor = Depends(actor)):
    """A straight proxy of what the predictor actually serves, so the UI can
    render an unserved endpoint (ClinTox on this build) as disabled with a real
    reason rather than guessing."""
    services = _services(request)
    models = await services.predictor.models()
    model_rows = [model.model_dump(mode="json") for model in models.models]
    served = set(models.served_endpoints)
    display_names = {
        "herg": "hERG blockade", "tox21": "Tox21 assays", "clintox": "Clinical toxicity",
    }
    endpoints = []
    for endpoint in ENDPOINTS:
        matching = [model for model in model_rows if endpoint in model.get("capabilities", [])]
        model = next((candidate for candidate in matching if candidate.get("loaded")), matching[0] if matching else None)
        enabled = endpoint in served
        endpoints.append({
            "id": endpoint,
            "display_name": display_names[endpoint],
            "enabled": enabled,
            "model_id": model.get("model_id") if model else None,
            "supports_explanation": enabled and endpoint in {"herg", "tox21"},
            "explanation_target_required": endpoint == "tox21",
            "tasks": list(TOX21_TASKS) if endpoint == "tox21" else [],
            "blocked_reason": None if enabled else (model.get("blocked_reason") if model else "No reproducible model artifact is available for this endpoint."),
            "models": [candidate for candidate in matching if candidate.get("loaded")],
        })
    return {
        "capability_version": "predict-capabilities-v2",
        "default_endpoints": [endpoint for endpoint in services.settings.policy.default_endpoints if endpoint in served],
        "served_endpoints": list(models.served_endpoints),
        "endpoints": endpoints,
        "models": model_rows,
        "predictor_id": services.predictor.base_url_id,
        "ocr_available": services.ocr is not None,
    }


@router.post("/predict/recognize", response_model=RecognizedStructure)
async def quick_recognize(
    request: Request, body: RecognizeRequest, principal: Actor = Depends(actor)
):
    """Image in, SMILES out. Stateless: the bytes are decoded, checked, passed
    to the OCR service, and discarded — no object store, no run, no analysis.

    Two-step by design (D-IMG-3): this returns the recognised SMILES and its
    confidence into an editable field; the user confirms before ``/v1/predict``.
    """
    services = _services(request)
    if services.ocr is None:
        raise CapabilityUnavailable("no structure recognition service is configured")
    image_bytes = decode_declared_image(
        body.mime_type,
        body.data_base64,
        max_bytes=services.settings.policy.max_image_bytes,
    )
    async with services.predict_limits.slot(principal.subject_id):
        try:
            result = await services.ocr.recognize(image_bytes, body.mime_type)
        except OcrError as exc:
            raise SmilesNotDetected(str(exc)) from exc
        except OcrUnavailable as exc:
            raise StructureRecognitionUnavailable(str(exc)) from exc
    return RecognizedStructure(
        smiles=result.smiles,
        canonical_smiles=result.canonical_smiles,
        confidence=result.confidence,
    )


@router.post("/predict/explain", responses={200: {"model": AtomAttribution}})
async def quick_explain(
    request: Request, body: ExplainRequest, principal: Actor = Depends(actor)
):
    """Atom-level XAI for one served endpoint (one Tox21 assay at a time).

    A thin proxy of ToxPred ``POST /v1/explanations``. Stateless, no persistence.
    The ``attribution_not_causality`` limitation is always echoed so the UI
    cannot render the highlight without it (mirrors the grounded-answer path).
    """
    services = _services(request)
    async with services.predict_limits.slot(principal.subject_id):
        explanation = await services.predictor.explain(
            body.smiles, body.endpoint, body.task
        )
    payload = explanation.model_dump(mode="json")
    limitations = list(payload.get("limitations") or [])
    if "attribution_not_causality" not in limitations:
        limitations.append("attribution_not_causality")
    payload["limitations"] = limitations
    return payload
