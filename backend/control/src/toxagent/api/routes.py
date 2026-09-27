"""Product HTTP routes (plan sections 6.1-6.5).

Handlers stay thin: authenticate, parse, delegate, serialise. Ownership is
enforced in the application and the store, not here, so a new endpoint cannot
forget it. Reads are all reconstructions of committed state, which is what makes
"the stream died" a non-event.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

from ..application.capabilities import CapabilityResolver
from ..application.policy import Actor
from ..application.sessions import run_projection
from ..application.submit_message import MessageSubmission
from ..domain.errors import (
    AnalysisNotFound,
    CapabilityUnavailable,
    EndpointUnavailable,
    InvalidRequest,
    NotFound,
    SmilesNotDetected,
    StructureRecognitionUnavailable,
)
from ..domain.evidence import EvidenceStatus
from ..domain.events import EventType
from ..domain.observation import ObservationKind
from ..domain.run import Intent
from ..domain.runtime import AuthMode
from ..connections import providers
from ..connections.probe import ProbeError
from ..connections.service import ConnectionNotFound
from ..predictor.ocr_client import OcrError, OcrUnavailable
from ..predictor.contract import ENDPOINTS, TOX21_TASKS
from ..streaming.sse import event_stream
from ._image import decode_declared_image, matches_declared_image_type
from .schemas import (
    AcceptedResponse,
    CancelResponse,
    CaseContextRequest,
    CaseQuestionRequest,
    CaseScopeRequest,
    SkillDraftRequest,
    SkillDraftReviewRequest,
    CreateSessionRequest,
    CreateReportRequest,
    ExplainRequest,
    PredictBatchRequest,
    PredictCompareRequest,
    PredictRequest,
    RecognizedStructure,
    RecognizeRequest,
    SendMessageRequest,
    SessionResponse,
    CreateModelConnectionRequest,
    SessionSettingsRequest,
    UpdateSessionRequest,
)
from ..persistence.object_store import ObjectRef, ObjectNotFound

router = APIRouter(prefix="/v1", tags=["toxagent"])
health = APIRouter(tags=["health"])


async def actor(request: Request) -> Actor:
    return await request.app.state.auth.authenticate(request)


def _services(request: Request):
    return request.app.state


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat()


# --- health ----------------------------------------------------------------

@health.get("/")
async def root(request: Request) -> dict[str, Any]:
    """Enough to tell a browser that landed on the bare host this is an API,
    not a dead service. The product UI is a separate deployment; this control
    plane has never served one at ``/`` and Phase 6 does not change that."""
    from .. import __version__

    return {"name": "toxagent-control", "version": __version__, "docs": "/docs"}


@health.get("/health/live")
async def live() -> dict[str, str]:
    """Process liveness. Says nothing about the predictor or a runtime."""
    return {"status": "alive"}


@health.get("/health/ready")
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
    from ..application.explanation_readiness import explainer_readiness

    dependencies["explainer"] = await explainer_readiness(services.predictor)
    dependencies["runtime"] = runtime_info
    dependencies["capabilities"] = capabilities
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"ready": ok, "mode": resolver.mode.value, **dependencies},
    )


# --- quick predict (stateless, no session) --------------------------------

@router.post("/predict")
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


@router.post("/predict:batch")
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


@router.post("/predict:compare")
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


@router.get("/system/effective-product")
async def effective_product(request: Request, principal: Actor = Depends(actor)):
    """The effective product configuration this deployment runs: flags, intent
    to lane/profile/tools, runtime binding, budgets, topology and hashes.

    Read by the eval runner so a live benchmark manifest records what was
    actually graded. Secret-free by construction (effective_product.py)."""
    from ..application.effective_product import describe_effective_product

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


@router.get("/predict/capabilities")
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


@router.post("/predict/explain")
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


# --- sessions --------------------------------------------------------------

@router.get("/model-connections:providers")
async def list_supported_providers() -> dict[str, Any]:
    """Providers this control plane has an adapter for, and their defaults.

    The UI used to carry its own list, which included providers whose wire
    format nothing here speaks and whose base URL default was blank — so the
    form could be filled in correctly and still fail every probe (I13). One
    list, served by the code that decides.
    """
    return {"providers": providers.catalogue()}


@router.post("/model-connections", status_code=201)
async def create_model_connection(
    request: Request, body: CreateModelConnectionRequest, principal: Actor = Depends(actor)
):
    try:
        item = await _services(request).connections.create(
            owner_id=principal.subject_id, provider_id=body.provider_id, model_id=body.model_id,
            auth_mode=AuthMode(body.auth_mode), base_url=body.base_url, credential=body.credential,
            display_name=body.display_name,
        )
    except providers.UnsupportedProvider as exc:
        # The reason is the useful part: it says what to pick instead.
        raise InvalidRequest(exc.reason, provider_id=exc.provider_id) from exc
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc
    return item.public_dict()


@router.get("/model-connections")
async def list_model_connections(request: Request, principal: Actor = Depends(actor)):
    items = await _services(request).connections.list(owner_id=principal.subject_id)
    return {"connections": [item.public_dict() for item in items]}


@router.get("/model-connections/{connection_id}")
async def get_model_connection(
    request: Request, connection_id: str, principal: Actor = Depends(actor)
):
    try:
        item = await _services(request).connections.get(connection_id, owner_id=principal.subject_id)
    except ConnectionNotFound as exc:
        raise NotFound("model connection not found") from exc
    return item.public_dict()


@router.post("/model-connections/{connection_id}:test")
async def test_model_connection(
    request: Request, connection_id: str, principal: Actor = Depends(actor)
):
    try:
        item = await _services(request).connections.test(connection_id, owner_id=principal.subject_id)
    except ConnectionNotFound as exc:
        raise NotFound("model connection not found") from exc
    except ProbeError as exc:
        # Typed and already redacted (connections/probe.py): unreachable,
        # unauthorized, blocked and protocol_error are different problems with
        # different fixes, and the connection has been marked failed.
        raise InvalidRequest(str(exc), probe_failure=exc.kind.value) from exc
    return item.public_dict()


@router.delete("/model-connections/{connection_id}", status_code=204)
async def delete_model_connection(
    request: Request, connection_id: str, principal: Actor = Depends(actor)
):
    try:
        await _services(request).connections.delete(connection_id, owner_id=principal.subject_id)
    except ConnectionNotFound as exc:
        raise NotFound("model connection not found") from exc
    return None

@router.get("/sessions")
async def list_sessions(
    request: Request,
    limit: int = Query(25, ge=1, le=50),
    offset: int = Query(0, ge=0),
    principal: Actor = Depends(actor),
):
    return await _services(request).sessions.list(principal, limit=limit, offset=offset)


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    request: Request, body: CreateSessionRequest, principal: Actor = Depends(actor)
):
    session = await _services(request).sessions.create(
        principal,
        preferred_language=body.preferred_language,
        title=body.title,
        client_session_id=body.client_session_id,
    )
    return SessionResponse(
        session_id=session.id,
        status=session.status.value,
        preferred_language=session.preferred_language.value,
        title=session.title,
        created_at=session.created_at.isoformat(),
        version=session.version,
        title_source=session.title_source.value if session.title_source else None,
        title_status=session.title_status,
    )


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(
    request: Request, session_id: str, body: UpdateSessionRequest, principal: Actor = Depends(actor)
):
    session = await _services(request).sessions.rename(
        principal, session_id, title=body.title, expected_version=body.expected_version
    )
    return SessionResponse(
        session_id=session.id, status=session.status.value,
        preferred_language=session.preferred_language.value, title=session.title,
        created_at=session.created_at.isoformat(), version=session.version,
        title_source=session.title_source.value if session.title_source else None,
        title_status=session.title_status,
    )


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str, principal: Actor = Depends(actor)):
    return await _services(request).sessions.projection(principal, session_id)


@router.get("/sessions/{session_id}/settings")
async def get_session_settings(request: Request, session_id: str, principal: Actor = Depends(actor)):
    return await _services(request).sessions.settings(principal, session_id)


@router.patch("/sessions/{session_id}/settings")
async def update_session_settings(request: Request, session_id: str, body: SessionSettingsRequest,
                                  principal: Actor = Depends(actor)):
    return await _services(request).sessions.update_settings(
        principal, session_id, ai_profile_id=body.ai_profile_id,
        predictor_bindings=dict(body.predictor_bindings),
    )


@router.get("/sessions/{session_id}/messages")
async def list_messages(
    request: Request,
    session_id: str,
    after_sequence: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    principal: Actor = Depends(actor),
):
    messages = await _services(request).sessions.messages(
        principal, session_id, after_sequence=after_sequence, limit=limit
    )
    return {"messages": messages, "count": len(messages)}


def _decode_image(image) -> tuple[str | None, int, bytes | None]:
    """Decode the upload here, at the transport boundary. A malformed
    ``data_base64`` or a MIME/signature mismatch is a client mistake, not a
    500. The decoded bytes then pass once to ``MessageSubmission`` so it can
    persist them before accepting an OCR run (W4-07/08)."""
    if image is None:
        return None, 0, None
    try:
        decoded = base64.b64decode(image.data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequest("image.data_base64 is not valid base64") from exc
    if not matches_declared_image_type(image.mime_type, decoded):
        raise InvalidRequest("image bytes do not match the declared mime_type")
    return image.mime_type, len(decoded), decoded


@router.post("/sessions/{session_id}/messages", response_model=AcceptedResponse, status_code=202)
async def send_message(
    request: Request, session_id: str, body: SendMessageRequest, principal: Actor = Depends(actor)
):
    options = body.analysis_options
    molecule = body.molecule
    image_mime_type, image_size_bytes, image_bytes = _decode_image(body.image)
    accepted = await _services(request).submit_message.execute(
        actor=principal,
        session_id=session_id,
        submission=MessageSubmission(
            text=body.text,
            client_message_id=body.client_message_id,
            intent_hint=body.intent_hint,
            smiles=molecule.smiles if molecule else None,
            batch_smiles=tuple(molecule.batch_smiles or ()) if molecule else (),
            endpoints=tuple(options.endpoints) if options and options.endpoints else None,
            model_selection=options.model_selection if options else None,
            threshold_overrides=options.threshold_overrides if options else None,
            include_attribution=options.include_attribution if options else False,
            explanation_mode=options.explanation_mode if options else "on_demand",
            explanation_targets=tuple((target.endpoint, target.task) for target in (options.explanation_targets if options else [])),
            analysis_id=body.analysis_id,
            image_mime_type=image_mime_type,
            image_size_bytes=image_size_bytes,
            image_bytes=image_bytes,
        ),
    )
    return AcceptedResponse(**accepted.to_dict())


# --- report builds and immutable artifacts --------------------------------

@router.post("/sessions/{session_id}/reports", response_model=AcceptedResponse, status_code=202)
async def create_report(
    request: Request, session_id: str, body: CreateReportRequest,
    principal: Actor = Depends(actor),
):
    """Admit a report build through the same durable run path as chat."""
    molecule = body.molecule
    targets = tuple(("tox21", task) for task in body.selected_tox21_tasks)
    accepted = await _services(request).submit_message.execute(
        actor=principal,
        session_id=session_id,
        submission=MessageSubmission(
            text="Build a complete toxicity screening report.",
            intent_hint="build_report",
            smiles=molecule.smiles if molecule else None,
            endpoints=tuple(body.selected_endpoints) if body.selected_endpoints else None,
            explanation_mode="required" if body.include_explanations else "none",
            explanation_targets=targets,
            analysis_id=body.analysis_id,
            report_language=body.report_language,
            report_audience=body.audience,
            include_external_evidence=body.include_external_evidence,
            report_output_formats=tuple(body.output_formats),
        ),
    )
    return AcceptedResponse(**accepted.to_dict())


@router.get("/sessions/{session_id}/reports")
async def list_reports(
    request: Request, session_id: str, limit: int = Query(50, ge=1, le=200),
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        reports = await uow.reports.list_artifacts_for_session(session_id, limit=limit)
    return {"reports": reports, "count": len(reports)}


@router.get("/sessions/{session_id}/reports/{report_id}")
async def get_report(
    request: Request, session_id: str, report_id: str,
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        report = await uow.reports.get_artifact(report_id, session_id=session_id)
    if report is None:
        raise NotFound("no such report", report_id=report_id)
    return report


@router.get("/sessions/{session_id}/reports/{report_id}/renderings/{format}")
async def download_report_rendering(
    request: Request, session_id: str, report_id: str, format: str,
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        rendering = await uow.reports.get_rendering(
            report_id, format, session_id=session_id
        )
    if rendering is None:
        raise NotFound("no such report rendering", report_id=report_id, format=format)
    try:
        data = await services.object_store.get(ObjectRef(rendering["object_uri"]))
    except ObjectNotFound as exc:
        raise NotFound("the report rendering bytes are unavailable") from exc
    suffix = {"markdown": "md", "html": "html", "pdf": "pdf"}.get(format, format)
    return Response(
        data, media_type=rendering["media_type"],
        headers={"Content-Disposition": f'attachment; filename="{report_id}.{suffix}"'},
    )


@router.get("/sessions/{session_id}/reports/{report_id}/figures/{figure_id}")
async def get_report_figure(
    request: Request, session_id: str, report_id: str, figure_id: str,
    principal: Actor = Depends(actor),
):
    """One figure's bytes, scoped to the report that shows it (REP-02).

    Three checks, none of them optional. The session must belong to the caller;
    the figure must be one this report actually carries — otherwise a valid
    figure id from another report in the same session would serve an image the
    caller was never shown; and the bytes must hash to what the figure claims,
    because a report's integrity guarantee covers its pictures and an image is
    the one part of a document nobody proofreads.

    Cached immutably: the artifact is immutable and the object key is the content
    hash, so the same URL can only ever return the same bytes. ``private``
    because the URL is session-scoped and a shared cache serving it to a second
    caller would be serving them someone else's report.
    """
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        report = await uow.reports.get_artifact(report_id, session_id=session_id)
        if report is None:
            raise NotFound("no such report", report_id=report_id)
        carried = {f["figure_id"] for f in report.get("figures") or ()}
        if figure_id not in carried:
            # 404 rather than 403: whether a figure exists elsewhere is not
            # something this endpoint should be willing to confirm.
            raise NotFound(
                "this report does not carry that figure",
                report_id=report_id, figure_id=figure_id,
            )
        row = await uow.reports.get_figure(figure_id, session_id=session_id)
        if row is None:
            raise NotFound("no such figure", figure_id=figure_id)
        attachment = await uow.attachments.get(
            row["attachment_id"], owner_id=principal.subject_id
        )
    if attachment is None or attachment.session_id != session_id:
        raise NotFound("the figure's attachment is unavailable", figure_id=figure_id)
    if attachment.media_type != row["media_type"]:
        raise NotFound("the figure's stored media type does not match", figure_id=figure_id)
    try:
        data = await services.object_store.get(ObjectRef(attachment.object_uri))
    except ObjectNotFound as exc:
        raise NotFound("the figure bytes are unavailable", figure_id=figure_id) from exc
    if hashlib.sha256(data).hexdigest() != row["content_sha256"]:
        raise NotFound(
            "the stored figure bytes do not match the recorded content hash",
            figure_id=figure_id,
        )
    return Response(
        data,
        media_type=row["media_type"],
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            "ETag": f'"{row["content_sha256"]}"',
            # An SVG served as a document could script; served as an image it
            # cannot. Belt and braces over the sanitizer, which is the thing
            # actually relied on.
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'inline; filename="{figure_id}.svg"',
        },
    )


@router.get("/sessions/{session_id}/report-builds/{build_id}")
async def get_report_build(
    request: Request, session_id: str, build_id: str,
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        build = await uow.reports.get_build(build_id, session_id=session_id)
    if build is None:
        raise NotFound("no such report build", build_id=build_id)
    return build.to_dict()


@router.post("/sessions/{session_id}/report-builds/{build_id}:cancel", response_model=CancelResponse)
async def cancel_report_build(
    request: Request, session_id: str, build_id: str,
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        build = await uow.reports.get_build(build_id, session_id=session_id)
    if build is None:
        raise NotFound("no such report build", build_id=build_id)
    outcome = await services.scheduler.cancel(build.run_id)
    if outcome.requested:
        from ..domain.report import BuildStage
        async with services.database.unit_of_work() as uow:
            current = await uow.reports.get_build(build_id, session_id=session_id)
            if current is not None and not current.is_terminal:
                await uow.reports.save_build(current.advance(BuildStage.CANCELLED, now=datetime.now(timezone.utc)))
                uow.emit(
                    session_id=session_id, type=EventType.REPORT_CANCELLED,
                    entity_type="report_build", entity_id=build_id, run_id=build.run_id,
                    payload={"stage": "cancelled"},
                )
                await uow.commit()
    return CancelResponse(**outcome.to_dict())


# --- runs ------------------------------------------------------------------

@router.get("/sessions/{session_id}/runs/{run_id}")
async def get_run(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise NotFound("no such run", run_id=run_id)
        binding = (
            await uow.runtime_bindings.get(run.runtime_binding_id)
            if run.runtime_binding_id else None
        )
        tool_calls = await uow.tool_calls.list_for_run(run_id)
        usage_events = await uow.runtime_usage.list_for_run(run_id)
        configuration_snapshot = await uow.run_configuration_snapshots.get(run_id)
    projection = run_projection(run)
    projection["runtime"] = binding.manifest() if binding else None
    projection["configuration_snapshot"] = configuration_snapshot
    projection["usage"] = {
        # No runtime event is different from an explicit event containing
        # input=0/output=0. Consumers must not turn unavailable into zero.
        "status": "reported" if usage_events else "unknown",
        "events": [event.to_dict() for event in usage_events],
    }
    projection["tool_calls"] = [
        {
            "call_id": c["id"], "tool_name": c["tool_name"], "status": c["status"],
            "error_code": c["error_code"], "duration_ms": c["duration_ms"],
            # A hash, never the arguments: enough for a trajectory grader to
            # see a repeated call without exposing what the model sent.
            "arguments_sha256": c.get("arguments_sha256"),
            "observation_ids": list(c.get("observation_ids") or ()),
            "started_at": _isoformat(c["started_at"]),
            "ended_at": _isoformat(c["ended_at"]),
        }
        for c in tool_calls
    ]
    return projection


@router.get("/sessions/{session_id}/runs/{run_id}/decision-state")
async def get_decision_state(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    """The run's DecisionSupportStateV1: goal, propositions, coverage, usage
    and stop reason. 404 for a run that keeps none (every non-decision_support
    run, and decision_support runs admitted before the state existed)."""
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise NotFound("no such run", run_id=run_id)
        state = await uow.decision_states.get(run_id)
    if state is None:
        raise NotFound("this run keeps no decision-support state", run_id=run_id)
    return state.to_dict()


@router.get("/sessions/{session_id}/runs/{run_id}/evidence-relations")
async def list_evidence_relations(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    """What the run judged each source to say about each proposition.

    The relations an accepted grounded-answer-v2 draft proposed and the server
    resolved (``domain/evidence_relation.py``) were stored and never readable:
    a client could see the answer and the sources it cited, but not the
    product's own ``supports``/``contradicts`` assessment. The proposition text
    is in the same run's ``decision-state``, keyed by ``proposition_id``.
    Empty for a run whose answer schema carries no relations (W7-04).
    """
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise NotFound("no such run", run_id=run_id)
        relations = await uow.evidence_relations.list_for_run(run_id)
    return {"evidence_relations": [relation.to_dict() for relation in relations]}


# --- scientific cases (ADR 0012) ---------------------------------------------

def _case_summary(case) -> dict[str, Any]:
    return {
        "case_id": case.id, "question": case.question, "subject_key": case.subject_key,
        "subject_refs": list(case.subject_refs), "status": case.status,
        "revision": case.revision, "hypotheses": len(case.hypotheses),
        "evidence": len(case.evidence), "open_uncertainties": len(case.open_uncertainties),
        "runs": len(case.runs), "coverage": case.coverage, "updated_at": case.updated_at,
        "requester": case.requester, "external_search": case.data_scope.external_search,
    }


async def _owned_case(request: Request, principal: Actor, session_id: str, case_id: str):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        case = await uow.scientific_cases.get(case_id, session_id=session_id)
    if case is None:
        raise NotFound("no such scientific case", case_id=case_id)
    return services, case


@router.get("/sessions/{session_id}/cases")
async def list_scientific_cases(
    request: Request, session_id: str, principal: Actor = Depends(actor)
):
    """The session's scientific cases, newest first. Empty on a deployment
    that never had ``scientific_case_v1`` on."""
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        cases = await uow.scientific_cases.list_for_session(session_id)
    return {"cases": [_case_summary(case) for case in cases]}


@router.get("/sessions/{session_id}/cases/{case_id}")
async def get_scientific_case(
    request: Request, session_id: str, case_id: str, principal: Actor = Depends(actor)
):
    _, case = await _owned_case(request, principal, session_id, case_id)
    return case.to_dict()


@router.get("/sessions/{session_id}/cases/{case_id}/events")
async def get_scientific_case_events(
    request: Request, session_id: str, case_id: str, principal: Actor = Depends(actor)
):
    """The append-only log the case is the fold of: who changed what, in which run."""
    services, case = await _owned_case(request, principal, session_id, case_id)
    async with services.database.unit_of_work() as uow:
        events = await uow.scientific_cases.events(case.id, session_id=session_id)
    return {"case_id": case.id, "events": [event.to_dict() for event in events]}


@router.get("/sessions/{session_id}/cases/{case_id}/dossier")
async def get_latest_decision_dossier(
    request: Request, session_id: str, case_id: str, principal: Actor = Depends(actor)
):
    services, case = await _owned_case(request, principal, session_id, case_id)
    async with services.database.unit_of_work() as uow:
        dossier = await uow.scientific_cases.latest_dossier(case.id, session_id=session_id)
    if dossier is None:
        raise NotFound("no run of this case has finished yet", case_id=case_id)
    return dossier


@router.get("/sessions/{session_id}/runs/{run_id}/dossier")
async def get_run_decision_dossier(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    """The DecisionDossierV1 one run compiled. 404 for a run without a case."""
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        dossier = await uow.scientific_cases.get_dossier(run_id, session_id=session_id)
    if dossier is None:
        raise NotFound("this run compiled no decision dossier", run_id=run_id)
    return dossier


async def _user_case_update(request: Request, principal: Actor, session_id: str, case_id: str,
                            op: str, **payload: Any):
    from ..application import scientific_case_service
    from ..domain import scientific_case as sc
    from ..flags import is_enabled

    if not is_enabled("scientific_case_v1"):
        raise NotFound("scientific cases are not enabled on this deployment")
    services, case = await _owned_case(request, principal, session_id, case_id)
    async with services.database.unit_of_work() as uow:
        try:
            updated = await scientific_case_service.apply_updates(
                uow, case_id=case.id, session_id=session_id,
                updates=[scientific_case_service.update(op, actor=sc.Actor.USER.value, **payload)],
            )
        except sc.InvalidCaseUpdate as exc:
            raise InvalidRequest(str(exc), case_id=case_id) from None
        await uow.commit()
    return updated.to_dict()


@router.post("/sessions/{session_id}/cases/{case_id}/context")
async def add_scientific_case_context(
    request: Request, session_id: str, case_id: str, body: CaseContextRequest,
    principal: Actor = Depends(actor),
):
    """Add researcher-supplied context; the next turn sees it in the case."""
    return await _user_case_update(
        request, principal, session_id, case_id, "add_context",
        key=body.key, value=body.value, note=body.note or "",
    )


@router.post("/sessions/{session_id}/cases/{case_id}/question")
async def set_scientific_case_question(
    request: Request, session_id: str, case_id: str, body: CaseQuestionRequest,
    principal: Actor = Depends(actor),
):
    payload: dict[str, Any] = {"question": body.question}
    if body.decision_context:
        payload["decision_context"] = body.decision_context
    return await _user_case_update(request, principal, session_id, case_id, "set_question", **payload)


@router.post("/sessions/{session_id}/cases/{case_id}/scope")
async def set_scientific_case_scope(
    request: Request, session_id: str, case_id: str, body: CaseScopeRequest,
    principal: Actor = Depends(actor),
):
    """Set what the case may reach. With external search off, the evidence
    search tool refuses for every later turn of this case."""
    return await _user_case_update(
        request, principal, session_id, case_id, "set_scope",
        external_search=body.external_search, reason=body.reason or "",
    )


@router.post("/sessions/{session_id}/cases/{case_id}:close")
async def close_scientific_case(
    request: Request, session_id: str, case_id: str, principal: Actor = Depends(actor)
):
    """Close the case; the next turn on this subject opens a new one."""
    return await _user_case_update(request, principal, session_id, case_id, "close")


# --- skill drafts (RETHINK §4.8, W9-11) --------------------------------------

def _drafts_enabled() -> None:
    from ..flags import is_enabled

    if not is_enabled("skill_drafts_v1"):
        raise NotFound("skill drafts are not enabled on this deployment")


async def _visible_draft(request: Request, principal: Actor, draft_id: str):
    """An expert sees every draft; anyone else sees only their own."""
    from ..domain.skill_draft import REVIEWER_ROLE

    _drafts_enabled()
    async with _services(request).database.unit_of_work() as uow:
        draft = await uow.skill_drafts.get(draft_id)
    if draft is None or (
        not principal.has_role(REVIEWER_ROLE) and draft.author.subject_id != principal.subject_id
    ):
        raise NotFound("no such skill draft", draft_id=draft_id)
    return draft


@router.post("/skill-drafts", status_code=201)
async def propose_skill_draft(
    request: Request, body: SkillDraftRequest, principal: Actor = Depends(actor)
):
    """Propose a skill package for expert review. It is validated exactly as a
    shipped skill would be, and never offered to a run."""
    from ..application import skill_drafts
    from ..domain.skill_draft import DraftAuthor

    _drafts_enabled()
    services = _services(request)
    async with services.database.unit_of_work() as uow:
        try:
            draft = await skill_drafts.propose(
                uow, author=DraftAuthor(actor="user", subject_id=principal.subject_id),
                skill_md=body.skill_md, manifest=body.manifest, references=body.references,
                rationale=body.rationale, catalog=services.skill_catalog,
            )
        except skill_drafts.DraftRefused as exc:
            raise InvalidRequest(str(exc)) from None
        await uow.commit()
    return draft.to_dict()


@router.get("/skill-drafts")
async def list_skill_drafts(
    request: Request, status: str | None = Query(default=None, max_length=16),
    principal: Actor = Depends(actor),
):
    from ..domain.skill_draft import REVIEWER_ROLE

    _drafts_enabled()
    async with _services(request).database.unit_of_work() as uow:
        drafts = await uow.skill_drafts.list(
            status=status,
            author_subject=None if principal.has_role(REVIEWER_ROLE) else principal.subject_id,
        )
    return {"drafts": [
        {key: value for key, value in d.to_dict().items() if key not in ("skill_md", "references")}
        for d in drafts
    ]}


@router.get("/skill-drafts/{draft_id}")
async def get_skill_draft(request: Request, draft_id: str, principal: Actor = Depends(actor)):
    return (await _visible_draft(request, principal, draft_id)).to_dict()


@router.post("/skill-drafts/{draft_id}:review")
async def review_skill_draft(
    request: Request, draft_id: str, body: SkillDraftReviewRequest,
    principal: Actor = Depends(actor),
):
    """Approve or reject a proposed draft. Needs the expert role; an author
    does not review their own draft. Approval does not reach the catalog."""
    from ..application import skill_drafts
    from ..domain.errors import Forbidden
    from ..domain.skill_draft import InvalidDraftTransition

    await _visible_draft(request, principal, draft_id)
    async with _services(request).database.unit_of_work() as uow:
        try:
            draft = await skill_drafts.review(
                uow, draft_id=draft_id, reviewer=principal.subject_id,
                reviewer_roles=principal.roles, decision=body.decision, note=body.note,
            )
        except InvalidDraftTransition as exc:
            raise Forbidden(str(exc), draft_id=draft_id) from None
        await uow.commit()
    return draft.to_dict()


@router.post("/skill-drafts/{draft_id}:withdraw")
async def withdraw_skill_draft(request: Request, draft_id: str, principal: Actor = Depends(actor)):
    from ..application import skill_drafts
    from ..domain.errors import Forbidden
    from ..domain.skill_draft import InvalidDraftTransition

    await _visible_draft(request, principal, draft_id)
    async with _services(request).database.unit_of_work() as uow:
        try:
            draft = await skill_drafts.withdraw(uow, draft_id=draft_id, by=principal.subject_id)
        except InvalidDraftTransition as exc:
            raise Forbidden(str(exc), draft_id=draft_id) from None
        await uow.commit()
    return draft.to_dict()


@router.get("/skill-drafts/{draft_id}/package")
async def export_skill_draft(request: Request, draft_id: str, principal: Actor = Depends(actor)):
    """An approved draft as files, with the manifest made active — the input to
    ``scripts/promote_skill_draft.py`` and a reviewed change to the catalog."""
    from ..application.skill_drafts import package_digest
    from ..domain.errors import Conflict
    from ..domain.skill_draft import InvalidDraftTransition

    draft = await _visible_draft(request, principal, draft_id)
    try:
        files = draft.package()
    except InvalidDraftTransition as exc:
        raise Conflict(str(exc), draft_id=draft_id) from None
    return {"draft_id": draft.id, "skill_id": draft.skill_id, "version": draft.version,
            "review": draft.review.to_dict() if draft.review else None,
            "files": files, "package_sha256": package_digest(files)}


@router.post("/sessions/{session_id}/runs/{run_id}:cancel", response_model=CancelResponse)
async def cancel_run(
    request: Request, session_id: str, run_id: str, principal: Actor = Depends(actor)
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise NotFound("no such run", run_id=run_id)
        binding = (
            await uow.runtime_bindings.get(run.runtime_binding_id)
            if run.runtime_binding_id else None
        )
    supported = bool(binding and binding.capabilities.cancel_turn)
    outcome = await services.scheduler.cancel(run_id, runtime_cancel_supported=supported)
    return CancelResponse(**outcome.to_dict())


# --- analyses, answers, evidence -------------------------------------------

@router.get("/sessions/{session_id}/analyses/{analysis_id}")
async def get_analysis(
    request: Request,
    session_id: str,
    analysis_id: str,
    include_raw: bool = Query(False),
    principal: Actor = Depends(actor),
):
    from ..application.projections import display_projection

    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        snapshot = await uow.analyses.get(analysis_id, session_id=session_id)
    if snapshot is None:
        raise AnalysisNotFound("no such analysis", analysis_id=analysis_id)
    projection = display_projection(snapshot)
    if include_raw and principal.has_role("auditor"):
        # The lossless payload is audit material, not a default response body.
        projection["predictor_response"] = snapshot.predictor_response
    return projection


@router.get("/sessions/{session_id}/analyses/{analysis_id}/attributions")
async def list_attributions(
    request: Request,
    session_id: str,
    analysis_id: str,
    principal: Actor = Depends(actor),
):
    """List bounded attribution observations for one immutable analysis.

    The projection is intentionally the same top-token view supplied to the
    model, rather than canonical/raw provider output. An attribution belongs
    to exactly one endpoint (and one Tox21 task when applicable), so the UI
    cannot construct an aggregate explanation from this endpoint.
    """
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        snapshot = await uow.analyses.get(analysis_id, session_id=session_id)
        if snapshot is None:
            raise AnalysisNotFound("no such analysis", analysis_id=analysis_id)
        observations = await uow.observations.list_for_analysis(analysis_id)
    return {
        "attributions": [
            {
                "observation_id": observation.id,
                "run_id": observation.run_id,
                "created_at": observation.created_at.isoformat(),
                "content_sha256": observation.content_sha256,
                "required_limitations": list(observation.required_limitations),
                **observation.model_projection,
            }
            for observation in observations
            if observation.kind is ObservationKind.ATTRIBUTION
        ]
    }


@router.get("/sessions/{session_id}/answers/{answer_id}")
async def get_answer(
    request: Request, session_id: str, answer_id: str, principal: Actor = Depends(actor)
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        answer = await uow.answers.get(answer_id, session_id=session_id)
    if answer is None:
        raise NotFound("no such answer", answer_id=answer_id)
    return answer.to_dict()


@router.get("/sessions/{session_id}/observations/{observation_id}")
async def get_observation(
    request: Request,
    session_id: str,
    observation_id: str,
    principal: Actor = Depends(actor),
):
    """The other end of every claim's ``observation_id`` (plan section 5.5).

    Without this, ``field_path`` and ``source_value`` on a claim are citations
    to nothing a client can open. The lossless ``canonical_payload`` stays
    audit-only, same gate as ``analyses`` ``include_raw``: a claim only ever
    needed the bounded ``model_projection`` to be valid.
    """
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        observation = await uow.observations.get(observation_id, session_id=session_id)
    if observation is None:
        raise NotFound("no such observation", observation_id=observation_id)
    body: dict[str, Any] = {
        "observation_id": observation.id,
        "run_id": observation.run_id,
        "producer": observation.producer.value,
        "kind": observation.kind.value,
        "schema_version": observation.schema_version,
        "model_projection": observation.model_projection,
        "provenance": observation.provenance,
        "required_limitations": list(observation.required_limitations),
        "content_sha256": observation.content_sha256,
        "created_at": observation.created_at.isoformat(),
    }
    if principal.has_role("auditor"):
        body["canonical_payload"] = observation.canonical_payload
    return body


@router.get("/sessions/{session_id}/evidence")
async def list_evidence(
    request: Request,
    session_id: str,
    status: str = Query("accepted"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        records = await uow.evidence.list_for_session(
            session_id,
            status=EvidenceStatus(status) if status != "all" else None,
            limit=limit, offset=offset,
        )
    return {
        "evidence": [
            {
                **record.model_view(),
                "status": record.status.value,
                "provider": record.provider,
                "retrieved_at": record.retrieved_at.isoformat(),
                "content_sha256": record.content_sha256,
            }
            for record in records
        ],
        "count": len(records),
    }


@router.get("/sessions/{session_id}/evidence/{evidence_id}")
async def get_evidence(
    request: Request,
    session_id: str,
    evidence_id: str,
    principal: Actor = Depends(actor),
):
    """Return the bounded, normalized evidence projection for its owner.

    This deliberately mirrors the model-visible view, plus audit-safe
    transport metadata. ``raw_payload_ref`` remains object-store/auditor
    material (W4-09) and is never a browser URL or a model capability.
    """
    services = _services(request)
    await services.sessions.get(principal, session_id)
    async with services.database.unit_of_work() as uow:
        record = await uow.evidence.get(evidence_id, session_id=session_id)
    if record is None:
        raise NotFound("no such evidence", evidence_id=evidence_id)
    return {
        **record.model_view(),
        "status": record.status.value,
        "provider": record.provider,
        "retrieved_at": record.retrieved_at.isoformat(),
        "content_sha256": record.content_sha256,
    }


# --- change feed -----------------------------------------------------------

@router.get("/sessions/{session_id}/events")
async def stream_events(
    request: Request,
    session_id: str,
    after_sequence: int = Query(0, ge=0),
    principal: Actor = Depends(actor),
):
    services = _services(request)
    await services.sessions.get(principal, session_id)
    last_event_id = request.headers.get("last-event-id")
    cursor = after_sequence
    if last_event_id and last_event_id.isdigit():
        # Last-Event-ID wins: it is what the browser resends automatically, and
        # a stale query parameter would silently replay events the client has.
        cursor = int(last_event_id)
    return EventSourceResponse(
        event_stream(services.database.outbox(), services.notifier, session_id, after_sequence=cursor)
    )


@router.get("/sessions/{session_id}/events:list")
async def list_events(
    request: Request,
    session_id: str,
    after_sequence: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=500),
    run_id: str | None = Query(None),
    principal: Actor = Depends(actor),
):
    """A non-streaming read of the same outbox the SSE feed serves.

    The stream never terminates, so it is the wrong tool for "replay
    everything that happened in this one run" (a Run Inspector opened after
    the fact, or a page that reconnected and needs to fill a gap). The outbox
    row is retained forever — nothing here is a delivery guarantee beyond what
    ``/events`` already gives; this just lets a client stop listening.
    """
    services = _services(request)
    session = await services.sessions.get(principal, session_id)
    events = await services.database.outbox().read_after(
        session_id, after_sequence, limit=limit, run_id=run_id
    )
    return {
        "events": [e.to_dict() for e in events],
        "count": len(events),
        "latest_sequence": session.event_sequence,
    }
