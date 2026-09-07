"""Product HTTP routes (plan sections 6.1-6.5).

Handlers stay thin: authenticate, parse, delegate, serialise. Ownership is
enforced in the application and the store, not here, so a new endpoint cannot
forget it. Reads are all reconstructions of committed state, which is what makes
"the stream died" a non-event.
"""
from __future__ import annotations

import base64
import binascii
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..application.policy import Actor
from ..application.sessions import run_projection
from ..application.submit_message import MessageSubmission
from ..domain.errors import (
    AnalysisNotFound,
    CapabilityUnavailable,
    InvalidRequest,
    NotFound,
    SmilesNotDetected,
    StructureRecognitionUnavailable,
)
from ..domain.evidence import EvidenceStatus
from ..domain.observation import ObservationKind
from ..domain.run import Intent
from ..predictor.ocr_client import OcrError, OcrUnavailable
from ..streaming.sse import event_stream
from ._image import decode_declared_image, matches_declared_image_type
from .schemas import (
    AcceptedResponse,
    CancelResponse,
    CreateSessionRequest,
    ExplainRequest,
    PredictBatchRequest,
    PredictRequest,
    RecognizedStructure,
    RecognizeRequest,
    SendMessageRequest,
    SessionResponse,
)

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
    """Readiness of this control plane and, separately, of what it depends on.

    The predictor's readiness is reported, never merged into one boolean: a
    control plane that can serve sessions and reads while the predictor is down
    is in a different state from one that cannot start at all.

    A deployment with no report_qa handler registered used to still answer
    `ready=true` here, since this endpoint only ever probed the predictor and
    named the configured runtime *kind* — never whether a conversational
    intent could actually run. `capabilities` reports what
    `RunScheduler.handles()` actually has registered, and, when a runtime
    gateway is configured, its live health is probed the same way a real turn
    would probe it.
    """
    services = _services(request)
    dependencies: dict[str, Any] = {}
    ok = True
    try:
        readiness = await services.predictor.ready()
        dependencies["predictor"] = {
            "ready": readiness.ready, "served_endpoints": readiness.served_endpoints
        }
        ok = ok and readiness.ready
    except Exception as exc:  # noqa: BLE001 — reported as a dependency state
        dependencies["predictor"] = {"ready": False, "reason": type(exc).__name__}
        ok = False

    capabilities = {
        "analysis": services.scheduler.handles(Intent.ANALYSIS),
        "report_qa": services.scheduler.handles(Intent.REPORT_QA),
        "attribution": services.scheduler.handles(Intent.ATTRIBUTION),
        "evidence_research": services.scheduler.handles(Intent.EVIDENCE_RESEARCH),
        # No runtime gateway involved — a plain deterministic handler, same as
        # "analysis" — so it is reported here but left out of
        # conversational_registered below.
        "structure_recognition": services.scheduler.handles(Intent.STRUCTURE_RECOGNITION),
    }
    conversational_registered = any(
        capabilities[name] for name in ("report_qa", "attribution", "evidence_research")
    )
    runtime_info: dict[str, Any] = {"kind": services.settings.runtime.kind}
    gateway = services.runtime_gateway
    if gateway is not None:
        try:
            runtime_info["healthy"] = await gateway.health()
        except Exception as exc:  # noqa: BLE001 — reported as a dependency state
            runtime_info["healthy"] = False
            runtime_info["reason"] = type(exc).__name__
        ok = ok and runtime_info["healthy"]
    elif conversational_registered:
        # A conversational intent is registered with no runtime gateway
        # behind it — should not happen, but readiness must not lie if it does.
        runtime_info["healthy"] = False
        ok = False
    dependencies["runtime"] = runtime_info
    dependencies["capabilities"] = capabilities
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, **dependencies})


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
            threshold_overrides=body.threshold_overrides,
        )


@router.get("/predict/capabilities")
async def predict_capabilities(request: Request, principal: Actor = Depends(actor)):
    """A straight proxy of what the predictor actually serves, so the UI can
    render an unserved endpoint (ClinTox on this build) as disabled with a real
    reason rather than guessing."""
    services = _services(request)
    models = await services.predictor.models()
    return {
        "served_endpoints": list(models.served_endpoints),
        "models": [model.model_dump(mode="json") for model in models.models],
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
    )


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str, principal: Actor = Depends(actor)):
    return await _services(request).sessions.projection(principal, session_id)


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
            threshold_overrides=options.threshold_overrides if options else None,
            include_attribution=options.include_attribution if options else False,
            analysis_id=body.analysis_id,
            image_mime_type=image_mime_type,
            image_size_bytes=image_size_bytes,
            image_bytes=image_bytes,
        ),
    )
    return AcceptedResponse(**accepted.to_dict())


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
    projection = run_projection(run)
    projection["runtime"] = binding.manifest() if binding else None
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
            "started_at": _isoformat(c["started_at"]),
            "ended_at": _isoformat(c["ended_at"]),
        }
        for c in tool_calls
    ]
    return projection


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
