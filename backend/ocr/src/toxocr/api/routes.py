from __future__ import annotations

import asyncio
import base64
import binascii

from fastapi import APIRouter, HTTPException, Request

from .schemas import ReadinessResponse, StructureRecognitionRequest, StructureRecognitionResponse

health_router = APIRouter(tags=["health"])
v1_router = APIRouter(prefix="/v1", tags=["ocr"])


@health_router.get("/health/ready", response_model=ReadinessResponse)
def ready(request: Request) -> ReadinessResponse:
    predictor = request.app.state.predictor
    device, fingerprint = predictor.runtime_status() if hasattr(predictor, "runtime_status") else ("unknown", None)
    return ReadinessResponse(ready=predictor.is_ready(), device=device, checkpoint_fingerprint=fingerprint)


@v1_router.post("/structure-recognition", response_model=StructureRecognitionResponse)
async def recognize(request: Request, body: StructureRecognitionRequest) -> StructureRecognitionResponse:
    max_bytes = request.app.state.settings.max_image_bytes
    try:
        raw_bytes = base64.b64decode(body.data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "invalid_request", "message": "data_base64 is not valid base64"}},
        ) from exc
    if len(raw_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "invalid_request",
                    "message": f"image of {len(raw_bytes)} bytes exceeds the {max_bytes}-byte limit",
                }
            },
        )

    predictor = request.app.state.predictor
    # MolScribe's forward pass is synchronous CPU/GPU work; running it inline
    # would block every other request this process is holding open.
    result = await asyncio.to_thread(predictor.recognize, raw_bytes)
    return StructureRecognitionResponse(
        smiles=result.smiles, canonical_smiles=result.canonical_smiles, confidence=result.confidence
    )
