"""The HTTP layer, with a fake predictor — no real MolScribe model, no
network. What this exercises is FastAPI wiring and error mapping: base64
decoding, the size cap, and how ImageDecodeError/StructureNotDetected turn
into the response codes toxagent-control's OcrClient expects.
"""
from __future__ import annotations

import base64
from contextlib import asynccontextmanager

import httpx
import pytest

from toxocr.api.app import create_app
from toxocr.scientific.molscribe_predictor import (
    ImageDecodeError,
    RecognitionResult,
    StructureNotDetected,
)
from toxocr.settings import Settings

pytestmark = pytest.mark.anyio

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class FakePredictor:
    """Returns a fixed outcome — the FastAPI layer around it is what's under
    test here, not MolScribe itself (see the toxocr-env smoke test for that)."""

    def __init__(self, *, result: RecognitionResult | None = None, raise_error: Exception | None = None) -> None:
        self.result = result
        self.raise_error = raise_error
        self.calls: list[bytes] = []

    def is_ready(self) -> bool:
        return True

    def preload(self) -> None:
        pass

    def runtime_status(self) -> tuple[str, str | None]:
        return "cpu", "sha256:test"

    def recognize(self, raw_bytes: bytes) -> RecognitionResult:
        self.calls.append(raw_bytes)
        if self.raise_error is not None:
            raise self.raise_error
        assert self.result is not None
        return self.result


@asynccontextmanager
async def _client(predictor, *, settings: Settings | None = None):
    app = create_app(settings or Settings(eager_load=False), predictor=predictor)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://toxocr.test"
        ) as client:
            yield client


async def test_ready_reports_the_predictor_state():
    predictor = FakePredictor(result=RecognitionResult(ASPIRIN, ASPIRIN, 0.9))
    async with _client(predictor) as client:
        response = await client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"ready": True, "device": "cpu", "checkpoint_fingerprint": "sha256:test"}


async def test_a_recognised_structure_returns_its_smiles():
    predictor = FakePredictor(result=RecognitionResult(ASPIRIN, ASPIRIN, 0.87))
    async with _client(predictor) as client:
        response = await client.post(
            "/v1/structure-recognition",
            json={"mime_type": "image/png", "data_base64": base64.b64encode(b"pretend-png-bytes").decode()},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body == {"smiles": ASPIRIN, "canonical_smiles": ASPIRIN, "confidence": 0.87}
        assert predictor.calls == [b"pretend-png-bytes"]


async def test_malformed_base64_is_a_client_error_not_a_500():
    async with _client(FakePredictor()) as client:
        response = await client.post(
            "/v1/structure-recognition",
            json={"mime_type": "image/png", "data_base64": "not valid base64!!"},
        )
        assert response.status_code == 400


async def test_an_oversized_image_is_rejected_before_inference():
    predictor = FakePredictor(result=RecognitionResult(ASPIRIN, ASPIRIN, 0.9))
    async with _client(predictor, settings=Settings(eager_load=False, max_image_bytes=10)) as client:
        response = await client.post(
            "/v1/structure-recognition",
            json={"mime_type": "image/png", "data_base64": base64.b64encode(b"x" * 100).decode()},
        )
        assert response.status_code == 400
    assert predictor.calls == []


async def test_no_structure_detected_is_a_422():
    predictor = FakePredictor(raise_error=StructureNotDetected("no SMILES sequence was detected"))
    async with _client(predictor) as client:
        response = await client.post(
            "/v1/structure-recognition",
            json={"mime_type": "image/png", "data_base64": base64.b64encode(b"noise").decode()},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "smiles_not_detected"


async def test_an_undecodable_image_is_a_415():
    predictor = FakePredictor(raise_error=ImageDecodeError("image content could not be decoded"))
    async with _client(predictor) as client:
        response = await client.post(
            "/v1/structure-recognition",
            json={"mime_type": "image/png", "data_base64": base64.b64encode(b"not an image").decode()},
        )
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_image_format"
