"""``POST /v1/predict/recognize`` — the stateless OCR proxy (plan section 4.2).

Decode, check, forward, discard. No run, no analysis, no object store.
"""
from __future__ import annotations

import base64

import pytest
from sqlalchemy import text

from toxagent.config import PolicySettings
from tests.support.api import AUTH, api_client, settings
from tests.support.ocr import stub_no_structure_detected, stub_success, stub_unavailable
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

PNG = b"\x89PNG\r\n\x1a\n" + b"fake raster payload"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _body(data: bytes = PNG, mime_type: str = "image/png") -> dict:
    return {"mime_type": mime_type, "data_base64": _b64(data)}


async def test_a_recognised_structure_is_passed_through_verbatim(db):
    async with api_client(db, StubPredictor(), ocr_client=stub_success(ASPIRIN)) as client:
        response = await client.post("/v1/predict/recognize", json=_body(), headers=AUTH)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"smiles": ASPIRIN, "canonical_smiles": ASPIRIN, "confidence": 0.91}


async def test_no_structure_detected_is_422(db):
    async with api_client(
        db, StubPredictor(), ocr_client=stub_no_structure_detected()
    ) as client:
        response = await client.post("/v1/predict/recognize", json=_body(), headers=AUTH)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "smiles_not_detected"


async def test_an_unreachable_ocr_service_is_503(db):
    async with api_client(db, StubPredictor(), ocr_client=stub_unavailable()) as client:
        response = await client.post("/v1/predict/recognize", json=_body(), headers=AUTH)
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "structure_recognition_unavailable"
    assert body["error"]["retryable"] is True


async def test_an_oversize_image_is_400(db):
    config = settings(policy=PolicySettings(max_image_bytes=32))
    big = b"\x89PNG\r\n\x1a\n" + b"x" * 64
    async with api_client(
        db, StubPredictor(), ocr_client=stub_success(ASPIRIN), config=config
    ) as client:
        response = await client.post(
            "/v1/predict/recognize", json=_body(big), headers=AUTH
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


async def test_a_mime_signature_mismatch_is_400(db):
    async with api_client(db, StubPredictor(), ocr_client=stub_success(ASPIRIN)) as client:
        response = await client.post(
            "/v1/predict/recognize",
            json={"mime_type": "image/jpeg", "data_base64": _b64(PNG)},
            headers=AUTH,
        )
    assert response.status_code == 400
    assert "do not match" in response.json()["error"]["message"]


async def test_bad_base64_is_400(db):
    async with api_client(db, StubPredictor(), ocr_client=stub_success(ASPIRIN)) as client:
        response = await client.post(
            "/v1/predict/recognize",
            json={"mime_type": "image/png", "data_base64": "!!!not base64!!!"},
            headers=AUTH,
        )
    assert response.status_code == 400


async def test_no_ocr_service_configured_is_capability_unavailable(db):
    async with api_client(db, StubPredictor()) as client:  # no ocr_client
        response = await client.post("/v1/predict/recognize", json=_body(), headers=AUTH)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "capability_unavailable"


async def test_a_bearer_token_is_required(db):
    async with api_client(db, StubPredictor(), ocr_client=stub_success(ASPIRIN)) as client:
        response = await client.post("/v1/predict/recognize", json=_body())
    assert response.status_code == 401


async def test_recognition_writes_no_rows(db):
    tables = ("runs", "analysis_snapshots", "observations", "event_outbox", "attachments")

    async def counts():
        async with db.engine.connect() as conn:
            return {t: await conn.scalar(text(f"SELECT count(*) FROM {t}")) for t in tables}

    async with api_client(db, StubPredictor(), ocr_client=stub_success(ASPIRIN)) as client:
        before = await counts()
        assert (
            await client.post("/v1/predict/recognize", json=_body(), headers=AUTH)
        ).status_code == 200
        after = await counts()
    assert before == after
