"""``POST /v1/predict`` — the stateless predictor proxy (plan section 3.2).

Same typed-error discipline and provenance discipline as Lane D, none of the
session machinery. The predictor is the stub with its ``httpx.MockTransport``.
"""
from __future__ import annotations

import pytest

from dataclasses import replace

from sqlalchemy import text

from toxagent.config import PolicySettings, PredictorSettings
from tests.support.api import AUTH, EXPERT_AUTH, api_client, settings
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio


async def test_the_shape_is_the_display_projection_and_it_is_not_persisted(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post("/v1/predict", json={"smiles": ASPIRIN}, headers=AUTH)
        assert response.status_code == 200, response.text
        body = response.json()

    assert body["persisted"] is False
    assert body["analysis_id"] is None
    assert body["canonical_smiles"] == ASPIRIN
    assert set(body["served_endpoints"]) == {"herg", "tox21"}
    assert body["sections"]["herg"]["probability_blocker"] == 0.73064
    assert len(body["sections"]["tox21"]["assays"]) == 12
    # Provenance copied verbatim (SCI-10).
    assert body["provenance"]["predictor_service_version"] == "0.1.0.dev0"
    assert body["provenance"]["artifact_hashes"]
    assert "uncalibrated_probability" in body["required_limitations"]


async def test_an_unknown_field_is_refused(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post(
            "/v1/predict", json={"smiles": ASPIRIN, "persist": True}, headers=AUTH
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


async def test_the_endpoint_filter_is_honoured(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post(
            "/v1/predict", json={"smiles": ASPIRIN, "endpoints": ["herg"]}, headers=AUTH
        )
    body = response.json()
    assert body["served_endpoints"] == ["herg"]
    assert "tox21" not in body["sections"]


async def test_an_invalid_smiles_is_a_typed_error_never_a_prediction(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post(
            "/v1/predict", json={"smiles": "not-a-molecule"}, headers=AUTH
        )
    assert response.status_code in (400, 422)
    assert response.json()["error"]["code"] == "invalid_smiles"


async def test_clintox_is_endpoint_unavailable_not_a_retryable_503(db):
    async with api_client(db, StubPredictor(served=("herg", "tox21"))) as client:
        response = await client.post(
            "/v1/predict", json={"smiles": ASPIRIN, "endpoints": ["clintox"]}, headers=AUTH
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "endpoint_unavailable"
    assert body["error"]["retryable"] is False


async def test_a_predictor_that_is_not_ready_is_503_retryable(db):
    async with api_client(db, StubPredictor(fail_with=503)) as client:
        response = await client.post("/v1/predict", json={"smiles": ASPIRIN}, headers=AUTH)
    assert response.status_code == 503
    assert response.json()["error"]["retryable"] is True


async def test_threshold_override_by_a_non_expert_is_forbidden(db):
    config = settings(policy=PolicySettings(allow_threshold_overrides=True))
    async with api_client(db, StubPredictor(), config=config) as client:
        response = await client.post(
            "/v1/predict",
            json={"smiles": ASPIRIN, "threshold_overrides": {"herg": 0.3}},
            headers=AUTH,
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


async def test_threshold_override_by_an_expert_is_applied(db):
    config = settings(policy=PolicySettings(allow_threshold_overrides=True))
    stub = StubPredictor()
    async with api_client(db, stub, config=config) as client:
        response = await client.post(
            "/v1/predict",
            json={"smiles": ASPIRIN, "threshold_overrides": {"herg": 0.3}},
            headers=EXPERT_AUTH,
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["policy_snapshot"]["threshold_override_source"] == "request_override"
    assert stub.requests[-1]["body"]["threshold_overrides"] == {"herg": 0.3}


async def test_a_bearer_token_is_required(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post("/v1/predict", json={"smiles": ASPIRIN})
    assert response.status_code == 401


async def test_batch_preserves_order_and_reports_per_item_errors(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post(
            "/v1/predict:batch",
            json={"smiles": ["CCO", "not-a-molecule", ASPIRIN], "endpoints": ["herg"]},
            headers=AUTH,
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 3
    assert [r["input_smiles"] for r in body["results"]] == ["CCO", ASPIRIN]
    assert all(r["persisted"] is False and r["analysis_id"] is None for r in body["results"])
    assert body["errors"][0]["index"] == 1
    assert body["errors"][0]["error"] == "invalid_smiles"


async def test_batch_over_the_maximum_is_422(db):
    config = replace(
        settings(),
        predictor=PredictorSettings(base_url="http://predictor.test", max_batch_size=2),
    )
    async with api_client(db, StubPredictor(), config=config) as client:
        response = await client.post(
            "/v1/predict:batch", json={"smiles": ["CCO", "CCO", "CCO"]}, headers=AUTH
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


async def test_batch_writes_no_rows(db):
    async with api_client(db, StubPredictor()) as client:
        async with db.engine.connect() as conn:
            before = await conn.scalar(text("SELECT count(*) FROM analysis_snapshots"))
        assert (
            await client.post(
                "/v1/predict:batch", json={"smiles": ["CCO", ASPIRIN]}, headers=AUTH
            )
        ).status_code == 200
        async with db.engine.connect() as conn:
            after = await conn.scalar(text("SELECT count(*) FROM analysis_snapshots"))
    assert before == after == 0


async def test_capabilities_proxies_what_the_predictor_serves(db):
    async with api_client(db, StubPredictor(served=("herg", "tox21"))) as client:
        response = await client.get("/v1/predict/capabilities", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert set(body["served_endpoints"]) == {"herg", "tox21"}
    assert body["predictor_id"]
    assert body["models"]
    assert body["ocr_available"] is False
