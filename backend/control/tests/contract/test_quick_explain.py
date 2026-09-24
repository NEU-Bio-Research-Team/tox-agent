"""``POST /v1/predict/explain`` — thin proxy to ToxPred ``/v1/explanations``
(plan section 5.2). Stateless, limiter-guarded, limitation always echoed.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from tests.support.api import AUTH, api_client
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio


async def test_a_herg_explanation_is_passed_through_with_the_limitation_echoed(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post(
            "/v1/predict/explain",
            json={"smiles": ASPIRIN, "endpoint": "herg"},
            headers=AUTH,
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["atom_order_version"] == "rdkit-output-order-v1"
    assert body["atoms"][0]["atom_index"] == 0
    assert "attribution_not_causality" in body["limitations"]


async def test_tox21_without_a_task_is_refused_at_this_boundary(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post(
            "/v1/predict/explain",
            json={"smiles": ASPIRIN, "endpoint": "tox21"},
            headers=AUTH,
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


async def test_tox21_with_a_task_reaches_the_predictor(db):
    stub = StubPredictor()
    async with api_client(db, stub) as client:
        response = await client.post(
            "/v1/predict/explain",
            json={"smiles": ASPIRIN, "endpoint": "tox21", "task": "NR-ER"},
            headers=AUTH,
        )
    assert response.status_code == 200, response.text
    assert stub.requests[-1]["body"]["task"] == "NR-ER"
    assert response.json()["task"] == "NR-ER"


async def test_a_partial_explanation_keeps_its_note(db):
    async with api_client(db, StubPredictor(explain_status="partial")) as client:
        response = await client.post(
            "/v1/predict/explain",
            json={"smiles": ASPIRIN, "endpoint": "herg"},
            headers=AUTH,
        )
    body = response.json()
    assert body["status"] == "partial"
    assert "over the" in body["metadata"]["note"]
    assert "attribution_not_causality" in body["limitations"]


async def test_an_unserved_endpoint_reports_predictor_not_ready(db):
    async with api_client(db, StubPredictor(served=("tox21",))) as client:
        response = await client.post(
            "/v1/predict/explain",
            json={"smiles": ASPIRIN, "endpoint": "herg"},
            headers=AUTH,
        )
    assert response.status_code == 503


async def test_a_bearer_token_is_required(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post(
            "/v1/predict/explain", json={"smiles": ASPIRIN, "endpoint": "herg"}
        )
    assert response.status_code == 401


async def test_explain_writes_no_rows(db):
    tables = ("runs", "analysis_snapshots", "observations", "event_outbox")

    async def counts():
        async with db.engine.connect() as conn:
            return {t: await conn.scalar(text(f"SELECT count(*) FROM {t}")) for t in tables}

    async with api_client(db, StubPredictor()) as client:
        before = await counts()
        assert (
            await client.post(
                "/v1/predict/explain",
                json={"smiles": ASPIRIN, "endpoint": "herg"},
                headers=AUTH,
            )
        ).status_code == 200
        after = await counts()
    assert before == after
