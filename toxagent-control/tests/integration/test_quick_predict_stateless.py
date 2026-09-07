"""Quick Predict writes zero rows (plan section 3.2, invariant checklist).

The point of the path is that it bypasses the session lifecycle entirely. If a
``/v1/predict`` call ever creates a run, an analysis, an observation or an outbox
event, that guarantee is broken — so this asserts the counts directly.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from tests.support.api import AUTH, api_client
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

TABLES = ("runs", "analysis_snapshots", "observations", "event_outbox")


async def _counts(db) -> dict[str, int]:
    async with db.engine.connect() as connection:
        return {
            table: await connection.scalar(text(f"SELECT count(*) FROM {table}"))
            for table in TABLES
        }


async def test_a_predict_call_changes_no_row_counts(db):
    async with api_client(db, StubPredictor()) as client:
        before = await _counts(db)
        response = await client.post("/v1/predict", json={"smiles": ASPIRIN}, headers=AUTH)
        assert response.status_code == 200, response.text
        after = await _counts(db)

    assert before == after
    body = response.json()
    assert body["persisted"] is False
    assert body["analysis_id"] is None


async def test_repeated_calls_still_write_nothing(db):
    async with api_client(db, StubPredictor()) as client:
        before = await _counts(db)
        for _ in range(3):
            assert (
                await client.post("/v1/predict", json={"smiles": ASPIRIN}, headers=AUTH)
            ).status_code == 200
        after = await _counts(db)
    assert before == after
