"""``POST /v1/explanations`` request contract (plan section 5.1).

Model-free: schema validation happens before the handler, and the happy path
runs against a stub service set on ``app.state``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from toxpred.api.errors import value_error_handler  # noqa: E402
from toxpred.api.routes import v1_router  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class _StubExplain:
    def explain(self, smiles, endpoint, task=None):
        return {
            "status": "completed",
            "endpoint": endpoint,
            "task": task,
            "input_smiles": smiles,
            "canonical_smiles": smiles,
            "atom_order_version": "rdkit-output-order-v1",
            "probability": 0.42,
            "atoms": [
                {"atom_index": 0, "symbol": "C", "importance": 1.0, "relative_importance": 0.5}
            ],
            "unmapped_importance": 0.5,
            "tokens": [{"token": "C", "importance": 1.0, "offsets": [0, 1]}],
            "method": "grad_x_embedding_l2_v1+token_atom_align_v1",
            "metadata": {"model_id": "m", "deterministic": True, "duration_ms": 3.0, "note": None},
        }


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(v1_router)
    app.add_exception_handler(ValueError, value_error_handler)
    app.state.explain = _StubExplain()
    return TestClient(app)


def test_tox21_without_a_task_is_422(client):
    res = client.post("/v1/explanations", json={"smiles": ASPIRIN, "endpoint": "tox21"})
    assert res.status_code == 422


def test_a_task_on_a_non_tox21_endpoint_is_422(client):
    res = client.post(
        "/v1/explanations",
        json={"smiles": ASPIRIN, "endpoint": "herg", "task": "NR-ER"},
    )
    assert res.status_code == 422


def test_an_unknown_tox21_task_is_422(client):
    res = client.post(
        "/v1/explanations",
        json={"smiles": ASPIRIN, "endpoint": "tox21", "task": "NOT-A-TASK"},
    )
    assert res.status_code == 422


def test_clintox_is_not_an_allowed_endpoint(client):
    res = client.post("/v1/explanations", json={"smiles": ASPIRIN, "endpoint": "clintox"})
    assert res.status_code == 422


def test_an_unknown_field_is_refused(client):
    res = client.post(
        "/v1/explanations",
        json={"smiles": ASPIRIN, "endpoint": "herg", "render": True},
    )
    assert res.status_code == 422


def test_herg_happy_path_shape(client):
    res = client.post("/v1/explanations", json={"smiles": ASPIRIN, "endpoint": "herg"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["endpoint"] == "herg"
    assert body["atom_order_version"] == "rdkit-output-order-v1"
    assert body["atoms"][0]["atom_index"] == 0
    assert "unmapped_importance" in body
    assert "heatmap" not in res.text and "base64" not in res.text
