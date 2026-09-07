"""Contract tests for the v1 API.

These run against the real registry and the real ChemBERTa artifact, so they
also pin the semantics — not just the shape — of what the service returns.
"""
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from toxpred.api.app import create_app  # noqa: E402
from toxpred.settings import Settings  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ARTIFACT_HERG_THRESHOLD = 0.4133453071117401


@pytest.fixture(scope="module")
def client():
    try:
        with TestClient(create_app(Settings.from_env())) as c:
            yield c
    except Exception as exc:  # artifact or torch unavailable in this environment
        pytest.skip(f"cannot start the app: {exc}")


# --- health and inventory --------------------------------------------------

def test_liveness_says_nothing_about_models(client):
    body = client.get("/health/live").json()
    assert body == {"status": "alive"}


def test_readiness_reports_served_endpoints(client):
    body = client.get("/health/ready").json()
    assert body["ready"] is True
    assert set(body["served_endpoints"]) == {"herg", "tox21"}


def test_models_inventory_lists_the_blocked_model_with_its_reason(client):
    body = client.get("/v1/models").json()
    by_id = {m["model_id"]: m for m in body["models"]}
    assert by_id["herg-tox21-chemberta-v1"]["loaded"] is True
    clintox = by_id["clintox-smilesgnn-v1"]
    assert clintox["loaded"] is False
    assert clintox["required"] is False
    assert "tokenizer.pkl" in clintox["blocked_reason"]


# --- prediction ------------------------------------------------------------

def test_prediction_shape_and_provenance(client):
    body = client.post("/v1/predictions", json={"smiles": ASPIRIN}).json()
    assert body["canonical_smiles"] == ASPIRIN
    assert set(body["predictions"]) == {"herg", "tox21"}

    herg = body["predictions"]["herg"]
    assert 0.0 <= herg["probability_blocker"] <= 1.0
    assert herg["threshold"] == pytest.approx(ARTIFACT_HERG_THRESHOLD)
    assert herg["threshold_source"] == "artifact"
    assert herg["label"] in {"blocker", "non_blocker"}

    tox21 = body["predictions"]["tox21"]
    assert tox21["task_order_version"] == "tox21-12task-v1"
    assert len(tox21["assays"]) == 12

    prov = body["provenance"]
    for field in ("request_id", "predictor_version", "policy_version", "artifacts"):
        assert field in prov
    artifact = prov["artifacts"][0]
    assert len(artifact["weights_sha256"]) == 64


def test_response_never_carries_a_clinical_field_for_herg(client):
    payload = json.dumps(client.post("/v1/predictions", json={"smiles": ASPIRIN}).json())
    assert "clinical" not in payload
    assert "p_toxic" not in payload


def test_no_aggregate_verdict(client):
    payload = json.dumps(client.post("/v1/predictions", json={"smiles": ASPIRIN}).json())
    for banned in ("final_verdict", "assay_hits", "mechanistic_alert"):
        assert banned not in payload


def test_endpoint_selection_is_honoured(client):
    body = client.post("/v1/predictions", json={"smiles": ASPIRIN, "endpoints": ["herg"]}).json()
    assert set(body["predictions"]) == {"herg"}


def test_unserved_endpoint_is_503_not_a_substitute(client):
    res = client.post("/v1/predictions", json={"smiles": ASPIRIN, "endpoints": ["clintox"]})
    assert res.status_code == 503
    assert res.json()["error"] == "model_not_ready"


# --- request validation ----------------------------------------------------

def test_unknown_request_field_is_rejected(client):
    """The legacy API silently dropped clinical_threshold. This one does not."""
    res = client.post("/v1/predictions", json={"smiles": ASPIRIN, "clinical_threshold": 0.3})
    assert res.status_code == 422


def test_invalid_smiles_is_a_typed_400_not_a_zero_probability(client):
    res = client.post("/v1/predictions", json={"smiles": "not_a_smiles"})
    assert res.status_code == 400
    body = res.json()
    assert body["error"] == "invalid_smiles"
    assert "predictions" not in body


def test_empty_smiles_is_rejected(client):
    assert client.post("/v1/predictions", json={"smiles": ""}).status_code == 422


def test_threshold_override_is_labelled(client):
    body = client.post(
        "/v1/predictions",
        json={"smiles": ASPIRIN, "endpoints": ["herg"],
              "threshold_overrides": {"herg": 0.2}},
    ).json()
    herg = body["predictions"]["herg"]
    assert herg["threshold"] == pytest.approx(0.2)
    assert herg["threshold_source"] == "request_override"


def test_unknown_tox21_task_override_is_rejected(client):
    res = client.post(
        "/v1/predictions",
        json={"smiles": ASPIRIN, "threshold_overrides": {"tox21": {"NOT-A-TASK": 0.5}}},
    )
    assert res.status_code == 422


# --- batch -----------------------------------------------------------------

def test_batch_preserves_order_and_reports_per_item_errors(client):
    body = client.post(
        "/v1/predictions:batch",
        json={"smiles": ["CCO", "not_a_smiles", ASPIRIN], "endpoints": ["herg"]},
    ).json()
    assert body["count"] == 3
    assert [r["input_smiles"] for r in body["results"]] == ["CCO", ASPIRIN]
    assert len(body["errors"]) == 1
    assert body["errors"][0]["index"] == 1
    assert body["errors"][0]["error"] == "invalid_smiles"


def test_batch_matches_single_prediction(client):
    single = client.post(
        "/v1/predictions", json={"smiles": ASPIRIN, "endpoints": ["herg"]}
    ).json()["predictions"]["herg"]["probability_blocker"]
    batch = client.post(
        "/v1/predictions:batch", json={"smiles": [ASPIRIN], "endpoints": ["herg"]}
    ).json()["results"][0]["predictions"]["herg"]["probability_blocker"]
    assert single == pytest.approx(batch, abs=1e-9)


# --- attribution -----------------------------------------------------------

def test_attribution_returns_numbers_not_an_image(client):
    body = client.post(
        "/v1/attributions", json={"smiles": ASPIRIN, "endpoint": "herg"}
    ).json()
    assert body["status"] == "completed"
    assert body["metadata"]["method"] == "grad_x_embedding_l2_v1"
    assert body["metadata"]["deterministic"] is True
    assert body["tokens"] and "importance" in body["tokens"][0]
    assert "heatmap_base64" not in json.dumps(body)


def test_tox21_attribution_requires_a_task(client):
    res = client.post("/v1/attributions", json={"smiles": ASPIRIN, "endpoint": "tox21"})
    assert res.status_code == 400
    assert "requires a task" in res.json()["message"]


def test_tox21_attribution_with_a_task_works(client):
    body = client.post(
        "/v1/attributions",
        json={"smiles": ASPIRIN, "endpoint": "tox21", "task": "NR-ER"},
    ).json()
    assert body["status"] == "completed"
    assert body["task"] == "NR-ER"


def test_attribution_probability_matches_the_prediction(client):
    predicted = client.post(
        "/v1/predictions", json={"smiles": ASPIRIN, "endpoints": ["herg"]}
    ).json()["predictions"]["herg"]["probability_blocker"]
    attributed = client.post(
        "/v1/attributions", json={"smiles": ASPIRIN, "endpoint": "herg"}
    ).json()["probability"]
    assert predicted == pytest.approx(attributed, abs=1e-6)


# --- OpenAPI ---------------------------------------------------------------

def test_openapi_exposes_only_the_prediction_surface(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {
        "/health/live", "/health/ready", "/v1/models",
        "/v1/predictions", "/v1/predictions:batch", "/v1/attributions",
        "/v1/explanations",
    }


def test_openapi_carries_no_agent_or_chat_schema(client):
    spec = json.dumps(client.get("/openapi.json").json()).lower()
    for banned in ("agent", "chat", "report_state", "adk", "session_id"):
        assert banned not in spec, f"{banned!r} leaked into the OpenAPI document"
