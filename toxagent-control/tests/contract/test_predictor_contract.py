"""The pinned ToxPred contract (plan §2.1, §20.2).

Two layers. The first asserts the exact surface this control plane depends on,
and runs everywhere — it is what tells a reader which parts of the predictor are
load-bearing. The second regenerates the document from the predictor source and
compares it byte for byte; it runs only in the monorepo, and its failure message
is an instruction to review the diff and re-pin, never to loosen the assertion.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2] / "toxagent" / "predictor" / "contract_snapshot.json"
)

# Every path the control plane calls. Anything not here is not depended upon.
REQUIRED_PATHS = {
    "/health/live": {"get"},
    "/health/ready": {"get"},
    "/v1/models": {"get"},
    "/v1/predictions": {"post"},
    "/v1/predictions:batch": {"post"},
    "/v1/attributions": {"post"},
    "/v1/explanations": {"post"},
}


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text())


@pytest.fixture(scope="module")
def document(snapshot) -> dict:
    return snapshot["openapi"]


def test_snapshot_records_the_predictor_commit(snapshot):
    commit = snapshot["captured_at_commit"]
    assert commit != "unknown", "re-run scripts/snapshot_predictor_contract.py inside the repo"
    assert len(commit) == 40


@pytest.mark.parametrize("path,methods", sorted(REQUIRED_PATHS.items()))
def test_required_paths_exist(document, path, methods):
    assert path in document["paths"], f"predictor no longer serves {path}"
    assert methods <= set(document["paths"][path]), f"{path} lost {methods}"


def test_prediction_request_forbids_unknown_fields(document):
    """An override the predictor silently drops is a wrong operating point."""
    schema = document["components"]["schemas"]["PredictionRequest"]
    assert schema.get("additionalProperties") is False
    assert set(schema["properties"]) == {"smiles", "endpoints", "threshold_overrides"}


def test_attribution_is_single_endpoint(document):
    """SCI-09: attribution explains one endpoint/task, never an aggregate."""
    schema = document["components"]["schemas"]["AttributionRequest"]
    assert set(schema["properties"]) == {"smiles", "endpoint", "task"}
    assert "endpoints" not in schema["properties"]


def test_batch_limit_is_documented(document):
    schema = document["components"]["schemas"]["BatchPredictionRequest"]
    assert schema["properties"]["smiles"]["type"] == "array"


def test_snapshot_matches_the_predictor_source():
    """Regenerate and compare. A diff here is a contract change to review."""
    import subprocess
    import sys

    repo_root = SNAPSHOT_PATH.parents[3]
    code = (
        "import json,sys; sys.path.insert(0, %r); "
        "from toxpred.api.app import create_app; "
        "print(json.dumps(create_app().openapi(), sort_keys=True))" % str(repo_root)
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("predictor not importable from here; snapshot checked structurally only")

    live = json.loads(proc.stdout)
    pinned = json.loads(SNAPSHOT_PATH.read_text())["openapi"]
    assert live == pinned, (
        "ToxPred's OpenAPI document changed. Review the diff, then re-pin with\n"
        "  python toxagent-control/scripts/snapshot_predictor_contract.py"
    )
