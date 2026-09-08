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

import toxagent.predictor

# Locate the snapshot through the package that owns it, not by counting
# directories up from this file: the src-layout move (I24) broke the count and
# the whole contract suite stopped at fixture setup instead of checking
# anything. Importing the package works in a checkout, a wheel and the image.
SNAPSHOT_PATH = Path(toxagent.predictor.__file__).resolve().parent / "contract_snapshot.json"
#: backend/control/src/toxagent/predictor/… → repository root.
_REPO_ROOT = SNAPSHOT_PATH.parents[5]
PREDICTOR_SRC = _REPO_ROOT / "backend" / "predictor" / "src"

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
    # ``model_selection`` was added by the predictor and only became visible
    # here once I24's path bug stopped skipping the regeneration check. It is
    # the field K04's binding travels in — an admitted model id per endpoint,
    # not a free-form provider hint.
    assert set(schema["properties"]) == {
        "smiles",
        "endpoints",
        "threshold_overrides",
        "model_selection",
    }
    assert schema["properties"]["model_selection"]["anyOf"][0]["propertyNames"]["enum"] == [
        "clintox",
        "herg",
        "tox21",
    ]


def test_attribution_is_single_endpoint(document):
    """SCI-09: attribution explains one endpoint/task, never an aggregate."""
    schema = document["components"]["schemas"]["AttributionRequest"]
    assert set(schema["properties"]) == {"smiles", "endpoint", "task", "method"}
    assert "endpoints" not in schema["properties"]
    assert schema["properties"]["method"]["enum"] == [
        "grad_x_input",
        "integrated_gradients",
    ]


def test_batch_limit_is_documented(document):
    schema = document["components"]["schemas"]["BatchPredictionRequest"]
    assert schema["properties"]["smiles"]["type"] == "array"


def test_snapshot_matches_the_predictor_source():
    """Regenerate and compare. A diff here is a contract change to review."""
    import subprocess
    import sys

    # Only the monorepo has the predictor source. Skip when the directory is
    # absent (a wheel install, the control image); never skip because an
    # import merely failed — that is the contract drift this test exists to
    # catch, and swallowing it is how the suite came to prove nothing.
    if not PREDICTOR_SRC.is_dir():
        pytest.skip(f"predictor source not in this install: {PREDICTOR_SRC}")

    code = (
        "import json,sys; sys.path.insert(0, %r); "
        "from toxpred.api.app import create_app; "
        "print(json.dumps(create_app().openapi(), sort_keys=True))" % str(PREDICTOR_SRC)
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, (
        "the predictor source is present but its app would not build, so the "
        "pinned contract cannot be verified:\n" + proc.stderr[-4000:]
    )

    live = json.loads(proc.stdout)
    pinned = json.loads(SNAPSHOT_PATH.read_text())["openapi"]
    assert live == pinned, (
        "ToxPred's OpenAPI document changed. Review the diff, then re-pin with\n"
        "  python backend/control/scripts/snapshot_predictor_contract.py"
    )
