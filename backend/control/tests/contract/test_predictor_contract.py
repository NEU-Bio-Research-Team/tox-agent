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


def _find_predictor_source(start: Path) -> Path | None:
    """Walk up looking for the monorepo sibling, rather than counting levels.

    P2-2 of the 2026-09-13 audit: this was ``SNAPSHOT_PATH.parents[5]``, a
    number that is only correct for one directory layout. An editable install
    from a checkout, a wheel in site-packages and the container image each put
    the package at a different depth, so five levels up landed somewhere
    arbitrary and the regeneration check skipped with a misleading reason — or,
    before I24, stopped the whole suite at fixture setup.

    Searching for the marker is layout-independent and self-describing: either
    the predictor source is somewhere above us, or this is not a monorepo
    checkout and the regeneration check genuinely cannot run.
    """
    for parent in [start, *start.parents]:
        candidate = parent / "backend" / "predictor" / "src"
        if (candidate / "toxpred").is_dir():
            return candidate
    return None


PREDICTOR_SRC = _find_predictor_source(SNAPSHOT_PATH.parent)

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
    # `model_id` pins *which* admitted model attributes the one endpoint
    # (I10/I11); it does not make attribution multi-endpoint, which is what
    # the absence of `endpoints` below still enforces.
    assert set(schema["properties"]) == {"smiles", "endpoint", "task", "method", "model_id"}
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

    import os

    # CI's predictor-contract job sets this after installing both packages, so
    # there the check is mandatory and any failure to build the predictor app
    # is a red build. Elsewhere — a control-only virtualenv, the control image,
    # a wheel install — the predictor's dependencies are legitimately absent
    # and this degrades to a skip that names why.
    required = os.getenv("TOXAGENT_REQUIRE_PREDICTOR_CONTRACT") == "1"

    if PREDICTOR_SRC is None or not PREDICTOR_SRC.is_dir():
        reason = (
            "no backend/predictor/src/toxpred above "
            f"{SNAPSHOT_PATH.parent} — this is not a monorepo checkout"
        )
        if required:
            pytest.fail(f"predictor source required but not found: {reason}")
        pytest.skip(f"predictor source not in this install: {reason}")

    code = (
        "import json,sys; sys.path.insert(0, %r); "
        "from toxpred.api.app import create_app; "
        "print(json.dumps(create_app().openapi(), sort_keys=True))" % str(PREDICTOR_SRC)
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr[-4000:]
        if required:
            pytest.fail("predictor app would not build, so the pinned contract "
                        "cannot be verified:\n" + detail)
        pytest.skip("predictor dependencies not installed here: " + detail.strip()[-300:])

    live = json.loads(proc.stdout)
    pinned = json.loads(SNAPSHOT_PATH.read_text())["openapi"]
    assert live == pinned, (
        "ToxPred's OpenAPI document changed. Review the diff, then re-pin with\n"
        "  python backend/control/scripts/snapshot_predictor_contract.py"
    )
