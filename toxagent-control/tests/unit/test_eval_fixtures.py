"""Frozen fixtures load, hash-check, and serve the pinned ToxPred shape."""
from __future__ import annotations

import json

import pytest

from evals.build_fixtures import main as build_fixtures_main
from evals.frozen import ASPIRIN, FIXTURES_DIR, FrozenPredictor, fixture_digest, load_fixture

FIXTURE_NAMES = sorted(p.stem for p in FIXTURES_DIR.glob("*.json"))


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_fixture_loads_with_a_matching_hash(name):
    fixture = load_fixture(name)  # raises on hash drift
    assert fixture["content_sha256"] == fixture_digest(fixture)


def test_build_fixtures_is_idempotent():
    before = {p.name: p.read_text() for p in sorted(FIXTURES_DIR.glob("*.json"))}
    build_fixtures_main()
    after = {p.name: p.read_text() for p in sorted(FIXTURES_DIR.glob("*.json"))}
    assert before == after


@pytest.mark.anyio
async def test_frozen_predictor_serves_the_pinned_prediction_shape():
    predictor = FrozenPredictor(load_fixture("aspirin-herg-tox21")["predictor"])
    result = await predictor.client().predict(ASPIRIN, endpoints=("herg", "tox21"))
    assert result.predictions.herg.probability_blocker == 0.281
    assert result.predictions.tox21 is not None
    assert result.applicability.method == "element_rules_v1"
    assert result.raw["predictions"]["herg"]["threshold_source"] == "model_default"


@pytest.mark.anyio
async def test_frozen_predictor_does_not_substitute_an_unserved_endpoint():
    from toxagent.domain.errors import EndpointUnavailable

    predictor = FrozenPredictor(load_fixture("clintox-unavailable")["predictor"])
    # The frozen transport drops clintox from the body; the client then refuses
    # to pretend it was served (SCI-06) rather than returning a substitute.
    with pytest.raises(EndpointUnavailable):
        await predictor.client().predict(
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C", endpoints=("herg", "tox21", "clintox")
        )


@pytest.mark.anyio
async def test_frozen_predictor_rejects_an_unknown_smiles():
    from toxagent.domain.errors import InvalidSmiles

    predictor = FrozenPredictor(load_fixture("aspirin-herg-tox21")["predictor"])
    with pytest.raises(InvalidSmiles):
        await predictor.client().predict("not a molecule", endpoints=("herg",))


@pytest.mark.anyio
async def test_frozen_503_fixture_is_a_predictor_outage():
    from toxagent.domain.errors import PredictorNotReady

    predictor = FrozenPredictor(load_fixture("predictor-503")["predictor"])
    with pytest.raises(PredictorNotReady):
        await predictor.client().predict(ASPIRIN, endpoints=("herg",))
