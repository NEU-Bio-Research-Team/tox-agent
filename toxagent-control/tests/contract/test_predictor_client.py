"""The predictor client's half of the contract (plan sections 2.1, 20.2).

What is asserted here is mostly about failure: which HTTP outcome becomes which
typed product error, and which responses are refused rather than stored.
"""
from __future__ import annotations

import httpx
import pytest

from toxagent.config import PredictorSettings
from toxagent.domain.errors import (
    EndpointUnavailable,
    InvalidSmiles,
    PredictorNotReady,
    PredictorProtocolError,
)
from toxagent.predictor.client import PredictorClient
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio


async def test_a_prediction_round_trips_with_its_provenance():
    stub = StubPredictor()
    client = stub.client()
    response = await client.predict(ASPIRIN, ("herg", "tox21"))
    assert response.served_endpoints() == ("herg", "tox21")
    assert response.predictions.herg.probability_blocker == 0.73064

    provenance = client.provenance_of(response)
    assert provenance.git_commit == "562b988de9714106fd842bb503072cfe8cd2852a"
    # audit_5_9.md A14: predictor_version (not service_version) is the real
    # field, and artifacts is a list of dicts, not a flat mapping — the
    # client must resolve both, not just echo str(dict) for each artifact.
    assert provenance.service_version == "0.1.0.dev0"
    assert provenance.artifact_hashes == (
        "herg-tox21-chemberta-v1:tokenizer_sha256="
        "ba6a21b7958b8aebf1f3ac341a883c430ae9906cba797b4f186ac79dcd00d785",
        "herg-tox21-chemberta-v1:weights_sha256="
        "c851e81541f8975f66589879ba9bd35c3068c3fbd57417bb7939214183f62690",
    )
    assert provenance.base_url_id == "toxpred-local"
    await client.aclose()


async def test_an_unparseable_molecule_is_a_validation_error_not_a_prediction():
    """SCI-08. The failure mode this replaces answered with p_toxic = 0.0."""
    client = StubPredictor().client()
    with pytest.raises(InvalidSmiles) as raised:
        await client.predict("not-a-molecule")
    assert raised.value.retryable is False
    assert raised.value.detail["smiles"] == "not-a-molecule"
    await client.aclose()


async def test_an_endpoint_this_build_does_not_serve_fails_loudly():
    """SCI-06. Nothing is substituted for ClinTox."""
    client = StubPredictor(served=("herg", "tox21")).client()
    with pytest.raises(EndpointUnavailable) as raised:
        await client.predict(ASPIRIN, ("herg", "clintox"))
    assert raised.value.detail["unavailable"] == ["clintox"]
    assert raised.value.detail["served"] == ["herg"]
    await client.aclose()


async def test_a_503_naming_the_unserved_endpoint_is_unavailable_not_not_ready():
    """Live sweep (2026-09-05, progress log §3.13): the real ToxPred fails
    the *whole* /v1/predictions call with 503 model_not_ready when any
    requested endpoint is permanently unserved (toxpred/application/predictor.py),
    not a 200 with a partial served_endpoints list the way StubPredictor's
    default /v1/predictions handler simulates it. That 503 was being
    classified as PredictorNotReady (retryable — implying trying again could
    help) for a build limitation retrying can never fix. The body here is
    the exact shape captured from a live predictor."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "error": "model_not_ready",
                "message": "endpoint 'clintox' is not served by this build "
                "(available: ['herg', 'tox21'])",
            },
        )

    client = PredictorClient(
        PredictorSettings(base_url="http://predictor.test"), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(EndpointUnavailable) as raised:
        await client.predict(ASPIRIN, ("herg", "tox21", "clintox"))
    assert raised.value.detail["requested"] == ["clintox"]
    await client.aclose()


async def test_a_missing_artifact_is_retryable_not_a_protocol_error():
    client = StubPredictor(fail_with=503).client()
    with pytest.raises(PredictorNotReady) as raised:
        await client.predict(ASPIRIN)
    assert raised.value.retryable is True
    await client.aclose()


async def test_an_unreachable_predictor_is_reported_as_not_ready():
    """A refused connection is retryable; it is not a contract violation."""
    from toxagent.config import PredictorSettings
    from toxagent.predictor.client import PredictorClient

    client = PredictorClient(
        PredictorSettings(base_url="http://127.0.0.1:1", connect_timeout_s=0.5)
    )
    with pytest.raises(PredictorNotReady):
        await client.predict(ASPIRIN)
    await client.aclose()


async def test_a_response_that_does_not_match_the_contract_is_refused():
    client = StubPredictor(malformed=True).client()
    with pytest.raises(PredictorProtocolError, match="pinned contract"):
        await client.predict(ASPIRIN)
    await client.aclose()


async def test_a_moved_tox21_task_order_is_refused():
    """SCI-01/SCI-05: if the column-to-assay mapping moved, no answer built on
    it can be trusted, and it cannot be resolved at runtime."""
    stub = StubPredictor()
    client = stub.client()

    import httpx

    from tests.support import predictor as fixtures

    def handler(request: httpx.Request) -> httpx.Response:
        body = fixtures.prediction(ASPIRIN)
        body["predictions"]["tox21"]["task_order_version"] = "tox21-12task-v2"
        return httpx.Response(200, json=body)

    client._client = httpx.AsyncClient(
        base_url="http://predictor.test", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(PredictorProtocolError, match="task order version"):
        await client.predict(ASPIRIN)
    await client.aclose()


async def test_batch_keeps_order_and_reports_per_item_errors():
    client = StubPredictor().client()
    result = await client.predict_batch([ASPIRIN, "not-a-molecule", "CCO"])
    assert result.count == 3
    assert [r.input_smiles for r in result.results] == [ASPIRIN, "CCO"]
    assert [e.index for e in result.errors] == [1]
    await client.aclose()


async def test_a_batch_over_the_documented_limit_never_leaves_the_process():
    client = StubPredictor().client(max_batch_size=2)
    with pytest.raises(PredictorProtocolError, match="documented maximum"):
        await client.predict_batch(["CCO"] * 3)
    await client.aclose()


async def test_a_tox21_attribution_must_name_its_assay():
    client = StubPredictor().client()
    result = await client.attribution(ASPIRIN, "tox21", "SR-p53")
    assert result.task == "SR-p53"
    assert result.metadata["method"] == "integrated_gradients_v1"
    await client.aclose()


async def test_readiness_is_reported_not_inferred():
    client = StubPredictor(ready=False).client()
    with pytest.raises(PredictorNotReady):
        await client.ready()
    await client.aclose()
