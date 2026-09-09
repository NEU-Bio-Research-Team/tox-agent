"""``snapshot_from_prediction`` — the construction shared by the persisted Lane D
path and the stateless Quick Predict path (plan section 3.3).

It must be a pure move: same object, same content hash, same idempotency key as
``AnalysisSnapshot.create`` produced when the call lived inside
``CreateAnalysis``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from toxagent.domain.analysis import (
    AnalysisSnapshot,
    PredictorProvenance,
    snapshot_from_prediction,
)
from toxagent.domain.ids import new_id
from tests.support.predictor import ASPIRIN, prediction

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _args():
    return dict(
        session_id=new_id("ses"),
        run_id=new_id("run"),
        input_smiles=ASPIRIN,
        requested_endpoints=("herg", "tox21"),
        predictor_response=prediction(endpoints=("herg", "tox21")),
        provenance=PredictorProvenance(base_url_id="toxpred-local", artifact_hashes=("a",)),
        policy_snapshot={"requested_endpoints": ["herg", "tox21"]},
        now=NOW,
    )


def test_it_builds_an_analysis_snapshot_losslessly():
    args = _args()
    snapshot = snapshot_from_prediction(**args)

    assert isinstance(snapshot, AnalysisSnapshot)
    assert snapshot.canonical_smiles == ASPIRIN
    assert snapshot.served_endpoints == ("herg", "tox21")
    # The predictor payload is stored exactly as handed in.
    assert snapshot.predictor_response == args["predictor_response"]


def test_it_matches_analysis_snapshot_create():
    args = _args()
    direct = AnalysisSnapshot.create(**args)
    factored = snapshot_from_prediction(**args)

    assert factored.content_sha256 == direct.content_sha256
    assert factored.idempotency_key == direct.idempotency_key
    assert factored.requested_endpoints == direct.requested_endpoints
    assert factored.provenance == direct.provenance
