from datetime import datetime, timezone

import pytest

from toxpred.domain.endpoints import TOX21_TASKS
from toxpred.scientific.artifacts import ArtifactError
from toxpred.scientific.calibration import (
    CalibrationArtifact,
    PlattCalibrator,
    TemperatureCalibrator,
)
from toxpred.scientific.providers.contracts import (
    HergTox21RawOutput,
    ProviderBatchResult,
    TokenizationProvenance,
)
from toxpred.scientific.uncertainty import fit_binary_conformal, prediction_set


def row(**overrides):
    values = {
        "model_id": "m1",
        "herg_probability_blocker": 0.25,
        "tox21_probability_activity": {task: 0.5 for task in TOX21_TASKS},
        "tokenization": TokenizationProvenance(7, 7, 128, False),
    }
    values.update(overrides)
    return HergTox21RawOutput(**values)


def test_provider_batch_keeps_owner_and_is_immutable():
    batch = ProviderBatchResult("m1", (row(),))
    assert batch.provider_id == "m1"
    assert batch[0].model_id == "m1"
    with pytest.raises(Exception):
        batch.rows += (row(),)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_provider_probability_validation_fails_loud(value):
    with pytest.raises(ArtifactError):
        row(herg_probability_blocker=value)


def test_token_provenance_is_per_sample_not_batch_padding():
    short = TokenizationProvenance(8, 8, 128, False)
    long = TokenizationProvenance(200, 128, 128, True)
    assert short.truncated is False and long.truncated is True
    with pytest.raises(ValueError):
        TokenizationProvenance(8, 8, 128, True)


def test_calibration_is_distinct_from_threshold_policy():
    assert TemperatureCalibrator(2).calibrate(0.8) != pytest.approx(0.8)
    assert 0 < PlattCalibrator(1.2, -0.1).calibrate(0.8) < 1
    artifact = CalibrationArtifact(
        "toxpred-calibration-v1", "m1", "herg", "temperature", {"temperature": 2},
        "a" * 64, datetime.now(timezone.utc), {"ece": 0.2}, {"ece": 0.1},
    )
    assert len(artifact.sha256) == 64


def test_split_conformal_contract_reports_explicit_set():
    artifact = fit_binary_conformal(
        [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], endpoint="herg", task=None,
        target_coverage=0.9, calibration_split_sha256="b" * 64,
    )
    assert artifact.target_coverage == 0.9
    assert set(prediction_set(0.5, artifact)) <= {"negative", "positive"}
