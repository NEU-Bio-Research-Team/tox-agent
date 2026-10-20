from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PREDICTOR = Path(__file__).resolve().parents[2]
if str(PREDICTOR / "research") not in sys.path:
    sys.path.insert(0, str(PREDICTOR / "research"))

from sair.baseline import (  # noqa: E402
    FEATURE_NAMES,
    BaselineInputError,
    deterministic_sample,
    featurize_pair,
    predict_with_baseline,
)


def test_pair_features_have_a_stable_finite_shape():
    vector = featurize_pair("CCO", "ACDEFGHIKLMNPQRSTVWY")
    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vector).all()
    assert sum(vector[-20:]) == pytest.approx(1.0)


def test_invalid_chemistry_and_sequence_fail_closed():
    with pytest.raises(BaselineInputError, match="invalid SMILES"):
        featurize_pair("not-smiles", "MPEPTIDE")
    with pytest.raises(BaselineInputError, match="no recognised amino acids"):
        featurize_pair("CCO", "XXXX")


def test_sampling_is_deterministic_and_not_row_order_dependent():
    frame = pd.DataFrame({"entry_id": range(20), "value": range(20)})
    first = deterministic_sample(frame, limit=5, salt="v1")
    second = deterministic_sample(frame.iloc[::-1], limit=5, salt="v1")
    assert first.entry_id.tolist() == second.entry_id.tolist()


def test_inference_returns_interval_and_never_hides_non_admission():
    class ConstantModel:
        def predict(self, matrix):
            return np.asarray([7.0] * len(matrix))

    artifact = {
        "model": ConstantModel(),
        "model_id": "fixture-b0",
        "admitted": False,
        "interval_90_half_width": 0.5,
        "feature_lower_005": np.full(len(FEATURE_NAMES), -1e9),
        "feature_upper_995": np.full(len(FEATURE_NAMES), 1e9),
    }
    result = predict_with_baseline(
        artifact, smiles="CCO", protein_sequence="ACDEFGHIKLMNPQRSTVWY"
    )
    assert result["predicted_pic50"] == 7.0
    assert result["prediction_interval"] == [6.5, 7.5]
    assert result["applicability"] == "in_domain"
    assert result["admitted"] is False
