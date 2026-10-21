from __future__ import annotations

import json
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
    LABEL_FILTER_SPLITS,
    BaselineInputError,
    deterministic_sample,
    featurize_pair,
    label_filter_report,
    partition_splits,
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


def _pair_table() -> pd.DataFrame:
    """Three splits where every split holds one high-span pair."""
    rows = []
    for split in ("train", "validation", "test"):
        for index in range(4):
            rows.append(
                {
                    "entry_id": f"{split}-{index}",
                    "split": split,
                    # one contradictory label per split
                    "training_eligible": index != 0,
                    "assay_primary": "IC50" if index % 2 else "Ki",
                    "pic50_median": 6.0 + index,
                }
            )
    return pd.DataFrame(rows)


def test_the_training_label_filter_never_reaches_the_test_split():
    """training_eligible excludes contradictory labels from the fit. Applying
    it to test would score the model on the easy part of SAIR and report it as
    the whole distribution."""
    partitions = partition_splits(_pair_table())

    assert "test" not in LABEL_FILTER_SPLITS
    assert partitions["test"]["training_eligible"].tolist() == [False, True, True, True]
    assert len(partitions["test"]) == 4
    for split in LABEL_FILTER_SPLITS:
        assert partitions[split]["training_eligible"].all()
        assert len(partitions[split]) == 3


def test_the_label_filter_report_states_what_was_dropped_and_where():
    report = label_filter_report(_pair_table())

    assert report["filtered_splits"] == list(LABEL_FILTER_SPLITS)
    assert report["per_split"]["test"]["filter_applied"] is False
    assert report["per_split"]["test"]["excluded_rows"] == 1
    assert report["per_split"]["train"]["filter_applied"] is True
    assert report["per_split"]["train"]["excluded_fraction"] == pytest.approx(0.25)
    # exclusion is reported per assay, not only as one pooled number
    assert report["by_primary_assay"]["Ki"]["excluded_rows"] == 3
    assert report["by_primary_assay"]["IC50"]["excluded_rows"] == 0


def test_train_baseline_scores_the_full_test_set_and_keeps_the_filtered_view(tmp_path):
    """The test split keeps every row; train and validation keep only the rows
    whose replicate labels agree."""
    from sair.baseline import _file_sha256, train_baseline

    smiles = ["CCO", "CCN", "CCC", "CCCC", "c1ccccc1", "CCOCC"]
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "ACDEFGHIKLMNPQRSTVW",
        "MKTAYIAKQRQISFVKSHFSRQ",
        "MKTAYIAKQRQISFVKSHF",
        "ACDEFGHIKLMNPQRST",
        "MNPQRSTVWYACDEFGHIKL",
    ]
    rows = []
    for split, count in (("train", 60), ("validation", 24), ("test", 24)):
        for index in range(count):
            # a handful of contradictory labels in every split, test included
            contradictory = index % 8 == 0
            rows.append(
                {
                    "entry_id": f"{split}-{index}",
                    "source_smiles": smiles[index % len(smiles)],
                    "protein_sequence": sequences[(index // 2) % len(sequences)],
                    "pic50_median": 5.0 + 0.3 * (index % 11),
                    "training_eligible": not contradictory,
                    "assay_primary": "IC50" if index % 3 else "Ki",
                    "split": split,
                }
            )
    phase1_path = tmp_path / "phase1.parquet"
    pd.DataFrame(rows).to_parquet(phase1_path, index=False)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "output": {"sha256": _file_sha256(phase1_path)},
                "source": {"sha256": "synthetic-source"},
                "split_policy": {"name": "exact_sequence_cold_target"},
            }
        )
    )

    report = train_baseline(
        phase1_path=phase1_path,
        phase1_manifest_path=manifest_path,
        artifact_path=tmp_path / "b0.joblib",
        report_path=tmp_path / "b0.json",
        max_train=60,
        max_validation=24,
        max_test=24,
        sample_salt="unit-test",
    )
    counts = report["sample"]["counts"]
    per_split = report["dataset"]["label_filter"]["per_split"]

    # test is scored on every row it has, contradictory labels included
    assert per_split["test"]["excluded_rows"] > 0
    assert per_split["test"]["filter_applied"] is False
    assert counts["test"] == per_split["test"]["rows"]

    # train and validation are scored on the eligible rows only
    for split in LABEL_FILTER_SPLITS:
        assert per_split[split]["filter_applied"] is True
        assert counts[split] == per_split[split]["training_eligible_rows"]
        assert counts[split] < per_split[split]["rows"]

    # the old, fully filtered test view stays reported beside the full one
    filtered = report["metrics"]["test_training_eligible_only"]
    assert filtered["rows"] == per_split["test"]["training_eligible_rows"]
    assert filtered["excluded_rows"] == per_split["test"]["excluded_rows"]
    assert "mae" in filtered and "mae" in report["metrics"]["test"]
    assert report["admitted"] is False
