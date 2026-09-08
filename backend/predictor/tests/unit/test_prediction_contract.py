"""The semantic repair this refactor exists for.

`backend/inference.py` (lines 851-853 at commit e6882b2) computed
`p_toxic = float(herg_probs[local_idx])` and emitted it under a `clinical` key
with a "clinical" threshold. These tests fail if that ever becomes expressible
again.
"""
import json

import pytest

from toxpred.domain.endpoints import TOX21_TASKS
from toxpred.domain.policy import PredictionPolicySnapshot, ThresholdSource
from toxpred.domain.prediction import (
    ApplicabilityAssessment,
    ClinToxPrediction,
    HergPrediction,
    PredictionResult,
    Tox21AssayPrediction,
    Tox21Prediction,
)

POLICY = PredictionPolicySnapshot.from_artifact(
    herg_threshold=0.4133453071117401,
    tox21_thresholds={t: 0.5 for t in TOX21_TASKS},
    clintox_threshold=0.35,
)


def herg(p=0.82):
    return HergPrediction(p, POLICY.herg_threshold, "herg-tox21-chemberta-v1")


def tox21(p=0.6):
    return Tox21Prediction(
        tuple(Tox21AssayPrediction(t, p, POLICY.tox21_thresholds[t]) for t in TOX21_TASKS),
        "herg-tox21-chemberta-v1",
    )


def result(**kw):
    return PredictionResult(
        input_smiles="CCO",
        canonical_smiles="CCO",
        applicability=ApplicabilityAssessment("ok", "element_rules_v1"),
        provenance={"request_id": "r"},
        **kw,
    )


# --- the invariant ---------------------------------------------------------

def test_herg_result_never_serialises_a_clinical_key():
    payload = json.dumps(result(herg=herg(), tox21=tox21()).to_dict())
    assert "clinical" not in payload
    assert "p_toxic" not in payload


def test_herg_probability_is_named_for_what_it_measures():
    assert herg().to_dict()["probability_blocker"] == pytest.approx(0.82)
    assert "probability_clinical_toxicity" not in herg().to_dict()


def test_herg_labels_are_blocker_not_toxic():
    assert herg(0.82).label == "blocker"
    assert herg(0.10).label == "non_blocker"
    assert herg(0.82).to_dict()["label"] == "blocker"


def test_clintox_is_a_separate_type_with_its_own_field():
    clintox = ClinToxPrediction(0.08, POLICY.clintox_threshold, "clintox-smilesgnn-v1")
    payload = clintox.to_dict()
    assert payload["probability_clinical_toxicity"] == pytest.approx(0.08)
    assert payload["label"] == "negative"


def test_clintox_slot_cannot_hold_a_herg_prediction():
    # PredictionResult keys its payload off the declared type of each slot, so a
    # hERG value placed in the clintox slot still cannot be read as clinical.
    payload = result(clintox=None, herg=herg()).to_dict()
    assert set(payload["predictions"]) == {"herg"}


def test_endpoints_absent_from_the_request_are_absent_from_the_payload():
    assert set(result(herg=herg()).to_dict()["predictions"]) == {"herg"}
    assert set(result(tox21=tox21()).to_dict()["predictions"]) == {"tox21"}


# --- no synthetic global verdict ------------------------------------------

def test_no_aggregate_verdict_field():
    payload = json.dumps(result(herg=herg(), tox21=tox21()).to_dict())
    for banned in ("final_verdict", "assay_hits", "mechanistic_alert", "overall"):
        assert banned not in payload


# --- threshold provenance --------------------------------------------------

def test_every_label_carries_its_threshold_and_source():
    payload = result(herg=herg(), tox21=tox21()).to_dict()["predictions"]
    assert payload["herg"]["threshold"] == pytest.approx(0.4133453071117401)
    assert payload["herg"]["threshold_source"] == ThresholdSource.ARTIFACT.value
    for assay in payload["tox21"]["assays"].values():
        assert "threshold" in assay and "threshold_source" in assay


def test_tox21_payload_declares_its_task_order_version():
    assert result(tox21=tox21()).to_dict()["predictions"]["tox21"][
        "task_order_version"
    ] == "tox21-12task-v1"


# --- value validation ------------------------------------------------------

def test_probabilities_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError):
        HergPrediction(1.4, POLICY.herg_threshold, "m")
    with pytest.raises(ValueError):
        ClinToxPrediction(-0.1, POLICY.clintox_threshold, "m")


def test_unknown_tox21_task_is_rejected():
    with pytest.raises(ValueError, match="unknown Tox21 task"):
        Tox21AssayPrediction("NOT-A-TASK", 0.5, POLICY.tox21_thresholds["NR-AR"])


def test_tox21_assays_must_arrive_in_the_frozen_order():
    shuffled = tuple(
        Tox21AssayPrediction(t, 0.5, POLICY.tox21_thresholds[t]) for t in reversed(TOX21_TASKS)
    )
    with pytest.raises(ValueError, match="order"):
        Tox21Prediction(shuffled, "m")


def test_applicability_method_is_carried_in_the_payload():
    payload = result(herg=herg()).to_dict()["applicability"]
    assert payload["method"] == "element_rules_v1"
    assert payload["status"] in ("ok", "limited", "out_of_domain")


def test_applicability_status_is_constrained():
    with pytest.raises(ValueError):
        ApplicabilityAssessment("in_distribution", "element_rules_v1")
