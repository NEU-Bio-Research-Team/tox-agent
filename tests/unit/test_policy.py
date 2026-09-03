"""Threshold resolution: the artifact is the only default."""
import pytest

from toxpred.domain.endpoints import TOX21_TASKS
from toxpred.domain.policy import (
    PredictionPolicySnapshot,
    ResolvedThreshold,
    ThresholdSource,
    apply_threshold,
)

ARTIFACT_HERG = 0.4133453071117401
TOX21 = {t: 0.5 for t in TOX21_TASKS}


def make(**kw):
    return PredictionPolicySnapshot.from_artifact(
        herg_threshold=ARTIFACT_HERG, tox21_thresholds=TOX21, **kw
    )


def test_artifact_threshold_is_the_default():
    policy = make()
    assert policy.herg_threshold.value == ARTIFACT_HERG
    assert policy.herg_threshold.source is ThresholdSource.ARTIFACT


def test_override_is_labelled_as_an_override():
    policy = make(herg_override=0.30)
    assert policy.herg_threshold.value == 0.30
    assert policy.herg_threshold.source is ThresholdSource.REQUEST_OVERRIDE


def test_every_tox21_task_has_its_own_threshold():
    policy = make()
    assert set(policy.tox21_thresholds) == set(TOX21_TASKS)


def test_missing_tox21_threshold_is_rejected():
    with pytest.raises(ValueError, match="missing Tox21 thresholds"):
        PredictionPolicySnapshot.from_artifact(
            herg_threshold=ARTIFACT_HERG, tox21_thresholds={"NR-AR": 0.5}
        )


def test_unknown_tox21_override_is_rejected():
    with pytest.raises(ValueError, match="unknown Tox21 task"):
        make(tox21_override={"NOT-A-TASK": 0.5})


def test_threshold_must_be_a_probability():
    with pytest.raises(ValueError):
        ResolvedThreshold(1.5, ThresholdSource.ARTIFACT)


def test_boundary_is_inclusive():
    threshold = ResolvedThreshold(0.4133453071117401, ThresholdSource.ARTIFACT)
    assert apply_threshold(0.4133453071117401, threshold) is True
    assert apply_threshold(0.4133453071117400, threshold) is False
    assert apply_threshold(0.5, threshold) is True
    assert apply_threshold(0.0, threshold) is False


def test_probability_outside_unit_interval_is_rejected():
    threshold = ResolvedThreshold(0.5, ThresholdSource.ARTIFACT)
    with pytest.raises(ValueError):
        apply_threshold(1.2, threshold)


def test_snapshot_thresholds_are_not_mutable():
    policy = make()
    with pytest.raises(TypeError):
        policy.tox21_thresholds["NR-AR"] = ResolvedThreshold(0.1, ThresholdSource.ARTIFACT)


# --- calibrated vs declared operating points -------------------------------

def test_artifact_and_declared_thresholds_are_distinguishable():
    """A number someone chose must not look like a calibrated one.

    hERG ships 0.4133 fitted by Youden-J over 3-fold CV; ClinTox ships nothing,
    so its 0.35 is a policy choice and is labelled as such.
    """
    policy = PredictionPolicySnapshot.from_artifact(
        herg_threshold=ARTIFACT_HERG,
        tox21_thresholds=TOX21,
        clintox_threshold=0.35,
    )
    assert policy.herg_threshold.source is ThresholdSource.ARTIFACT
    assert policy.clintox_threshold.source is ThresholdSource.MANIFEST_DECLARED


def test_clintox_override_is_labelled_as_an_override():
    policy = PredictionPolicySnapshot.from_artifact(
        herg_threshold=ARTIFACT_HERG,
        tox21_thresholds=TOX21,
        clintox_threshold=0.35,
        clintox_override=0.5,
    )
    assert policy.clintox_threshold.value == 0.5
    assert policy.clintox_threshold.source is ThresholdSource.REQUEST_OVERRIDE


def test_absent_clintox_threshold_stays_absent():
    policy = PredictionPolicySnapshot.from_artifact(
        herg_threshold=ARTIFACT_HERG, tox21_thresholds=TOX21
    )
    assert policy.clintox_threshold is None
