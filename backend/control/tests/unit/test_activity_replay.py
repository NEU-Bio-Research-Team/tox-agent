"""The replay store and what it projects (#44), without a database.

A replayed screening must read exactly as a served one would: the same
canonical payload, the same content hash, the same required limitations. And
the file it comes from must be the one the deployment pinned, holding records
nobody edited after they were frozen.
"""
from __future__ import annotations

import json

import pytest

from tests.support.activity import NOW, frozen_screening, replay_file, screening_id
from toxagent.application.activity import replay
from toxagent.application.activity.projections import ACTIVITY_SCHEMA_VERSION
from toxagent.domain.ids import RUN, SESSION, new_id
from toxagent.domain.observation import ObservationKind, Producer


def _load(path, sha):
    return replay.FrozenReplayStore.load(path, sha256=sha)


def test_a_pinned_file_loads_its_screenings(tmp_path):
    record = frozen_screening()
    path, sha = replay_file(tmp_path, record)
    store = _load(path, sha)
    assert len(store) == 1
    assert store.get(screening_id(record))["content_sha256"] == record["content_sha256"]
    assert store.describe() == {"path_name": "screenings.json", "sha256": sha, "screenings": 1}


def test_a_file_that_is_not_the_pinned_one_is_refused(tmp_path):
    path, _ = replay_file(tmp_path, frozen_screening())
    with pytest.raises(replay.ReplayError, match="not the pinned"):
        _load(path, "0" * 64)
    with pytest.raises(replay.ReplayError, match="pinned by its sha256"):
        _load(path, "")


def test_a_record_edited_after_it_was_frozen_is_refused(tmp_path):
    """Re-pinning the file is not enough: each record carries its own hash."""
    record = frozen_screening()
    record["activity_estimate"]["estimate"]["predicted_pic50"] = 9.9
    path, sha = replay_file(tmp_path, record)
    with pytest.raises(replay.ReplayError, match="does not match its content_sha256"):
        _load(path, sha)


def test_the_same_screening_twice_is_refused(tmp_path):
    record = frozen_screening()
    path, sha = replay_file(tmp_path, record, record)
    with pytest.raises(replay.ReplayError, match="appears twice"):
        _load(path, sha)


def test_a_file_of_another_kind_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema_version": "something-else", "screenings": []}))
    import hashlib

    with pytest.raises(replay.ReplayError, match="activity-replay-v1"):
        _load(path, hashlib.sha256(path.read_bytes()).hexdigest())


def _observe(record, artifact):
    return replay.observation_for(
        record, artifact, session_id=new_id(SESSION), run_id=new_id(RUN), now=NOW,
    )


def test_a_replayed_estimate_is_the_estimate_a_live_run_would_cite():
    record = frozen_screening()
    observation = _observe(record, "activity_estimate")
    assert observation.kind is ObservationKind.ACTIVITY_ESTIMATE
    assert observation.producer is Producer.PREDICTOR
    assert observation.schema_version == ACTIVITY_SCHEMA_VERSION
    assert observation.canonical_payload == record["activity_estimate"]
    assert observation.value_at("estimate.predicted_pic50") == pytest.approx(7.1)
    # the provenance names the frozen record it came from
    assert observation.provenance["replayed_from"]["record_sha256"] == record["content_sha256"]
    assert set(observation.required_limitations) == {
        "activity_interval_is_marginal_coverage", "pic50_not_therapeutic_efficacy",
        "activity_model_not_admitted",
    }


def test_two_replays_of_one_record_hash_the_same():
    record = frozen_screening()
    one, other = _observe(record, "activity_estimate"), _observe(record, "activity_estimate")
    assert one.id != other.id
    assert one.content_sha256 == other.content_sha256


def test_the_decision_and_the_stage_keep_their_own_limitations():
    record = frozen_screening()
    assert _observe(record, "prioritisation_decision").required_limitations == (
        "policy_decision_not_a_measurement",
    )
    state = _observe(record, "screening_state")
    assert state.required_limitations == ("toxicity_not_run_not_negative",)
    assert state.value_at("toxicity.run") is False


def test_a_stage_that_never_ran_has_no_artifact_to_replay():
    record = frozen_screening(activity_fails=True)
    assert _observe(record, "activity_estimate") is None
    assert _observe(record, "prioritisation_decision") is None
    assert _observe(record, "screening_state").value_at("status") == "activity_failed"


def test_the_summary_names_the_subject_and_not_the_numbers():
    summary = replay.summary(frozen_screening())
    assert summary["target"]["target_id"] == "EGFR"
    assert summary["toxicity_run"] is False
    assert summary["model_admitted"] is False
    assert "predicted_pic50" not in json.dumps(summary)
