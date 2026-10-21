"""Protocol v2 cases, inputs and gold, and the audit that holds them apart (#51).

The checked-in technical seed set audits clean; each rule is then broken on a
copy to show the audit names it: a changed inputs file, a gold statement
leaked into what a system is given, a span the frozen text does not hold, a
turn that switches to a subject the inputs never gave, a lone T3 polarity.
Dependency groups decide the split: no compound or document crosses it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evals.investigation import case_inputs as ci
from toxagent.application.activity.replay import FrozenReplayStore

SEED = Path(ci.__file__).resolve().parent / "cases_protocol_v2_technical"


def _stores(root: Path) -> dict:
    return ci.stores_from_files(replay_path=root / "replay.json",
                                snapshot_path=root / "evidence.json")


@pytest.fixture
def seed(tmp_path) -> Path:
    target = tmp_path / "seed"
    shutil.copytree(SEED, target)
    return target


def _codes(root: Path) -> set[str]:
    loaded = ci.load_set(root / "cases", with_gold=True)
    return {f.code for f in ci.audit_set(loaded, **_stores(root))}


def _rewrite(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=2))


def _repin(root: Path, case_id: str) -> None:
    case_path = root / "cases" / f"{case_id}.json"
    case = json.loads(case_path.read_text())
    case["inputs_ref"]["sha256"] = ci.file_sha256(root / "inputs" / f"{case_id}.json")
    case["gold_ref"]["sha256"] = ci.file_sha256(root / "gold" / f"{case_id}.json")
    case_path.write_text(json.dumps(case, indent=2))


def test_the_checked_in_seed_set_audits_clean_and_covers_every_group():
    loaded = ci.load_set(SEED / "cases", with_gold=True)
    assert len(loaded) == 12
    assert ci.audit_set(loaded, **_stores(SEED)) == []
    groups = {(c.case["task_group"], c.case["workflow_context"]) for c in loaded}
    assert len(groups) == 12
    assert all(c.case["provenance"]["kind"] == "synthetic_engineering_fixture" for c in loaded)
    assert all(c.gold["status"] == "synthetic_engineering_gold" for c in loaded)
    polarity = {c.gold["predictor_error"]["actually_wrong"] for c in loaded
                if c.case["task_group"] == "T3"}
    assert polarity == {True, False}


def test_the_seed_replay_file_loads_with_its_frozen_analyses():
    manifest = json.loads((SEED / "seed-manifest.json").read_text())
    store = FrozenReplayStore.load(SEED / "replay.json", sha256=manifest["replay"]["sha256"])
    assert store.describe()["screenings"] == 3 and store.describe()["analyses"] == 2


def test_a_changed_inputs_file_is_caught(seed):
    _rewrite(seed / "inputs" / "inv-a01-screening-review-explained.json",
             lambda d: d["turns"][0].update(text="A different question."))
    assert "inputs_hash_mismatch" in _codes(seed)


def test_gold_leaking_into_the_inputs_is_caught(seed):
    case_id = "inv-a01-screening-review-explained"
    gold = json.loads((seed / "gold" / f"{case_id}.json").read_text())
    _rewrite(seed / "inputs" / f"{case_id}.json",
             lambda d: d["turns"][0].update(text=gold["acceptable_conclusions"][0]["statement"]))
    _repin(seed, case_id)
    assert "gold_leaked_into_inputs" in _codes(seed)


def test_a_span_the_frozen_text_does_not_hold_is_caught(seed):
    case_id = "inv-a02-egfr-ic50-from-table"
    _rewrite(seed / "gold" / f"{case_id}.json",
             lambda d: d["evidence_spans"][0].update(locator="table-wrap:1/row:9"))
    _repin(seed, case_id)
    assert "span_locator_not_in_text" in _codes(seed)


def test_a_span_reference_no_run_could_retrieve_is_caught(seed):
    """A gold span names its source the way a run retrieves one (`pmid:…`).
    A case-local name is dropped from every retrieval denominator, so a set
    that ships one would measure nothing and say nothing about it."""
    case_id = "inv-a02-egfr-ic50-from-table"
    _rewrite(seed / "gold" / f"{case_id}.json",
             lambda d: d["evidence_spans"][0].update(evidence_ref="MED:90001"))
    _repin(seed, case_id)
    codes = _codes(seed)
    assert "span_ref_not_retrievable" in codes
    # It still resolves to the record, so the text check is not confused by it.
    assert "span_text_not_frozen" not in codes and "span_locator_not_in_text" not in codes


def test_a_span_reference_outside_the_corpus_is_caught(seed):
    case_id = "inv-a02-egfr-ic50-from-table"
    _rewrite(seed / "gold" / f"{case_id}.json",
             lambda d: d["evidence_spans"][0].update(evidence_ref="pmid:99999"))
    _repin(seed, case_id)
    assert "span_ref_not_in_corpus" in _codes(seed)


def test_switching_to_a_subject_the_inputs_never_gave_is_caught(seed):
    case_id = "inv-a05-update-then-switch-target"
    _rewrite(seed / "inputs" / f"{case_id}.json",
             lambda d: d["turns"][2]["actions"][0]["subject_ref"].update(
                 screening_id="scr_" + "f" * 32))
    _repin(seed, case_id)
    assert "subject_not_an_input_artifact" in _codes(seed)


def test_a_set_with_one_t3_polarity_is_caught(seed):
    case_id = "inv-t03-predictor-missed-a-blocker"
    _rewrite(seed / "gold" / f"{case_id}.json",
             lambda d: d["predictor_error"].update(actually_wrong=False))
    _repin(seed, case_id)
    assert "t3_polarity_not_balanced" in _codes(seed)


def test_an_artifact_hash_that_disagrees_with_the_replay_file_is_caught(seed):
    case_id = "inv-a01-screening-review-explained"
    _rewrite(seed / "inputs" / f"{case_id}.json",
             lambda d: d["artifacts"][0].update(content_sha256="0" * 64))
    _repin(seed, case_id)
    assert "artifact_hash_mismatch" in _codes(seed)


def test_shared_dependencies_keep_cases_on_one_side_of_the_split():
    cases = [
        {"case_id": "a", "dependency_groups": {"compounds": ["X"]}},
        {"case_id": "b", "dependency_groups": {"compounds": ["X"], "documents": ["D1"]}},
        {"case_id": "c", "dependency_groups": {"documents": ["D1"]}},
        {"case_id": "d", "dependency_groups": {"compounds": ["Y"]}},
        {"case_id": "e", "dependency_groups": {"compounds": ["Z"]},
         "paired_with": {"case_id": "d", "varies": "target_or_context"}},
    ]
    assert ci.dependency_components(cases) == [["a", "b", "c"], ["d", "e"]]
    split = ci.dependency_split(cases, test_fraction=0.4, seed=7)
    assert len({split[c] for c in ("a", "b", "c")}) == 1
    assert split["d"] == split["e"]


def test_a_tampered_frozen_analysis_is_refused(seed):
    import hashlib

    from toxagent.application.activity.replay import ReplayError

    path = seed / "replay.json"
    data = json.loads(path.read_text())
    ref = next(iter(data["analyses"]))
    data["analyses"][ref]["predictor_response"]["predictions"]["herg"]["probability_blocker"] = 0.01
    path.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        FrozenReplayStore.load(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
