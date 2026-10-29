"""What one turn changed in a case and what it kept (AG-UI case board, US4).

The before and after are the case's log replayed to two revisions, so these
fixtures are logs: two runs, the researcher's context between them, and the
artifacts each run was given.
"""
from __future__ import annotations

import pytest

from toxagent.domain import scientific_case as sc

SESSION = "ses_" + "1" * 32
CASE_ID = "scase_" + "2" * 32
OBS = "obs_" + "3" * 32
RUN_1 = "run_" + "6" * 32
RUN_2 = "run_" + "7" * 32
SCREENING = {"kind": "screening", "source_id": "scr_frozen_1", "content_sha256": "a" * 64}


def upd(op, actor="model", run_id=RUN_1, **payload):
    return sc.CaseUpdate(op=op, payload=payload, actor=actor, at="2026-10-22T00:00:00+00:00",
                         run_id=run_id)


def _log(*, second_artifact=SCREENING) -> list[sc.CaseUpdate]:
    """Run 1 opens the case, poses a hypothesis and an uncertainty and a
    conclusion; the researcher adds a measurement; run 2 weakens the
    hypothesis, resolves the uncertainty and replaces the conclusion."""
    updates = [
        upd("open", actor="server", case_id=CASE_ID, session_id=SESSION,
            subject_key="activity:" + "f" * 40, question="Is this ligand worth a direct assay?",
            subject_refs=["screening:scr_frozen_1"]),
        upd("attach_run", actor="server", goal="first look",
            binding={"binding_sha256": "b" * 64, "artifact_refs": [SCREENING]}),
        upd("add_hypothesis", statement="The ligand is potent on EGFR", kind="model_signal",
            refutation_condition="A measured pIC50 below 6"),
        upd("record_uncertainty", kind="missing_data", description="No direct assay",
            severity="high"),
        upd("set_conclusion", can_say=[], cannot_say=["Whether it is potent"]),
        upd("finish_run", actor="server", stop_reason="needs_user_input"),
        upd("add_context", actor="user", run_id=None, key="measured_pIC50", value="5.4",
            unit="pIC50", source="researcher"),
        upd("attach_run", actor="server", run_id=RUN_2, goal="with the measurement",
            binding={"binding_sha256": "c" * 64, "artifact_refs": [second_artifact]}),
        upd("record_evidence", run_id=RUN_2, claim="The lab measured pIC50 5.4",
            source_class="user_supplied", source_ref="context:c1", stance="contradicts",
            hypothesis_ids=["h1"]),
        upd("revise_hypothesis", run_id=RUN_2, hypothesis_id="h1", status="weakened",
            reason="the measurement is below the estimate's interval"),
        upd("resolve_uncertainty", run_id=RUN_2, uncertainty_id="u1",
            resolution="the researcher's assay", evidence_ids=["e1"]),
        upd("set_conclusion", run_id=RUN_2, can_say=[
            {"text": "The measured potency is below the estimate", "evidence_ids": ["e1"]}]),
        upd("finish_run", actor="server", run_id=RUN_2, stop_reason="sufficient"),
    ]
    revision = 0
    out = []
    for update in updates:
        revision += 1
        out.append(sc.CaseUpdate(op=update.op, payload=update.payload, actor=update.actor,
                                 at=update.at, run_id=update.run_id, revision=revision))
    return out


def test_the_second_turn_reports_what_it_changed_and_what_it_kept():
    changes = sc.run_changes(_log(), RUN_2)

    assert changes["schema_version"] == sc.CHANGES_SCHEMA_VERSION
    assert changes["previous_run_id"] == RUN_1
    assert (changes["from_revision"], changes["to_revision"]) == (6, 13)
    # The researcher's value is new in the turn that first saw it.
    assert [c["key"] for c in changes["context"]["added"]] == ["measured_pIC50"]
    assert changes["context"]["added"][0]["actor"] == "user"
    assert changes["updates_by_actor"] == {"user": 1, "server": 2, "model": 4}
    (hypothesis,) = changes["hypotheses"]["changed"]
    assert hypothesis["id"] == "h1"
    assert {d["field"]: (d["before"], d["after"]) for d in hypothesis["changes"]}["status"] == (
        "open", "weakened")
    (uncertainty,) = changes["uncertainties"]["changed"]
    assert {d["field"] for d in uncertainty["changes"]} == {"status", "resolution"}
    assert [e["id"] for e in changes["evidence"]["added"]] == ["e1"]
    assert changes["conclusion"]["changed"] is True
    assert changes["conclusion"]["before"]["cannot_say"] == ["Whether it is potent"]
    assert changes["run"]["stop_reason"] == "sufficient"


def test_the_frozen_artifacts_a_turn_kept_are_reported_as_kept():
    changes = sc.run_changes(_log(), RUN_2)
    artifacts = changes["artifacts"]
    assert artifacts["compared"] is True
    assert artifacts["kept"] == [SCREENING]
    assert artifacts["changed"] == [] and artifacts["added"] == [] and artifacts["removed"] == []
    assert changes["subject"] == {"subject_key": "activity:" + "f" * 40, "kept": True}


def test_a_changed_artifact_hash_is_a_changed_input():
    replaced = {**SCREENING, "content_sha256": "d" * 64}
    artifacts = sc.run_changes(_log(second_artifact=replaced), RUN_2)["artifacts"]
    assert artifacts["kept"] == []
    assert artifacts["changed"] == [{"kind": "screening", "source_id": "scr_frozen_1",
                                     "before_sha256": "a" * 64, "after_sha256": "d" * 64}]


def test_the_first_turn_has_nothing_to_compare_against():
    changes = sc.run_changes(_log(), RUN_1)
    assert changes["previous_run_id"] is None and changes["from_revision"] == 0
    assert changes["artifacts"]["compared"] is False
    assert [h["id"] for h in changes["hypotheses"]["added"]] == ["h1"]
    assert changes["hypotheses"]["kept"] == []
    # The context the researcher adds later is not part of the first turn.
    assert changes["context"]["added"] == []


def test_a_turn_that_touched_nothing_keeps_everything():
    log = _log()
    quiet = log + [
        sc.CaseUpdate(op="attach_run", payload={"goal": "again", "binding": {
            "binding_sha256": "e" * 64, "artifact_refs": [SCREENING]}},
            actor="server", at="2026-10-22T00:00:00+00:00", run_id="run_" + "8" * 32,
            revision=14),
    ]
    changes = sc.run_changes(quiet, "run_" + "8" * 32)
    assert changes["hypotheses"]["kept"] == ["h1"]
    assert changes["uncertainties"]["kept"] == ["u1"]
    assert changes["context"]["kept"] == ["c1"]
    assert changes["conclusion"]["changed"] is False
    assert changes["artifacts"]["kept"] == [SCREENING]


def test_a_run_that_never_touched_the_case_is_not_in_it():
    with pytest.raises(sc.RunNotInCase):
        sc.run_changes(_log(), "run_" + "9" * 32)
