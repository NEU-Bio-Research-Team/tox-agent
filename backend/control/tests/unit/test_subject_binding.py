"""The subject a run is about, as a pure contract (#44).

What the binding must guarantee before anything persists it: one ligand on two
targets is two case keys; the same subject read from two screenings is one;
an optional enrichment neither splits a case nor hides a different protein;
the binding cannot be edited without its hash noticing; and a run with a
binding is written in an envelope an older worker refuses to claim.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from toxagent.application.policy import Actor
from toxagent.application.runs import envelope
from toxagent.application.runs.envelope import UnreadableEnvelope, from_envelope, to_envelope
from toxagent.application.runs.scheduler import RunContext
from toxagent.domain import scientific_case as sc
from toxagent.domain.activity import TargetIdentity
from toxagent.domain.ids import ANALYSIS, RUN, SCREENING, SESSION, new_id
from toxagent.domain.subject import (
    ArtifactRef,
    InvalidSubject,
    RunSubjectBinding,
    SubjectIdentity,
    SubjectKind,
    activity_subject_key,
    analysis_binding,
)

LIGAND = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
CONTEXT = "IC50|wild_type"


def _screening(target: TargetIdentity, *, screening_id: str | None = None,
               context: str = CONTEXT, refs=()) -> RunSubjectBinding:
    return RunSubjectBinding(
        kind=SubjectKind.ACTIVITY_SCREENING,
        subject_ref={"screening_id": screening_id or new_id(SCREENING)},
        identity=SubjectIdentity(canonical_smiles=LIGAND, target=target, assay_context=context),
        artifact_refs=tuple(refs),
    )


EGFR = TargetIdentity(target_id="EGFR", uniprot_id="P00533")
KCNH2 = TargetIdentity(target_id="KCNH2", uniprot_id="Q12809")


def test_one_ligand_on_two_targets_is_two_cases():
    assert _screening(EGFR).subject_key != _screening(KCNH2).subject_key


def test_another_assay_context_or_variant_is_another_case():
    base = _screening(EGFR)
    assert base.subject_key != _screening(EGFR, context="Ki|wild_type").subject_key
    mutant = TargetIdentity(target_id="EGFR", uniprot_id="P00533", variant="T790M")
    assert base.subject_key != _screening(mutant).subject_key
    assert "variant" in base.mismatches(_screening(mutant).identity)


def test_the_same_subject_from_two_screenings_is_one_case():
    first, second = _screening(EGFR), _screening(EGFR)
    assert first.screening_id != second.screening_id
    assert first.subject_key == second.subject_key
    assert first.binding_sha256 != second.binding_sha256


def test_an_enrichment_neither_splits_the_case_nor_hides_another_protein():
    resolved = _screening(EGFR)
    bare = _screening(TargetIdentity(target_id="EGFR"))
    assert resolved.subject_key == bare.subject_key
    assert resolved.mismatches(bare.identity) == ()
    other = _screening(TargetIdentity(target_id="EGFR", uniprot_id="P99999"))
    assert resolved.subject_key == other.subject_key
    assert resolved.mismatches(other.identity) == ("uniprot_id",)


def test_the_case_key_fits_its_column_and_never_carries_the_structure():
    key = _screening(EGFR).subject_key
    assert key.startswith("activity:") and len(key) <= sc.MAX_TEXT and len(key) <= 80
    assert LIGAND not in key
    analysis = analysis_binding(new_id(ANALYSIS), canonical_smiles=LIGAND,
                                content_sha256_="0" * 64, resolved_by="explicit")
    assert analysis.subject_key == f"analysis:{analysis.analysis_id}"


def test_the_key_is_computed_from_what_an_artifact_states():
    binding = _screening(EGFR)
    assert activity_subject_key(binding.identity.to_dict()) == binding.subject_key


def test_a_binding_round_trips_and_cannot_be_edited_silently():
    binding = _screening(EGFR, refs=[ArtifactRef("screening", new_id(SCREENING), "a" * 64)])
    data = binding.to_dict()
    assert RunSubjectBinding.from_dict(data) == binding
    tampered = {**data, "identity": {**data["identity"], "assay_context": "Ki|wild_type"}}
    with pytest.raises(InvalidSubject):
        RunSubjectBinding.from_dict(tampered)


def test_an_activity_subject_needs_its_whole_identity():
    with pytest.raises(InvalidSubject):
        RunSubjectBinding(
            kind=SubjectKind.ACTIVITY_SCREENING, subject_ref={"screening_id": new_id(SCREENING)},
            identity=SubjectIdentity(canonical_smiles=LIGAND),
        )


def test_an_unset_variant_keeps_every_existing_hash():
    assert "variant" not in EGFR.to_dict()
    assert TargetIdentity.from_dict(EGFR.to_dict()) == EGFR


def _context(**changes) -> RunContext:
    base = RunContext(
        actor=Actor(subject_id="user-1"), session_id=new_id(SESSION), run_id=new_id(RUN),
        intent=__import__("toxagent.domain.run", fromlist=["Intent"]).Intent.DECISION_SUPPORT,
        text="why review?",
    )
    return replace(base, **changes)


def test_a_bound_run_is_a_v2_envelope_and_a_legacy_run_stays_v1():
    binding = _screening(EGFR)
    bound = to_envelope(_context(subject=binding.to_dict()))
    assert bound["v"] == 2
    assert from_envelope(bound).subject == binding.to_dict()
    assert to_envelope(_context())["v"] == 1


def test_a_worker_that_predates_subjects_refuses_a_bound_run(monkeypatch):
    bound = to_envelope(_context(subject=_screening(EGFR).to_dict()))
    monkeypatch.setattr(envelope, "ENVELOPE_VERSION", 1)
    with pytest.raises(UnreadableEnvelope):
        from_envelope(bound)


def test_an_envelope_with_an_edited_subject_is_unreadable():
    bound = to_envelope(_context(subject=_screening(EGFR).to_dict()))
    bound["subject"]["identity"]["target"]["target_id"] = "KCNH2"
    with pytest.raises(UnreadableEnvelope):
        from_envelope(bound)


def _update(op: str, actor: str = "server", **payload) -> sc.CaseUpdate:
    return sc.CaseUpdate(op=op, payload=payload, actor=actor, at="2026-10-21T00:00:00+00:00",
                         run_id=new_id(RUN))


def test_case_event_replay_rebuilds_subject_scope_spans_and_provenance():
    binding = _screening(EGFR)
    updates = [
        _update("open", case_id="scase_" + "1" * 32, session_id=new_id(SESSION),
                subject_key=binding.subject_key, subject=binding.case_subject()),
        _update("attach_run", goal="why review?",
                binding={"binding_sha256": binding.binding_sha256, "artifact_refs": []}),
        _update(
            "record_evidence", claim="Measured IC50 against EGFR.",
            source_class="external_experimental", source_ref="evidence:evd_" + "2" * 32,
            stance="contextual", locator="table-wrap:1/row:3",
            spans=["table-wrap:1/row:3", "table-wrap:1/row:4"],
            scope={"target_id": "EGFR", "assay_type": "binding", "unit": "nM",
                   "ligand_id": LIGAND, "comparator": "="},
            provenance={"observation_kind": "evidence_record"},
        ),
    ]
    case, applied = sc_fold(updates)
    replayed = sc.replay(applied)
    assert replayed == case
    assert replayed.subject["identity"]["target"]["target_id"] == "EGFR"
    entry = replayed.evidence[0]
    assert entry.scope["assay_type"] == "binding" and entry.scope["unit"] == "nM"
    assert entry.spans == ("table-wrap:1/row:3", "table-wrap:1/row:4")
    assert replayed.runs[0].binding["binding_sha256"] == binding.binding_sha256
    assert sc.ScientificCaseV1.from_dict(replayed.to_dict()) == replayed


def test_only_the_server_records_where_a_source_came_from():
    binding = _screening(EGFR)
    case = sc.apply(None, _update("open", case_id="scase_" + "1" * 32,
                                  session_id=new_id(SESSION), subject_key=binding.subject_key))
    with pytest.raises(sc.InvalidCaseUpdate):
        sc.apply(case, _update(
            "record_evidence", actor="model", claim="x", source_class="predictor_fact",
            source_ref="observation:obs_" + "3" * 32, stance="contextual",
            provenance={"record_sha256": "forged"},
        ))


def test_policy_and_workflow_facts_are_not_independent_evidence():
    assert sc.SourceClass.POLICY_FACT.value not in sc.INDEPENDENT_SOURCES
    assert sc.SourceClass.WORKFLOW_FACT.value not in sc.INDEPENDENT_SOURCES
    assert sc.REF_KIND[sc.SourceClass.WORKFLOW_FACT.value] == "observation"


def sc_fold(updates):
    from toxagent.application.investigation.scientific_case_service import fold

    return fold(None, updates)
