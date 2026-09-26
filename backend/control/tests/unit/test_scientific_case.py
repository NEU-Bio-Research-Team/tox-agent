"""ScientificCaseV1 transitions (RETHINK §3.1, §4.3, §4.4 step 1)."""
from __future__ import annotations

import pytest

from toxagent.domain import scientific_case as sc

SESSION = "ses_" + "1" * 32
CASE_ID = "scase_" + "2" * 32
OBS = "obs_" + "3" * 32
EVD = "evd_" + "4" * 32
EVD2 = "evd_" + "5" * 32
RUN = "run_" + "6" * 32


def upd(op, actor="model", run_id=RUN, **payload):
    return sc.CaseUpdate(op=op, payload=payload, actor=actor, at="2026-09-25T00:00:00+00:00",
                         run_id=run_id)


def opened(**extra):
    return sc.apply(None, upd(
        "open", actor="server", case_id=CASE_ID, session_id=SESSION,
        subject_key="analysis:ana_x", question="Should compound A go to hERG patch clamp next?",
        subject_refs=["analysis:ana_x"], **extra,
    ))


def with_hypotheses(*statements):
    case = opened()
    for statement in statements:
        case = sc.apply(case, upd(
            "add_hypothesis", statement=statement, kind="model_signal",
            refutation_condition="A patch-clamp IC50 far above exposure",
        ))
    return case


def test_opening_a_case_starts_at_revision_one_with_the_question():
    case = opened()
    assert case.revision == 1
    assert case.question.startswith("Should compound A")
    assert case.status == "open"


def test_hypothesis_ids_are_issued_by_the_server_in_order():
    case = with_hypotheses("A blocks hERG", "The signal is an assay artefact")
    assert [h.id for h in case.hypotheses] == ["h1", "h2"]
    assert case.revision == 3


def test_a_hypothesis_needs_a_refutation_condition():
    with pytest.raises(sc.InvalidCaseUpdate, match="refutation_condition"):
        sc.apply(opened(), upd("add_hypothesis", statement="A blocks hERG", kind="mechanism"))


def test_a_repeated_hypothesis_is_refused():
    case = with_hypotheses("A blocks hERG")
    with pytest.raises(sc.InvalidCaseUpdate, match="already"):
        sc.apply(case, upd("add_hypothesis", statement="a blocks herg", kind="mechanism",
                           refutation_condition="x"))


def test_the_hypothesis_ceiling_is_enforced():
    case = with_hypotheses(*[f"hypothesis {i}" for i in range(sc.MAX_HYPOTHESES)])
    with pytest.raises(sc.InvalidCaseUpdate, match="at most"):
        sc.apply(case, upd("add_hypothesis", statement="one more", kind="other",
                           refutation_condition="x"))


def evidence(case, **overrides):
    payload = dict(
        claim="Patch-clamp IC50 of 0.1 µM reported for A", source_class="external_experimental",
        source_ref=f"evidence:{EVD}", stance="supports", directness="direct",
        hypothesis_ids=["h1"], scope={"endpoint": "herg", "assay": "patch clamp"},
    )
    payload.update(overrides)
    return sc.apply(case, upd("record_evidence", **payload))


def test_an_entry_must_cite_the_ref_kind_its_source_class_names():
    with pytest.raises(sc.InvalidCaseUpdate, match="'observation:' ref"):
        evidence(with_hypotheses("A blocks hERG"), source_class="predictor_fact")


def test_agent_synthesis_is_not_a_source_class():
    with pytest.raises(sc.InvalidCaseUpdate, match="source_class"):
        evidence(with_hypotheses("A blocks hERG"), source_class="agent_synthesis")


def test_a_stance_needs_the_hypothesis_it_bears_on():
    with pytest.raises(sc.InvalidCaseUpdate, match="hypothesis_ids"):
        evidence(with_hypotheses("A blocks hERG"), hypothesis_ids=[])


def test_an_unknown_hypothesis_is_named_in_the_refusal():
    with pytest.raises(sc.InvalidCaseUpdate, match=r"\['h9'\]"):
        evidence(with_hypotheses("A blocks hERG"), hypothesis_ids=["h9"])


def test_contextual_evidence_may_stand_unlinked():
    case = evidence(with_hypotheses("A blocks hERG"), stance="contextual", hypothesis_ids=[])
    assert case.evidence[0].id == "e1"


def test_recording_the_same_entry_twice_is_one_entry_and_no_revision():
    case = evidence(with_hypotheses("A blocks hERG"))
    again = evidence(case)
    assert again is case
    assert len(again.evidence) == 1


def test_scope_keys_are_a_closed_set():
    with pytest.raises(sc.InvalidCaseUpdate, match="scope keys"):
        evidence(with_hypotheses("A blocks hERG"), scope={"toxicity": "high"})


def test_user_context_is_citable_only_once_it_exists():
    case = with_hypotheses("A blocks hERG")
    with pytest.raises(sc.InvalidCaseUpdate, match="no context item"):
        evidence(case, source_class="user_supplied", source_ref="context:c1")
    case = sc.apply(case, upd("add_context", actor="user", key="assay_result",
                              value="In-house patch clamp IC50 = 30 µM"))
    case = evidence(case, source_class="user_supplied", source_ref="context:c1",
                    stance="contradicts")
    assert case.evidence[0].independent


def test_only_the_user_adds_context():
    with pytest.raises(sc.InvalidCaseUpdate, match="cannot be performed by the model"):
        sc.apply(opened(), upd("add_context", key="k", value="v"))


def test_only_the_server_attaches_runs():
    with pytest.raises(sc.InvalidCaseUpdate, match="cannot be performed by the model"):
        sc.apply(opened(), upd("attach_run", run_id=RUN))


def test_a_supported_status_needs_supporting_evidence():
    case = with_hypotheses("A blocks hERG")
    with pytest.raises(sc.InvalidCaseUpdate, match="record_evidence first"):
        sc.apply(case, upd("revise_hypothesis", hypothesis_id="h1", status="supported",
                           reason="the model says so"))
    case = evidence(case)
    case = sc.apply(case, upd("revise_hypothesis", hypothesis_id="h1", status="supported",
                              reason="direct patch-clamp data"))
    assert case.hypotheses[0].status == "supported"


def test_refuting_needs_contradicting_evidence():
    case = evidence(with_hypotheses("A blocks hERG"))
    with pytest.raises(sc.InvalidCaseUpdate, match="contradicts"):
        sc.apply(case, upd("revise_hypothesis", hypothesis_id="h1", status="refuted", reason="x"))


def test_unresolvable_needs_only_a_reason():
    case = sc.apply(with_hypotheses("A blocks hERG"), upd(
        "revise_hypothesis", hypothesis_id="h1", status="unresolvable",
        reason="no exposure data is reachable"))
    assert case.hypotheses[0].status_reason == "no exposure data is reachable"


def test_a_conclusion_line_must_trace_to_the_ledger():
    case = evidence(with_hypotheses("A blocks hERG"))
    with pytest.raises(sc.InvalidCaseUpdate, match="cannot_say or is a hypothesis"):
        sc.apply(case, upd("set_conclusion", can_say=[{"text": "A blocks hERG", "evidence_ids": []}]))
    case = sc.apply(case, upd(
        "set_conclusion", can_say=[{"text": "A was reported to block hERG in patch clamp",
                                    "evidence_ids": ["e1"]}],
        cannot_say=["whether this matters at therapeutic exposure"],
        what_would_change=["a free Cmax well below the IC50"],
    ))
    assert case.conclusion.can_say[0].evidence_ids == ("e1",)


def test_only_the_model_sets_a_conclusion():
    with pytest.raises(sc.InvalidCaseUpdate, match="cannot be performed by the user"):
        sc.apply(opened(), upd("set_conclusion", actor="user"))


def test_a_proposed_test_must_discriminate_something():
    case = with_hypotheses("A blocks hERG", "The signal is an assay artefact")
    with pytest.raises(sc.InvalidCaseUpdate, match="discriminate"):
        sc.apply(case, upd("propose_next_test", test="Patch clamp", rationale="standard",
                           discriminates=[]))
    case = sc.apply(case, upd(
        "propose_next_test", test="Manual patch clamp at 3 concentrations",
        rationale="separates real block from a binding-assay artefact",
        discriminates=["h1", "h2"], expected_readouts=["IC50 < 1 µM supports h1"],
    ))
    assert case.next_tests[0].id == "t1"


def test_resolving_an_uncertainty_keeps_it_with_its_resolution():
    case = with_hypotheses("A blocks hERG")
    case = sc.apply(case, upd("record_uncertainty", kind="missing_exposure",
                              description="No free Cmax is known", severity="blocking",
                              hypothesis_ids=["h1"]))
    case = sc.apply(case, upd("add_context", actor="user", key="free_cmax", value="0.05 µM"))
    case = evidence(case, source_class="user_supplied", source_ref="context:c1", stance="contextual",
                    hypothesis_ids=["h1"], claim="Free Cmax is 0.05 µM")
    case = sc.apply(case, upd("resolve_uncertainty", uncertainty_id="u1",
                              resolution="The user supplied free Cmax", evidence_ids=["e1"]))
    assert case.uncertainties[0].status == "resolved"
    assert case.open_uncertainties == ()
    assert case.coverage["blocking_uncertainties"] == 0


def test_ref_coverage_and_quality_coverage_are_reported_apart():
    case = with_hypotheses("A blocks hERG", "The signal is an assay artefact")
    case = evidence(case, source_class="predictor_fact", source_ref=f"observation:{OBS}",
                    claim="hERG probability 0.73", locator="predictions.herg.probability_blocker")
    coverage = case.coverage
    assert coverage["with_any_source"] == 1
    # A model score is a source, not independent direct evidence.
    assert coverage["with_independent_direct_evidence"] == 0
    assert coverage["with_counterevidence_considered"] == 0
    case = evidence(case)
    case = sc.apply(case, upd("record_action", action="counterevidence_search",
                              purpose="look for data showing A does not block hERG",
                              hypothesis_ids=["h1", "h2"], outcome="nothing found"))
    coverage = case.coverage
    assert coverage["with_independent_direct_evidence"] == 1
    assert coverage["with_counterevidence_considered"] == 2
    assert "score" not in coverage


def test_a_closed_case_refuses_new_work():
    case = sc.apply(opened(), upd("close", actor="user"))
    with pytest.raises(sc.InvalidCaseUpdate, match="closed"):
        sc.apply(case, upd("add_hypothesis", statement="x", kind="other", refutation_condition="y"))


def test_replaying_the_log_rebuilds_the_same_case():
    log = [upd("open", actor="server", case_id=CASE_ID, session_id=SESSION,
               subject_key="analysis:ana_x", question="q")]
    log.append(upd("add_hypothesis", statement="A blocks hERG", kind="mechanism",
                   refutation_condition="IC50 above 30 µM"))
    log.append(upd("record_evidence", claim="c", source_class="external_experimental",
                   source_ref=f"evidence:{EVD}", stance="supports", directness="direct",
                   hypothesis_ids=["h1"]))
    log.append(upd("attach_run", actor="server", run_id=RUN, goal="q"))
    log.append(upd("finish_run", actor="server", run_id=RUN, stop_reason="sufficient",
                   usage={"tool_calls": 3}))
    case = None
    for item in log:
        case = sc.apply(case, item)
    assert sc.replay(log) == case
    assert case.revision == len(log)


def test_the_dict_form_round_trips():
    case = evidence(with_hypotheses("A blocks hERG"))
    case = sc.apply(case, upd("record_uncertainty", kind="assay_mismatch", description="d"))
    case = sc.apply(case, upd("set_conclusion", can_say=[{"text": "t", "evidence_ids": ["e1"]}]))
    assert sc.ScientificCaseV1.from_dict(case.to_dict()) == case


def relation(**overrides):
    item = {"proposition": "A blocks hERG", "source_class": "external_experimental",
            "source_id": EVD, "relation": "supports", "directness": "direct",
            "endpoint": "herg", "species": "human"}
    item.update(overrides)
    return item


def test_answer_relations_become_server_entries_linked_by_statement():
    case = with_hypotheses("A blocks hERG")
    updates = sc.updates_from_answer(case, [
        relation(),
        relation(source_class="agent_synthesis", source_id="syn_x"),
        relation(proposition="Something nobody hypothesised", source_id=EVD2),
        relation(source_class="predictor_fact", source_id=OBS, relation="contextual",
                 directness="indirect"),
    ], run_id=RUN, at="t")
    assert [u.actor for u in updates] == ["server"] * 3
    for item in updates:
        case = sc.apply(case, item)
    linked, unlinked, predictor = case.evidence
    assert linked.hypothesis_ids == ("h1",) and linked.stance == "supports"
    # A stance with nothing to bear on is only context.
    assert unlinked.hypothesis_ids == () and unlinked.stance == "contextual"
    assert predictor.source_ref == f"observation:{OBS}" and predictor.directness == "indirect"
    assert linked.scope == {"endpoint": "herg", "species": "human"}


def test_what_an_answer_cited_becomes_unlinked_context_once():
    """W9-04: a citation says the answer relied on a source, not its stance."""
    case = with_hypotheses("A blocks hERG")
    case = evidence(case)  # the model already recorded evidence:EVD
    cited = [
        {"source_class": "predictor_fact", "source_id": OBS, "claim": "p = 0.73",
         "locator": "predictions.herg.probability_blocker"},
        {"source_class": "predictor_fact", "source_id": OBS, "claim": "label blocker"},
        {"source_class": "external_experimental", "source_id": EVD, "claim": "held already"},
        {"source_class": "external_regulatory", "source_id": EVD2, "claim": "skipped"},
        {"source_class": "user_supplied", "source_id": "c1", "claim": "never from an answer"},
    ]
    updates = sc.updates_from_citations(case, cited, run_id=RUN, at="t",
                                        skip_refs=[f"evidence:{EVD2}"])
    assert [u.payload["source_ref"] for u in updates] == [f"observation:{OBS}"]
    after = sc.apply(case, updates[0])
    entry = after.evidence[-1]
    assert (entry.actor, entry.stance, entry.hypothesis_ids) == ("server", "contextual", ())
    assert entry.locator == "predictions.herg.probability_blocker"


def test_the_dossier_keeps_the_three_explanation_layers_apart():
    case = with_hypotheses("A blocks hERG")
    case = evidence(case)
    case = evidence(case, source_class="explanation_fact", source_ref=f"observation:{OBS}",
                    claim="The basic amine carries the largest attribution", stance="contextual",
                    hypothesis_ids=["h1"], directness="not_assessed")
    case = evidence(case, source_class="external_regulatory", source_ref=f"evidence:{EVD2}",
                    stance="contextual", hypothesis_ids=[], claim="Label mentions QT")
    case = sc.apply(case, upd("record_action", action="search_toxicology_evidence",
                              purpose="find direct hERG data for A", decision="continue"))
    case = sc.apply(case, upd("set_conclusion", can_say=[{"text": "t", "evidence_ids": ["e1"]}]))
    statement = {"faithfulness_vs_random_control": "not_better_than_random_control"}
    dossier = sc.compile_dossier(case, run_id=RUN, stop_reason="sufficient", answer_id=None,
                                 explainer_statements={f"observation:{OBS}": statement})
    layers = dossier["explanation_layers"]
    assert [e["id"] for e in layers["model_attribution"]] == ["e2"]
    assert layers["model_attribution"][0]["explainer_validation"] == statement
    assert [e["id"] for e in layers["scientific_evidence"]] == ["e1", "e3"]
    assert layers["agent_decisions"][0]["purpose"] == "find direct hERG data for A"
    assert [e["id"] for e in dossier["hypotheses"][0]["evidence_for"]] == ["e1"]
    assert [e["id"] for e in dossier["unlinked_evidence"]] == ["e3"]
    assert dossier["conclusion"]["can_say"][0]["sources"] == [f"evidence:{EVD}"]
    assert dossier["schema_version"] == sc.DOSSIER_SCHEMA_VERSION


def test_the_checkpoint_names_the_case_and_how_to_read_it():
    case = evidence(with_hypotheses("A blocks hERG"))
    text = sc.checkpoint_summary(case)
    assert CASE_ID in text and "h1 [open]" in text and "get_scientific_case" in text


def test_the_checkpoint_carries_the_ids_a_turn_writes_against():
    """W9-01: one update per turn, without a read first."""
    case = evidence(with_hypotheses("A blocks hERG"))
    case = sc.apply(case, upd(
        "set_conclusion", can_say=[{"text": "IC50 0.1 µM reported", "evidence_ids": ["e1"]}],
        cannot_say=["whether A blocks at its exposure"],
    ))
    summary = sc.checkpoint_summary(case)
    assert f"- e1 supports on h1 [external_experimental evidence:{EVD}]" in summary
    assert "can say: IC50 0.1 µM reported (e1)" in summary
    assert "cannot say: whether A blocks at its exposure" in summary


def test_the_checkpoint_shows_only_the_latest_ledger_entries():
    case = with_hypotheses("A blocks hERG")
    for index in range(12):
        case = evidence(case, claim=f"entry {index}")
    summary = sc.checkpoint_summary(case, evidence_limit=8)
    assert "Ledger (latest 8 of 12):" in summary
    assert "- e12 " in summary and "- e4 " not in summary
