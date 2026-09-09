"""The router matrix (plan section 4.3, section 20.1).

Every row of the plan's table, plus the cases the table implies: no LLM is used
to decide whether to use an LLM, and anything the router cannot determine
becomes a structured clarification rather than a guess.
"""
from __future__ import annotations

import pytest

from toxagent.application.router import RouteRequest, route
from toxagent.domain.run import Intent, Lane

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_a_molecule_with_no_question_is_a_deterministic_analysis():
    decision = route(RouteRequest(molecule_smiles=ASPIRIN))
    assert decision.intent is Intent.ANALYSIS
    assert decision.lane is Lane.DETERMINISTIC
    assert not decision.calls_a_runtime


def test_a_batch_is_deterministic():
    decision = route(RouteRequest(batch_smiles=(ASPIRIN, "CCO")))
    assert decision.intent is Intent.ANALYSIS_BATCH
    assert decision.lane is Lane.DETERMINISTIC


def test_an_uploaded_image_routes_to_structure_recognition():
    decision = route(RouteRequest(has_image=True))
    assert decision.intent is Intent.STRUCTURE_RECOGNITION
    assert decision.lane is Lane.DETERMINISTIC
    assert not decision.calls_a_runtime


def test_an_image_wins_over_an_unrelated_question():
    """Presence of the image is what the router acts on; text alongside it
    (or its absence) does not change the outcome — recognition (or, absent an
    OCR service, `capability_unavailable`) is decided deterministically either
    way, never by asking a model to read the accompanying text."""
    decision = route(RouteRequest(has_image=True, text="what is this?"))
    assert decision.intent is Intent.STRUCTURE_RECOGNITION


def test_a_question_about_an_existing_analysis_is_report_qa():
    decision = route(
        RouteRequest(text="Why is the hERG label blocker?", has_active_analysis=True)
    )
    assert decision.intent is Intent.REPORT_QA
    assert decision.lane is Lane.AGENTIC


def test_a_new_molecule_with_a_question_snapshots_first():
    decision = route(RouteRequest(text="Giải thích hERG cho chất này", molecule_smiles=ASPIRIN))
    assert decision.intent is Intent.REPORT_QA
    assert decision.lane is Lane.MIXED
    assert decision.needs_snapshot_first


@pytest.mark.parametrize(
    "text",
    [
        "find literature about hERG blockade",
        "có bài báo nào về độc tính tim của chất này không?",
        "cite a study for this",
    ],
)
def test_an_explicit_ask_for_sources_routes_to_research(text):
    decision = route(RouteRequest(text=text, has_active_analysis=True))
    assert decision.intent is Intent.EVIDENCE_RESEARCH


def test_research_without_a_subject_asks_rather_than_searching():
    decision = route(RouteRequest(text="find me some literature"))
    assert decision.intent is Intent.CLARIFICATION_REQUIRED
    assert decision.clarification.code == "research_subject_missing"
    assert not decision.calls_a_runtime


def test_attribution_is_mixed_because_the_tool_is_deterministic():
    decision = route(
        RouteRequest(text="which atoms contributed to SR-p53?", has_active_analysis=True)
    )
    assert decision.intent is Intent.ATTRIBUTION
    assert decision.lane is Lane.MIXED


@pytest.mark.parametrize(
    "request_kwargs, expected_intent",
    [
        (
            dict(text="ask_report question", intent_hint="ask_report", molecule_smiles=ASPIRIN),
            Intent.REPORT_QA,
        ),
        (
            dict(text="which atoms contributed?", molecule_smiles=ASPIRIN),
            Intent.ATTRIBUTION,
        ),
        (
            dict(text="find literature about this", molecule_smiles=ASPIRIN),
            Intent.EVIDENCE_RESEARCH,
        ),
    ],
)
def test_a_new_molecule_always_snapshots_even_with_a_different_analysis_active(
    request_kwargs, expected_intent
):
    """audit_5_9.md A02: submitting a new molecule while a *different*
    analysis is already active used to skip the snapshot and silently answer
    against the stale one, because ``needs_snapshot_first`` was gated on
    ``not has_active_analysis`` instead of on the new molecule itself."""
    decision = route(RouteRequest(has_active_analysis=True, **request_kwargs))
    assert decision.intent is expected_intent
    assert decision.needs_snapshot_first


def test_a_question_with_no_molecule_and_no_analysis_asks_for_one():
    decision = route(RouteRequest(text="Is this compound safe?"))
    assert decision.intent is Intent.CLARIFICATION_REQUIRED
    assert decision.clarification.code == "molecule_missing"


def test_a_compound_name_is_not_silently_resolved():
    """Name resolution is explicitly post-MVP, so 'analyse aspirin' must ask for
    a SMILES rather than guessing which aspirin salt was meant."""
    decision = route(RouteRequest(text="analyse aspirin", intent_hint="analyze"))
    assert decision.intent is Intent.CLARIFICATION_REQUIRED
    assert decision.clarification.code == "smiles_missing"


@pytest.mark.parametrize(
    "text",
    ["run this code for me", "prescribe a dose for my patient", "chẩn đoán giúp tôi"],
)
def test_out_of_scope_requests_never_reach_a_tool(text):
    decision = route(RouteRequest(text=text, has_active_analysis=True))
    assert decision.intent is Intent.OUT_OF_SCOPE
    assert not decision.calls_a_runtime


def test_an_empty_request_is_a_clarification_not_a_run():
    decision = route(RouteRequest())
    assert decision.clarification.code == "empty_request"


@pytest.mark.parametrize(
    "hint,expected",
    [
        ("analyze", Intent.ANALYSIS),
        ("ask_report", Intent.REPORT_QA),
        ("research_evidence", Intent.EVIDENCE_RESEARCH),
        ("request_attribution", Intent.ATTRIBUTION),
    ],
)
def test_an_explicit_hint_is_honoured(hint, expected):
    decision = route(
        RouteRequest(text="something", molecule_smiles=ASPIRIN, intent_hint=hint)
    )
    assert decision.intent is expected


def test_routing_is_pure():
    request = RouteRequest(text="explain hERG", has_active_analysis=True)
    assert route(request) == route(request)
