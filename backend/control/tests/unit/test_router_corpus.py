"""P1-9: an adversarial bilingual corpus for the router.

The old router matched terms with ``term in text``. Two of these cases are the
exact sentences that breaks on — "executive summary" contains "execute", and
"contributing factors" contains "contribut" — and both of them routed a
perfectly ordinary question to the wrong workflow.

The corpus is the gate. A term list may grow, but not by breaking a row here.
"""
from __future__ import annotations

import pytest

from toxagent.application.intent_matching import (
    contains_phrase,
    matched_terms,
    normalize,
    tokenize,
)
from toxagent.application.router import (
    ATTRIBUTION_TERMS,
    OUT_OF_SCOPE_TERMS,
    RESEARCH_TERMS,
    ROUTER_VERSION,
    RouteRequest,
    route,
)
from toxagent.domain.run import Intent, Lane

ANALYSIS = "analysis_" + "0" * 24


def _req(text: str = "", **kwargs) -> RouteRequest:
    return RouteRequest(text=text, **kwargs)


# --- the substring false positives ------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Summarise the executive summary of this analysis.",
        "Tóm tắt phần executive summary giúp tôi.",
    ],
)
def test_executive_summary_is_not_a_request_to_execute_something(text: str) -> None:
    decision = route(_req(text, has_active_analysis=True, analysis_id=ANALYSIS))
    assert decision.intent is not Intent.OUT_OF_SCOPE
    assert matched_terms(text, OUT_OF_SCOPE_TERMS) == ()


def test_contributing_factors_is_not_an_attribution_request() -> None:
    text = "What are the contributing factors to the uncertainty in this score?"
    decision = route(_req(text, has_active_analysis=True, analysis_id=ANALYSIS))
    # Both an attribution phrase and a plain fallback question land on
    # DECISION_SUPPORT now (ADR 0010); the reason code is what proves this
    # did not fire as an attribution request.
    assert decision.intent is Intent.DECISION_SUPPORT
    assert decision.decision.reason_codes == ("question_about_active",)
    assert matched_terms(text, ATTRIBUTION_TERMS) == ()


@pytest.mark.parametrize(
    "text,terms",
    [
        ("The reviewer was excited about the result.", RESEARCH_TERMS),
        ("Please recite the endpoints you covered.", RESEARCH_TERMS),
        ("Is this item flagged?", ATTRIBUTION_TERMS),
    ],
)
def test_a_term_inside_a_longer_word_does_not_match(text: str, terms) -> None:
    assert matched_terms(text, terms) == ()


# --- what must still route ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Which atoms contribute most to the hERG score?",
        "Show me the atom contributions for this endpoint.",
        "Nguyên tử nào đóng góp nhiều nhất?",
    ],
)
def test_a_real_attribution_request_still_routes(text: str) -> None:
    decision = route(_req(text, has_active_analysis=True, analysis_id=ANALYSIS))
    assert decision.intent is Intent.DECISION_SUPPORT
    assert decision.decision.confidence == "high"
    assert "attribution_phrase" in decision.decision.reason_codes


@pytest.mark.parametrize(
    "text",
    [
        "Find published studies on this compound.",
        "Is there any literature supporting this?",
        "Tìm các nghiên cứu đã công bố về chất này.",
        "Có bằng chứng nào trong y văn không?",
    ],
)
def test_a_real_research_request_still_routes(text: str) -> None:
    decision = route(_req(text, has_active_analysis=True, analysis_id=ANALYSIS))
    assert decision.intent is Intent.DECISION_SUPPORT
    assert "research_phrase" in decision.decision.reason_codes


@pytest.mark.parametrize(
    "text",
    [
        "Build me a report for this molecule.",
        "Generate a report I can download.",
        "Tạo báo cáo đầy đủ cho chất này.",
    ],
)
def test_a_real_report_build_still_routes(text: str) -> None:
    decision = route(_req(text, has_active_analysis=True, analysis_id=ANALYSIS))
    assert decision.intent is Intent.BUILD_REPORT
    assert decision.lane is Lane.MIXED


@pytest.mark.parametrize(
    "text",
    [
        "Execute this shell command for me.",
        "Please diagnose my patient from these numbers.",
        "Kê đơn giúp tôi.",
    ],
)
def test_out_of_scope_still_routes(text: str) -> None:
    decision = route(_req(text, has_active_analysis=True, analysis_id=ANALYSIS))
    assert decision.intent is Intent.OUT_OF_SCOPE
    assert decision.decision.matched


# --- negation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Answer from the prediction only — do not search the literature.",
        "Trả lời dựa trên dự đoán thôi, đừng tìm kiếm tài liệu.",
    ],
)
def test_a_negated_request_is_not_treated_as_a_request(text: str) -> None:
    """Naming a thing while refusing it is not asking for it.

    Spending a literature search on "do not search the literature" is wrong
    twice over — it answers the opposite question and it bills a provider for
    doing so.
    """
    decision = route(_req(text, has_active_analysis=True, analysis_id=ANALYSIS))
    assert decision.intent is Intent.DECISION_SUPPORT
    assert decision.decision.reason_codes == ("question_about_active",)
    assert decision.decision.matched == ()


def test_negation_only_reaches_backwards_a_few_words() -> None:
    """"Search the literature, I do not trust the prediction" still searches.

    A negator anywhere in a sentence would suppress a request it has nothing
    to do with, which trades one false positive for another.
    """
    decision = route(
        _req(
            "Search the literature; I do not trust this prediction.",
            has_active_analysis=True,
            analysis_id=ANALYSIS,
        )
    )
    assert decision.intent is Intent.DECISION_SUPPORT
    assert "research_phrase" in decision.decision.reason_codes


def test_a_term_mentioned_twice_counts_if_either_use_is_a_real_request() -> None:
    decision = route(
        _req(
            "Do not cite reviews, but do find primary literature.",
            has_active_analysis=True,
            analysis_id=ANALYSIS,
        )
    )
    assert decision.intent is Intent.DECISION_SUPPORT
    assert "research_phrase" in decision.decision.reason_codes


# --- a molecule and a question ----------------------------------------------


def test_a_bare_molecule_is_an_analysis() -> None:
    decision = route(_req(molecule_smiles="CCO"))
    assert decision.intent is Intent.ANALYSIS
    assert decision.decision.confidence == "medium"
    assert decision.decision.reason_codes == ("bare_molecule",)


def test_a_molecule_with_a_question_snapshots_first() -> None:
    decision = route(_req("Is this likely to block hERG?", molecule_smiles="CCO"))
    assert decision.intent is Intent.DECISION_SUPPORT
    assert decision.needs_snapshot_first is True
    assert decision.decision.reason_codes == ("molecule_with_question",)


def test_a_molecule_with_a_research_request_snapshots_first() -> None:
    decision = route(_req("Find literature about this.", molecule_smiles="CCO"))
    assert decision.intent is Intent.DECISION_SUPPORT
    assert "research_phrase" in decision.decision.reason_codes
    assert decision.needs_snapshot_first is True


# --- with and without an active analysis ------------------------------------


def test_a_question_with_no_subject_clarifies_rather_than_guessing() -> None:
    decision = route(_req("Which atoms contribute most?"))
    assert decision.intent is Intent.CLARIFICATION_REQUIRED
    assert decision.decision.confidence == "low"
    assert decision.decision.required_context == ("analysis_id_or_smiles",)
    assert "subject_missing" in decision.decision.reason_codes


def test_an_empty_request_clarifies() -> None:
    decision = route(_req(""))
    assert decision.intent is Intent.CLARIFICATION_REQUIRED
    assert decision.decision.reason_codes == ("empty_request",)
    assert set(decision.decision.required_context) == {"molecule_smiles", "question_text"}


def test_text_with_no_molecule_asks_for_one() -> None:
    decision = route(_req("Tell me about toxicity."))
    assert decision.intent is Intent.CLARIFICATION_REQUIRED
    assert decision.decision.clarification_options == ("submit_smiles",)


# --- the hint is a request, not a decision ----------------------------------


def test_an_explicit_hint_wins_over_the_text() -> None:
    decision = route(
        _req(
            "Find published studies about this.",
            intent_hint="build_report",
            has_active_analysis=True,
            analysis_id=ANALYSIS,
        )
    )
    assert decision.intent is Intent.BUILD_REPORT
    assert decision.decision.reason_codes == ("explicit_hint",)
    assert decision.decision.hint_honoured is True


def test_an_unhonoured_hint_is_recorded_rather_than_hidden() -> None:
    """A hint for an intent that still needs a subject does not get honoured,
    and the decision says so instead of the UI and the backend silently
    disagreeing."""
    decision = route(_req("", intent_hint="build_report"))
    assert decision.intent is Intent.CLARIFICATION_REQUIRED
    assert decision.decision.requested_hint == "build_report"
    assert decision.decision.hint_honoured is False


def test_an_unknown_hint_is_flagged() -> None:
    decision = route(
        _req("What is the hERG score?", intent_hint="teleport", has_active_analysis=True)
    )
    assert "unknown_intent_hint" in decision.decision.reason_codes
    assert decision.decision.hint_honoured is False


def test_every_decision_names_its_router_version() -> None:
    for request in (
        _req("CCO", molecule_smiles="CCO"),
        _req("Which atoms contribute?", has_active_analysis=True),
        _req(""),
    ):
        assert route(request).decision.router_version == ROUTER_VERSION


def test_the_decision_round_trips_to_a_dict() -> None:
    decision = route(_req(molecule_smiles="CCO")).decision.to_dict()
    assert decision["intent"] == "analysis"
    assert decision["router_version"] == ROUTER_VERSION
    assert set(decision) == {
        "intent", "confidence", "reason_codes", "matched", "required_context",
        "clarification_options", "router_version", "requested_hint", "hint_honoured",
    }


# --- the matcher itself ------------------------------------------------------


def test_decomposed_and_composed_vietnamese_are_the_same_word() -> None:
    composed = "tài liệu"
    decomposed = "tài liệu"
    assert normalize(composed) == normalize(decomposed)
    assert contains_phrase(tokenize(decomposed), "tài liệu")


def test_a_phrase_matches_only_as_contiguous_words() -> None:
    tokens = tokenize("build a very long report")
    assert contains_phrase(tokens, "build a very long report")
    assert not contains_phrase(tokens, "build a report")


def test_punctuation_does_not_prevent_a_match() -> None:
    assert contains_phrase(tokenize("Find the literature, please."), "literature")


def test_an_empty_phrase_never_matches() -> None:
    assert not contains_phrase(tokenize("anything"), "")
