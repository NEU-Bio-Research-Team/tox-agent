"""Every fact a report may state exists once, with an id (PR-10).

The audit's report let the model gather facts, restate them section by section,
and mint the ids it referred to them by. Two sections then disagreed about one
explanation and nothing could catch it, because there was no single place either
sentence came from.
"""
from __future__ import annotations

import pytest

from tests.support.audit_fixtures import CCO_ATTRIBUTION, load
from toxagent.application.explanation.service import extract_highlights
from toxagent.domain.report import (
    ExplanationPackage,
    ExplanationStatus,
    SourceClass,
)
from toxagent.report.fact_bundle import (
    BUNDLE_SCHEMA_VERSION,
    FactKind,
    assemble,
    fact_id,
    render_value,
)

BUILD = "rpb_" + "a" * 32
ANALYSIS = "ana_" + "b" * 32
OBSERVATION = "obs_" + "c" * 32

PREDICTIONS = {
    "herg": {
        "probability_blocker": 0.7312,
        "label": "blocker",
        "threshold": 0.5,
        "threshold_source": "vendor_default",
        "model_id": "herg-chemberta-v3",
    },
    "tox21": {
        "model_id": "tox21-v2",
        "tasks": {
            "nr_ar": {"probability_activity": 0.12, "active": False, "threshold": 0.4},
            "sr_mmp": {"probability_activity": 0.88, "active": True, "threshold": 0.4},
        },
    },
}


def _bundle(**overrides):
    payload = {
        "report_build_id": BUILD,
        "analysis_id": ANALYSIS,
        "predictions": PREDICTIONS,
        "served_endpoints": ["herg", "tox21"],
        "selected_endpoints": ["herg", "tox21"],
        "selected_tox21_tasks": ["nr_ar", "sr_mmp"],
        "substance": {"canonical_smiles": "CCO", "preferred_name": "ethanol"},
        "observation_ids": {"herg": OBSERVATION},
    }
    payload.update(overrides)
    return assemble(**payload)


def _package(status=ExplanationStatus.COMPLETED):
    """The audit's CCO explanation, through the real highlight extractor."""
    highlights = extract_highlights(load(CCO_ATTRIBUTION)["payload"])
    return ExplanationPackage(
        explanation_id="xpl_" + "d" * 32,
        observation_id=OBSERVATION,
        endpoint="herg",
        task=None,
        method="integrated-gradients",
        status=status,
        highlights=highlights,
    )


# --- identity ---------------------------------------------------------------


def test_a_fact_id_is_stable_across_assemblies() -> None:
    """A stage re-run after a restart must produce the same ids, or a draft
    written against the first attempt refers to facts that no longer exist."""
    assert _bundle().by_path()["predictions.herg.label"].id == (
        _bundle().by_path()["predictions.herg.label"].id
    )


def test_two_builds_give_the_same_path_different_ids() -> None:
    other = _bundle(report_build_id="rpb_" + "e" * 32)
    assert (
        _bundle().by_path()["predictions.herg.label"].id
        != other.by_path()["predictions.herg.label"].id
    )


def test_the_id_is_derived_from_the_build_and_the_path() -> None:
    assert _bundle().by_path()["substance.canonical_smiles"].id == fact_id(
        BUILD, "substance.canonical_smiles"
    )


def test_resolving_reports_what_did_not_resolve() -> None:
    bundle = _bundle()
    known = bundle.facts[0].id
    found, missing = bundle.resolve([known, "fct_" + "9" * 32])
    assert [fact.id for fact in found] == [known]
    assert missing == ["fct_" + "9" * 32]


# --- values and rendering ---------------------------------------------------


def test_the_server_renders_the_value_under_the_reports_locale() -> None:
    english = _bundle(language="en").by_path()["predictions.herg.probability_blocker"]
    vietnamese = _bundle(language="vi").by_path()["predictions.herg.probability_blocker"]
    assert english.rendered == "0.731"
    assert vietnamese.rendered == "0,731"
    assert english.value == vietnamese.value == 0.7312


@pytest.mark.parametrize(
    "value,kind,language,expected",
    [
        (0.7312, FactKind.NUMERIC, "en", "0.731"),
        (0.3584, FactKind.FRACTION, "en", "35.84%"),
        (0.3584, FactKind.FRACTION, "vi", "35,84%"),
        (3, FactKind.COUNT, "en", "3"),
        (True, FactKind.BOOLEAN, "en", "true"),
        (False, FactKind.BOOLEAN, "en", "false"),
        ("blocker", FactKind.CLASSIFICATION, "en", "blocker"),
        (None, FactKind.NUMERIC, "en", ""),
    ],
)
def test_rendering_is_one_rule_in_one_place(value, kind, language, expected) -> None:
    assert render_value(value, kind, language=language) == expected


def test_a_fact_carries_the_observation_it_came_from() -> None:
    fact = _bundle().by_path()["predictions.herg.probability_blocker"]
    assert fact.observation_id == OBSERVATION
    assert fact.source_class is SourceClass.PREDICTOR_FACT
    assert fact.endpoint == "herg"


def test_substance_facts_are_structure_facts_not_predictor_facts() -> None:
    fact = _bundle().by_path()["substance.canonical_smiles"]
    assert fact.source_class is SourceClass.STRUCTURE_FACT


# --- endpoints and gaps -----------------------------------------------------


def test_an_endpoint_asked_for_and_not_served_becomes_a_gap() -> None:
    bundle = _bundle(selected_endpoints=["herg", "clintox"], served_endpoints=["herg"])
    clintox = next(item for item in bundle.endpoints if item.endpoint == "clintox")
    assert clintox.served is False
    assert clintox.gap_reason == "endpoint_not_served"
    assert any(gap["reason"] == "endpoint_not_served" for gap in bundle.gaps)


def test_an_unresolved_substance_is_a_gap_not_a_silence() -> None:
    bundle = _bundle(substance=None)
    assert any(
        gap["reason"] == "compound_identity_unresolved" for gap in bundle.gaps
    )
    assert "substance.canonical_smiles" not in bundle.by_path()


def test_tox21_assays_are_separate_facts_and_never_summed() -> None:
    """Twelve independent measurements. ADR 0002: no aggregate, and a count of
    active assays is not a severity."""
    paths = set(_bundle().by_path())
    assert "predictions.tox21.nr_ar.probability_activity" in paths
    assert "predictions.tox21.sr_mmp.probability_activity" in paths
    assert not any("aggregate" in path or "total" in path for path in paths)


def test_only_the_selected_tox21_assays_become_facts() -> None:
    bundle = _bundle(selected_tox21_tasks=["nr_ar"])
    paths = set(bundle.by_path())
    assert "predictions.tox21.nr_ar.probability_activity" in paths
    assert "predictions.tox21.sr_mmp.probability_activity" not in paths


def test_a_field_the_product_has_not_agreed_to_show_is_not_a_fact() -> None:
    bundle = _bundle(
        predictions={"herg": {"probability_blocker": 0.1, "internal_logit": 42.0}}
    )
    assert "predictions.herg.internal_logit" not in bundle.by_path()


# --- explanations: the compiler owns the counts ------------------------------


def test_contributor_counts_become_facts_the_model_cannot_recount() -> None:
    """The audit's executive summary reported zero contributors and zero
    unmapped mass for this exact explanation."""
    bundle = _bundle(explanations=[_package()])
    paths = bundle.by_path()
    assert paths["explanations.herg.negative_contributor_count"].value == 3
    assert paths["explanations.herg.positive_contributor_count"].value == 0
    assert paths["explanations.herg.unmapped_importance"].value == pytest.approx(
        0.3583836537608235, abs=1e-9
    )
    assert paths["explanations.herg.special_token_importance_fraction"].rendered == "35.84%"


def test_every_explanation_has_exactly_one_summary_sentence() -> None:
    bundle = _bundle(explanations=[_package()])
    summary = bundle.explanations[0]
    assert summary.summary_sentence == summary.coverage.summary_sentence()
    assert "3 contributor(s)" in summary.summary_sentence
    assert "sequence markers" in summary.summary_sentence


def test_an_explanation_fact_points_at_its_explanation_and_observation() -> None:
    bundle = _bundle(explanations=[_package()])
    fact = bundle.by_path()["explanations.herg.negative_contributor_count"]
    assert fact.explanation_id == "xpl_" + "d" * 32
    assert fact.observation_id == OBSERVATION
    assert fact.source_class is SourceClass.EXPLANATION_FACT


def test_a_failed_explanation_becomes_a_gap() -> None:
    package = _package(status=ExplanationStatus.FAILED)
    from dataclasses import replace

    package = replace(package, figure=None, failure_reason="the explainer timed out")
    bundle = _bundle(explanations=[package])
    assert any(gap["reason"] == "explanation_failed" for gap in bundle.gaps)


def test_an_explanation_stored_before_ws06_still_gets_coverage() -> None:
    """Recomputing from the highlights beats reporting 'unknown' for an
    explanation whose numbers are right there."""
    from dataclasses import replace

    from toxagent.domain.report import ExplanationHighlights

    package = replace(
        _package(),
        highlights=ExplanationHighlights(
            negative_contributors=({"atom_index": 0},),
            unmapped_importance=0.25,
        ),
    )
    bundle = _bundle(explanations=[package])
    assert bundle.explanations[0].coverage.mapped_importance_fraction == pytest.approx(0.75)
    assert bundle.by_path()["explanations.herg.negative_contributor_count"].value == 1


# --- what the model is shown -------------------------------------------------


def test_the_model_view_carries_ids_and_renderings_not_payloads() -> None:
    """A model handed the raw prediction object can restate a number from it,
    and then the report has two sources for one fact again."""
    view = _bundle(explanations=[_package()]).to_model_view()
    assert view["schema_version"] == BUNDLE_SCHEMA_VERSION
    for fact in view["facts"]:
        assert set(fact) == {
            "fact_id", "label", "rendered", "source_class", "endpoint", "task"
        }
        assert "value" not in fact
    for explanation in view["explanations"]:
        assert "summary" in explanation
        assert "positive_contributors" not in explanation
        assert "unmapped_importance" not in explanation


def test_the_model_view_lists_the_gaps_and_the_required_limitations() -> None:
    view = _bundle(
        selected_endpoints=["herg", "clintox"],
        served_endpoints=["herg"],
        required_limitations=["uncalibrated_probability"],
    ).to_model_view()
    assert view["required_limitations"] == ["uncalibrated_probability"]
    assert any(gap["reason"] == "endpoint_not_served" for gap in view["gaps"])
