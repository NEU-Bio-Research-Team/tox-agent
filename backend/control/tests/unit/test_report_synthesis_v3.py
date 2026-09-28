"""The narrow synthesis schema, the v3 compiler, and what they refuse (PR-11).

The audit's report was assembled by the model: it gathered its own facts, wrote
each into however many sections mentioned it, chose its own limitations and
declared its own gaps. Two sections disagreed about one explanation, and a
limitation described a literature search that never ran.

The gate for this PR is that the same artifact cannot be published. These tests
drive the sanitized fixture through the real compiler and the real validator.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.support.audit_fixtures import CCO_ATTRIBUTION, REPORT_CONTRADICTION, load
from toxagent.application.explanation.service import extract_highlights
from toxagent.domain.report import (
    ExplanationPackage,
    ExplanationStatus,
)
from toxagent.report.synthesis_compiler import (
    COMPILER_VERSION,
    compile_report,
    substitute_facts,
)
from toxagent.report.fact_bundle import assemble
from toxagent.validation.report.semantics import EvidenceSituation
from toxagent.report.synthesis_validator import validate_compiled_report
from toxagent.validation.report.synthesis_wire import ReportSynthesisV3

BUILD = "rpb_" + "a" * 32
ANALYSIS = "ana_" + "b" * 32
OBSERVATION = "obs_" + "c" * 32
EXPLANATION = "xpl_" + "d" * 32

PREDICTIONS = {
    "herg": {
        "probability_blocker": 0.0271,
        "label": "non_blocker",
        "threshold": 0.5,
        "threshold_source": "vendor_default",
        "model_id": "herg-chemberta-v3",
    }
}


def _package(status=ExplanationStatus.COMPLETED) -> ExplanationPackage:
    """The audit's CCO explanation: three negative contributors, 35.84% on
    sequence markers."""
    return ExplanationPackage(
        explanation_id=EXPLANATION,
        observation_id=OBSERVATION,
        endpoint="herg",
        task=None,
        method="integrated-gradients",
        status=status,
        highlights=extract_highlights(load(CCO_ATTRIBUTION)["payload"]),
    )


def _bundle(**overrides):
    payload = {
        "report_build_id": BUILD,
        "analysis_id": ANALYSIS,
        "predictions": PREDICTIONS,
        "served_endpoints": ["herg"],
        "selected_endpoints": ["herg"],
        "substance": {"canonical_smiles": "CCO", "preferred_name": "ethanol"},
        "explanations": [_package()],
        "observation_ids": {"herg": OBSERVATION},
        "required_limitations": [
            "uncalibrated_probability",
            "attribution_not_causality",
            "screening_not_safety_assessment",
        ],
        "policy": {"include_external_evidence": False},
    }
    payload.update(overrides)
    return assemble(**payload)


NARRATIVE_SECTIONS = (
    "executive_summary",
    "substance_profile",
    "explanation_and_visuals",
    "external_evidence",
    "integrated_interpretation",
    "conclusions",
    "recommendations",
)


def _synthesis(bundle, *, bodies: dict[str, str] | None = None, **overrides):
    bodies = bodies or {}
    probability = bundle.by_path()["predictions.herg.probability_blocker"].id
    payload = {
        "report_build_id": BUILD,
        "title": "hERG screening report for ethanol",
        "sections": [
            {
                "section_id": section_id,
                "heading": section_id.replace("_", " ").title(),
                "prose_markdown": bodies.get(
                    section_id, f"Narrative for {section_id}."
                ),
                "basis_fact_ids": [probability] if section_id == "executive_summary" else [],
            }
            for section_id in NARRATIVE_SECTIONS
        ],
        "conclusions": [
            {
                "local_ref": "c1",
                "text": "Predicted hERG liability is low under this model.",
                "basis_fact_ids": [probability],
                "endpoint": "herg",
            }
        ],
        "recommendations": [
            {
                "local_ref": "r1",
                "text": "Confirm with an in vitro patch clamp assay.",
                "basis_fact_ids": [probability],
                "action_category": "in_vitro_assay",
                "priority": "medium",
                "rationale": "A screening probability is not a measurement.",
            }
        ],
    }
    payload.update(overrides)
    return ReportSynthesisV3(**payload)


def _compile(bundle, synthesis, **kwargs):
    result = compile_report(bundle=bundle, synthesis=synthesis, **kwargs)
    assert result.ok, [v.code for v in result.violations]
    return result.report


# --- the gate: the audit artifact cannot be published ------------------------


def test_the_audit_contradiction_is_refused_through_the_v3_path() -> None:
    fixture = load(REPORT_CONTRADICTION)
    bodies = {
        section["section_id"]: section["body_markdown"]
        for section in fixture["sections"]
        if section["section_id"] in NARRATIVE_SECTIONS
    }
    bundle = _bundle()
    report = _compile(bundle, _synthesis(bundle, bodies=bodies))
    violations = validate_compiled_report(
        report,
        bundle=bundle,
        explanations={EXPLANATION: _package()},
        situation=EvidenceSituation(requested=False, search_performed=False),
    )
    codes = {violation.code for violation in violations}
    assert "explanation_summary_contradicted" in codes
    assert any("executive_summary" in v.path for v in violations)


def test_the_limitation_that_described_an_unperformed_search_cannot_be_written() -> None:
    """The model no longer has a limitations field at all. The compiler writes
    the wording, from what the build actually did."""
    bundle = _bundle()
    report = _compile(bundle, _synthesis(bundle))
    evidence_scope = [
        item for item in report.limitations if item["code"] == "evidence_scope_limited"
    ]
    assert evidence_scope == []  # not required: this build asked for no evidence
    with pytest.raises(ValidationError):
        _synthesis(
            bundle,
            sections=[
                {
                    "section_id": "limitations",
                    "heading": "Limitations",
                    "prose_markdown": "The search covered one provider.",
                }
            ],
        )


def test_a_build_that_searched_and_one_that_did_not_get_different_wording() -> None:
    searched = _bundle(
        required_limitations=["evidence_scope_limited"],
        policy={"include_external_evidence": True},
    )
    did_not = _bundle(required_limitations=["evidence_scope_limited"])
    with_search = _compile(
        searched, _synthesis(searched), search_performed=True
    ).limitations[0]["text"]
    without = _compile(did_not, _synthesis(did_not), search_performed=False).limitations[0][
        "text"
    ]
    assert "provider returned" in with_search
    assert "No external literature was consulted" in without
    assert with_search != without


# --- the model cannot write a number -----------------------------------------


def test_a_value_is_a_placeholder_the_server_renders() -> None:
    bundle = _bundle()
    probability = bundle.by_path()["predictions.herg.probability_blocker"].id
    report = _compile(
        bundle,
        _synthesis(
            bundle,
            bodies={
                "executive_summary": f"The hERG blocker probability is {{{{{probability}}}}}."
            },
        ),
    )
    summary = next(s for s in report.sections if s.section_id == "executive_summary")
    assert summary.body_markdown == "The hERG blocker probability is 0.027."
    assert summary.fact_ids == (probability,)


def test_one_placeholder_renders_the_same_string_in_every_section() -> None:
    """The structural half of P0-2: two sections cannot state two numbers for
    one fact, because neither section holds a number."""
    bundle = _bundle()
    probability = bundle.by_path()["predictions.herg.probability_blocker"].id
    placeholder = f"{{{{{probability}}}}}"
    report = _compile(
        bundle,
        _synthesis(
            bundle,
            bodies={
                "executive_summary": f"Probability {placeholder}.",
                "integrated_interpretation": f"Still {placeholder}.",
            },
        ),
    )
    rendered = [
        section.body_markdown
        for section in report.sections
        if section.section_id in ("executive_summary", "integrated_interpretation")
    ]
    assert rendered == ["Probability 0.027.", "Still 0.027."]


def test_a_number_typed_directly_into_prose_is_refused() -> None:
    bundle = _bundle()
    report = _compile(
        bundle,
        _synthesis(
            bundle, bodies={"executive_summary": "The probability is 0.99 for hERG."}
        ),
    )
    violations = validate_compiled_report(report, bundle=bundle)
    assert "unreferenced_measurement" in {v.code for v in violations}
    assert any("0.99" in str(v.actual) for v in violations)


def test_a_number_that_happens_to_match_a_fact_is_allowed() -> None:
    """It is the same string the server would have rendered, so nothing about
    it is ungrounded — refusing it would be pedantry, not safety."""
    bundle = _bundle()
    report = _compile(
        bundle, _synthesis(bundle, bodies={"executive_summary": "It is 0.027."})
    )
    codes = {v.code for v in validate_compiled_report(report, bundle=bundle)}
    assert "unreferenced_measurement" not in codes


def test_a_bare_integer_in_prose_is_not_a_smuggled_prediction() -> None:
    bundle = _bundle()
    report = _compile(
        bundle,
        _synthesis(bundle, bodies={"executive_summary": "Three atoms were examined."}),
    )
    codes = {v.code for v in validate_compiled_report(report, bundle=bundle)}
    assert "unreferenced_measurement" not in codes


def test_a_placeholder_for_a_fact_that_does_not_exist_fails_compilation() -> None:
    bundle = _bundle()
    ghost = "fct_" + "9" * 32
    result = compile_report(
        bundle=bundle,
        synthesis=_synthesis(
            bundle, bodies={"executive_summary": f"Probability {{{{{ghost}}}}}."}
        ),
    )
    assert not result.ok
    assert {v.code for v in result.violations} == {"fact_reference_unresolved"}


def test_an_unresolved_basis_fact_fails_compilation() -> None:
    bundle = _bundle()
    result = compile_report(
        bundle=bundle,
        synthesis=_synthesis(
            bundle,
            conclusions=[
                {
                    "local_ref": "c1",
                    "text": "Low liability.",
                    "basis_fact_ids": ["fct_" + "8" * 32],
                    "endpoint": "herg",
                }
            ],
        ),
    )
    assert not result.ok
    assert "basis_fact_unresolved" in {v.code for v in result.violations}


def test_substitution_reports_what_it_could_not_resolve() -> None:
    text, used, missing = substitute_facts("a {{fct_" + "1" * 32 + "}} b", {})
    assert used == ()
    assert missing == ("fct_" + "1" * 32,)
    assert "{{fct_" in text  # left visible rather than rendered as an empty gap


# --- the schema is narrow by construction ------------------------------------


def test_the_synthesis_has_no_field_for_a_limitation_or_a_gap() -> None:
    fields = set(ReportSynthesisV3.model_fields)
    assert "limitations" not in fields
    assert "gaps" not in fields
    assert "claims" not in fields
    assert "tables" not in fields
    assert "figures" not in fields


@pytest.mark.parametrize(
    "section_id", ["predictor_results", "limitations", "references", "provenance_appendix"]
)
def test_a_compiled_section_cannot_be_written_by_the_model(section_id: str) -> None:
    """Refused by the section-id type before the model validator is even
    reached — the narrowest refusal available, and the earliest."""
    with pytest.raises(ValidationError) as raised:
        ReportSynthesisV3(
            report_build_id=BUILD,
            title="x",
            sections=[{"section_id": section_id, "heading": "x", "prose_markdown": "y"}],
        )
    assert "section_id" in str(raised.value)


def test_a_section_written_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="written more than once"):
        ReportSynthesisV3(
            report_build_id=BUILD,
            title="x",
            sections=[
                {"section_id": "executive_summary", "heading": "a", "prose_markdown": "1"},
                {"section_id": "executive_summary", "heading": "b", "prose_markdown": "2"},
            ],
        )


def test_a_conclusion_must_state_its_scope() -> None:
    with pytest.raises(ValidationError, match="must name its endpoint"):
        ReportSynthesisV3(
            report_build_id=BUILD,
            title="x",
            conclusions=[
                {"local_ref": "c1", "text": "Low.", "basis_fact_ids": ["fct_" + "0" * 32]}
            ],
        )


def test_an_integrated_conclusion_is_not_about_one_endpoint() -> None:
    with pytest.raises(ValidationError, match="not about one endpoint"):
        ReportSynthesisV3(
            report_build_id=BUILD,
            title="x",
            conclusions=[
                {
                    "local_ref": "c1",
                    "text": "Low.",
                    "basis_fact_ids": ["fct_" + "0" * 32],
                    "endpoint": "herg",
                    "is_integrated": True,
                }
            ],
        )


def test_a_synthesis_for_another_build_is_refused() -> None:
    bundle = _bundle()
    other = _synthesis(bundle)
    other = other.model_copy(update={"report_build_id": "rpb_" + "e" * 32})
    result = compile_report(bundle=bundle, synthesis=other)
    assert {v.code for v in result.violations} == {"synthesis_build_mismatch"}


# --- what the compiler owns --------------------------------------------------


def test_the_compiler_writes_the_predictor_results_table() -> None:
    bundle = _bundle()
    report = _compile(bundle, _synthesis(bundle))
    section = next(s for s in report.sections if s.section_id == "predictor_results")
    assert section.compiled is True
    assert "| herg |" in section.body_markdown
    assert "0.027" in section.body_markdown
    assert "does not produce a combined score" in section.body_markdown


def test_the_report_carries_every_required_section_in_order() -> None:
    from toxagent.domain.report import REQUIRED_SECTION_IDS

    report = _compile(_bundle(), _synthesis(_bundle()))
    assert tuple(s.section_id for s in report.sections) == REQUIRED_SECTION_IDS


def test_the_provenance_appendix_names_the_compiler() -> None:
    report = _compile(_bundle(), _synthesis(_bundle()))
    section = next(s for s in report.sections if s.section_id == "provenance_appendix")
    assert COMPILER_VERSION in section.body_markdown


def test_the_content_hash_ignores_renderings_and_notices_numbers() -> None:
    bundle = _bundle()
    first = _compile(bundle, _synthesis(bundle))
    same = _compile(bundle, _synthesis(bundle))
    assert first.content_sha256() == same.content_sha256()

    changed_bundle = _bundle(
        predictions={"herg": {**PREDICTIONS["herg"], "probability_blocker": 0.9}}
    )
    changed = _compile(changed_bundle, _synthesis(changed_bundle))
    assert changed.content_sha256() != first.content_sha256()


# --- coverage and scope gates ------------------------------------------------


def test_an_unserved_endpoint_must_carry_a_gap() -> None:
    bundle = _bundle(selected_endpoints=["herg", "clintox"], served_endpoints=["herg"])
    report = _compile(bundle, _synthesis(bundle))
    assert any(item.endpoint == "clintox" and not item.served for item in report.endpoint_assessments)
    codes = {v.code for v in validate_compiled_report(report, bundle=bundle)}
    assert "endpoint_unserved_without_gap" not in codes
    assert any(gap["reason"] == "endpoint_not_served" for gap in report.gaps)


def test_a_conclusion_about_an_unassessed_endpoint_is_refused() -> None:
    bundle = _bundle()
    report = _compile(
        bundle,
        _synthesis(
            bundle,
            conclusions=[
                {
                    "local_ref": "c1",
                    "text": "ClinTox risk is low.",
                    "basis_fact_ids": [
                        bundle.by_path()["predictions.herg.probability_blocker"].id
                    ],
                    "endpoint": "clintox",
                }
            ],
        ),
    )
    codes = {v.code for v in validate_compiled_report(report, bundle=bundle)}
    assert "conclusion_endpoint_not_assessed" in codes


def test_a_missing_required_limitation_is_refused() -> None:
    bundle = _bundle(required_limitations=["uncalibrated_probability"])
    from dataclasses import replace

    report = _compile(bundle, _synthesis(bundle))
    stripped = replace(report, limitations=())
    codes = {v.code for v in validate_compiled_report(stripped, bundle=bundle)}
    assert "missing_required_limitation" in codes


def test_a_safety_verdict_in_compiled_prose_is_refused() -> None:
    bundle = _bundle()
    report = _compile(
        bundle,
        _synthesis(bundle, bodies={"executive_summary": "This compound is safe for humans."}),
    )
    codes = {v.code for v in validate_compiled_report(report, bundle=bundle)}
    assert "safety_verdict_out_of_scope" in codes


def test_interpreting_evidence_the_build_never_promoted_is_refused() -> None:
    bundle = _bundle()
    report = _compile(
        bundle,
        _synthesis(
            bundle,
            evidence_interpretations=[
                {
                    "evidence_id": "evd_" + "7" * 32,
                    "endpoint": "herg",
                    "relation": "supports",
                    "summary": "A paper I remember.",
                }
            ],
        ),
    )
    codes = {v.code for v in validate_compiled_report(report, bundle=bundle)}
    assert "evidence_not_citable" in codes


# --- a clean report passes ---------------------------------------------------


def test_a_well_formed_report_passes_every_gate() -> None:
    """The gates have to let a correct report through, or they are a wall."""
    bundle = _bundle()
    probability = bundle.by_path()["predictions.herg.probability_blocker"].id
    report = _compile(
        bundle,
        _synthesis(
            bundle,
            bodies={
                "executive_summary": (
                    f"The hERG blocker probability is {{{{{probability}}}}}, below the "
                    "decision threshold for this model."
                ),
                "explanation_and_visuals": (
                    "The attribution highlights the hydroxyl end of the molecule."
                ),
                "external_evidence": "No literature search was requested for this build.",
            },
        ),
    )
    violations = validate_compiled_report(
        report,
        bundle=bundle,
        explanations={EXPLANATION: _package()},
        situation=EvidenceSituation(requested=False, search_performed=False),
    )
    assert violations == [], [v.to_dict() for v in violations]


def test_the_compiler_appends_the_canonical_coverage_sentence() -> None:
    """A model asked to write this sentence can write it wrongly — the audit's
    did. Appending it means the correct statement is present whatever the
    narrative says, and it must not trip the gate that checks wording."""
    bundle = _bundle()
    report = _compile(
        bundle,
        _synthesis(
            bundle,
            bodies={"explanation_and_visuals": "The attribution is concentrated at one end."},
        ),
    )
    section = next(s for s in report.sections if s.section_id == "explanation_and_visuals")
    assert "Attribution coverage, as computed:" in section.body_markdown
    assert bundle.explanations[0].summary_sentence in section.body_markdown
    assert "Attribution coverage" not in section.authored_markdown
    violations = validate_compiled_report(
        report, bundle=bundle, explanations={EXPLANATION: _package()}
    )
    assert "explanation_summary_contradicted" not in {v.code for v in violations}
    assert "unreferenced_measurement" not in {v.code for v in violations}
