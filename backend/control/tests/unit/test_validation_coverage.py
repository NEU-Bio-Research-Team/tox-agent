"""Markdown/claim coverage (audit A01): a number or a link in the prose that
no claim actually backs must not pass silently."""
from __future__ import annotations

from toxagent.domain.ids import new_id
from toxagent.validation.answer.coverage import (
    cited_fact_values,
    validate_markdown_numeric_coverage,
    validate_no_uncited_links,
)
from toxagent.validation.answer.candidate_wire import ClaimCandidate


def claim(**overrides) -> ClaimCandidate:
    defaults = dict(
        claim_id=new_id("clm"), kind="numeric", text="x",
        observation_id=None, field_path="predictions.herg.probability_blocker",
        source_value=0.73064, rendered_value="0.731", transform="round:3",
    )
    defaults.update(overrides)
    return ClaimCandidate(**defaults)


def test_a_number_with_no_backing_claim_is_rejected():
    violations = validate_markdown_numeric_coverage(
        "The hERG probability is 99.99%.", claims=()
    )
    assert any(v.code == "unclaimed_numeric_value" for v in violations)


def test_a_number_matching_a_claims_rendered_value_is_accepted():
    violations = validate_markdown_numeric_coverage(
        "Predicted hERG blockade probability is 0.731.", claims=(claim(rendered_value="0.731"),)
    )
    assert violations == []


def test_plain_prose_integers_are_not_flagged():
    violations = validate_markdown_numeric_coverage(
        "This is candidate 1 of 2, step 3.", claims=()
    )
    assert violations == []


def test_a_percentage_with_no_backing_claim_is_rejected():
    violations = validate_markdown_numeric_coverage(
        "Roughly 12% of assays were active.", claims=()
    )
    assert any(v.code == "unclaimed_numeric_value" for v in violations)


def test_a_markdown_link_is_rejected():
    violations = validate_no_uncited_links(
        "See [this study](https://example.com/paper) for background."
    )
    assert any(v.code == "raw_link_in_answer_markdown" for v in violations)


def test_a_bare_url_is_rejected():
    violations = validate_no_uncited_links("Source: https://example.com/paper")
    assert any(v.code == "raw_link_in_answer_markdown" for v in violations)


def test_plain_prose_with_no_link_is_accepted():
    violations = validate_no_uncited_links("Predicted hERG blockade probability is 0.731.")
    assert violations == []


# --- version strings are not predictor numbers ------------------------------
#
# A live report build failed twice on this. The report profile requires the
# provenance appendix to name "analysis hash, predictor version, artifact
# hashes, model ids" — and then this check rejected the version, because no
# claim will ever carry `rendered_value == "0.1"`. The instructions demanded a
# number the validator forbade, so that one section was impossible to write and
# the build burned its single correction attempt discovering it.


def test_a_dotted_version_is_not_an_unclaimed_prediction():
    violations = validate_markdown_numeric_coverage(
        "Predictor service version 0.1.0.dev0, compiler toxagent-report-compiler-v2.",
        claims=(),
    )
    assert violations == [], [v.actual for v in violations]


def test_a_version_inside_a_hyphenated_identifier_is_not_flagged():
    violations = validate_markdown_numeric_coverage(
        "Rendered with weasyprint-66.0 from schema toxagent-report-v2.", claims=()
    )
    assert violations == [], [v.actual for v in violations]


def test_a_real_probability_in_the_appendix_is_still_caught():
    """The exclusion is for identifiers, not for the section. A bare decimal
    that reads as a measurement stays a violation wherever it appears."""
    violations = validate_markdown_numeric_coverage(
        "Threshold applied: 0.5, giving probability 0.73.", claims=()
    )
    assert {v.actual for v in violations} == {"0.5", "0.73"}


def test_a_negative_number_is_still_a_number():
    """`-0.5` must not be mistaken for the tail of a hyphenated identifier: the
    hyphen there follows a space, not a word character."""
    violations = validate_markdown_numeric_coverage(
        "The contribution shifted by -0.5 logit units.", claims=()
    )
    assert {v.actual for v in violations} == {"-0.5"}


def test_a_hash_prefix_is_not_read_as_a_number():
    violations = validate_markdown_numeric_coverage(
        "Analysis content hash 3907e013f13c1ce7, weights_sha256=aaa1.2bbb.", claims=()
    )
    assert violations == [], [v.actual for v in violations]


# --- W9-02: v2 prose numbers are checked against the claimed values ---------

from toxagent.validation.answer.coverage import faithful_rendering  # noqa: E402


def test_faithful_rendering_accepts_a_rounding_to_the_tokens_own_precision():
    for token in ("0.7999394536018372", "0.800", "0.80", "0.8", "80%", "80.0%", "0,80"):
        assert faithful_rendering(token, 0.7999394536018372), token


def test_faithful_rendering_refuses_a_different_number():
    for token in ("0.81", "0.79", "79%", "0.7998"):
        assert not faithful_rendering(token, 0.7999394536018372), token


def test_a_non_zero_value_is_never_faithfully_zero():
    """numeric-11: an inactive assay's small probability is not zero."""
    assert not faithful_rendering("0%", 0.004)
    assert not faithful_rendering("0.00", 0.004)
    assert faithful_rendering("0.0", 0.0)


def test_claimed_values_only_widen_coverage_when_passed():
    prose = "The hERG probability is 0.7999394536018372."
    assert validate_markdown_numeric_coverage(prose, claims=())
    assert not validate_markdown_numeric_coverage(
        prose, claims=(), claimed_values=(0.7999394536018372,)
    )


def _chembl_record(value: str):
    from datetime import datetime, timezone

    from toxagent.domain.evidence import EvidenceRecord, EvidenceStatus, SourceType

    record = EvidenceRecord.create(
        session_id=new_id("ses"), provider="chembl", provider_record_id=f"activity:{value}",
        source_type=SourceType.DATABASE, title="t",
        retrieved_at=datetime(2026, 9, 26, tzinfo=timezone.utc),
        normalized_facts={"standard_type": "IC50", "standard_value": value,
                          "standard_units": "nM", "pchembl_value": "7.25"},
    )
    return record.to_status(EvidenceStatus.NORMALIZED).to_status(EvidenceStatus.ACCEPTED)


def test_a_measured_value_of_a_cited_record_may_be_written_in_the_prose():
    """Live e2e, 2026-09-26: "IC50 56,0 nM" from a cited ChEMBL record was
    refused, so the answer dropped every measurement."""
    record = _chembl_record("56.0")
    cites = claim(kind="scientific", field_path=None, source_value=None, rendered_value=None,
                  transform="identity", citation_ids=[record.id])
    values = cited_fact_values(cites.citation_ids, {record.id: record})
    assert validate_markdown_numeric_coverage(
        "ChEMBL ghi IC50 56,0 nM (pIC50 7,25).", (cites,), claimed_values=values) == []


def test_a_record_nobody_cites_grounds_no_number():
    record = _chembl_record("56.0")
    uncited = cited_fact_values([], {record.id: record})
    violations = validate_markdown_numeric_coverage("IC50 56.0 nM.", (), claimed_values=uncited)
    assert [v.code for v in violations] == ["unclaimed_numeric_value"]


def test_a_decimal_comma_renders_the_claims_decimal_point():
    """Live e2e, 2026-09-26: "0,680" in Vietnamese prose for a claim rendered
    "0.680" cost the run its first draft."""
    assert validate_markdown_numeric_coverage(
        "Xác suất mô hình là 0,680.", (claim(rendered_value="0.680"),)) == []
