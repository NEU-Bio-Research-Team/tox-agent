"""Markdown/claim coverage (audit A01): a number or a link in the prose that
no claim actually backs must not pass silently."""
from __future__ import annotations

from toxagent.domain.ids import new_id
from toxagent.validation.coverage import (
    validate_markdown_numeric_coverage,
    validate_no_uncited_links,
)
from toxagent.validation.wire import ClaimCandidate


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
