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
