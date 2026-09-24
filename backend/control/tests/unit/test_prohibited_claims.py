"""Prohibited wording patterns (plan sections 2.2, 9.2, 16.5)."""
from __future__ import annotations

from toxagent.domain.ids import new_id
from toxagent.validation.prohibited_claims import (
    validate_answer_markdown,
    validate_claim_wording,
    validate_no_hitcount_severity,
)
from toxagent.validation.wire import ClaimCandidate


def claim(text: str, **overrides) -> ClaimCandidate:
    defaults = dict(claim_id=new_id("clm"), kind="scientific", text=text)
    defaults.update(overrides)
    return ClaimCandidate(**defaults)


def test_a_bare_safety_verdict_in_the_answer_is_flagged():
    result = validate_answer_markdown("Based on this, the compound is safe for use.")
    assert "safety_verdict_out_of_scope" in [v.code for v in result]


def test_an_aggregate_score_mention_is_flagged():
    result = validate_answer_markdown("The overall toxicity risk is moderate.")
    assert "aggregate_verdict_present" in [v.code for v in result]


def test_ordinary_scientific_prose_is_not_flagged():
    result = validate_answer_markdown(
        "The predicted hERG blockade probability is 0.731, above the model's default threshold."
    )
    assert result == []


def test_herg_claim_describing_clinical_toxicity_is_endpoint_substitution():
    result = validate_claim_wording(
        claim("This indicates clinical toxicity.", field_path="predictions.herg.probability_blocker")
    )
    assert [v.code for v in result] == ["endpoint_substitution_language"]


def test_clintox_claim_describing_herg_is_endpoint_substitution():
    result = validate_claim_wording(
        claim(
            "This is a cardiotoxicity signal via hERG.",
            field_path="predictions.clintox.probability_clinical_toxicity",
        )
    )
    assert [v.code for v in result] == ["endpoint_substitution_language"]


def test_herg_claim_mentioning_cardiotoxicity_itself_is_not_flagged():
    """Cardiotoxicity/channel-block language is what hERG *is*; it is only a
    problem when paired with clinical-trial toxicity wording."""
    result = validate_claim_wording(
        claim("This is a cardiotoxicity/channel-blockade liability signal.",
              field_path="predictions.herg.probability_blocker")
    )
    assert result == []


def test_applicability_described_as_in_distribution_is_flagged():
    result = validate_claim_wording(
        claim("The molecule is in-distribution.", field_path="applicability.status")
    )
    assert [v.code for v in result] == ["applicability_overinterpreted"]


def test_attribution_described_as_mechanistic_proof_is_flagged():
    result = validate_claim_wording(
        claim(
            "This attribution proves the mechanism of toxicity.",
            field_path="attribution.herg.tokens",
        )
    )
    assert [v.code for v in result] == ["attribution_overinterpreted"]


def test_a_hitcount_used_as_severity_is_flagged():
    claims = [claim("5 active assays indicate this compound is more severe.")]
    result = validate_no_hitcount_severity(claims, "")
    assert [v.code for v in result] == ["hitcount_as_severity"]


def test_a_bare_assay_count_with_no_severity_language_is_fine():
    claims = [claim("3 of the 12 Tox21 assays are active for this molecule.")]
    result = validate_no_hitcount_severity(claims, "")
    assert result == []


def test_vietnamese_safety_wording_is_also_caught():
    result = validate_answer_markdown("Chất này an toàn khi sử dụng.")
    assert "safety_verdict_out_of_scope" in [v.code for v in result]


def test_denying_an_aggregate_score_is_not_flagged():
    """audit_5_9.md §4.7: 'does not provide an overall toxicity score' still
    contains the literal phrase 'overall toxicity', but the sentence denies
    the aggregate verdict rather than asserting it."""
    result = validate_answer_markdown(
        "This deployment does not provide an overall toxicity score."
    )
    assert result == []


def test_denying_an_aggregate_score_in_vietnamese_is_not_flagged():
    result = validate_answer_markdown(
        "Kết quả này không cung cấp mức độ độc tính tổng."
    )
    assert result == []


def test_an_aggregate_score_asserted_after_a_negated_clause_is_still_flagged():
    """The negation guard only covers a short window immediately before the
    match; it must not blanket-suppress every aggregate mention once any
    negation word appears anywhere earlier in the text."""
    result = validate_answer_markdown(
        "The report does not include attribution. The overall toxicity risk is high."
    )
    assert "aggregate_verdict_present" in [v.code for v in result]


def test_denying_clinical_toxicity_for_an_herg_claim_is_not_flagged():
    """audit_5_9.md §4.7: the same negation-blindness applies to
    _CLINICAL_OVERREACH — a claim correctly stating hERG does not establish
    clinical toxicity must not be flagged as endpoint substitution."""
    result = validate_claim_wording(
        claim(
            "This result does not establish clinical toxicity in patients.",
            field_path="predictions.herg.probability_blocker",
        )
    )
    assert result == []


def test_asserting_clinical_toxicity_for_an_herg_claim_is_still_flagged():
    result = validate_claim_wording(
        claim(
            "This hERG result demonstrates clinical toxicity.",
            field_path="predictions.herg.probability_blocker",
        )
    )
    assert [v.code for v in result] == ["endpoint_substitution_language"]
