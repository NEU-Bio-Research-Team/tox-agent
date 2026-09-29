"""Prohibited wording patterns (plan sections 2.2, 9.2, 16.5)."""
from __future__ import annotations

from toxagent.domain.ids import new_id
from toxagent.validation.prohibited_claims import (
    validate_answer_markdown,
    validate_claim_wording,
    validate_no_hitcount_severity,
)
from toxagent.validation.answer.candidate_wire import ClaimCandidate


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


def test_a_flagged_phrase_is_returned_with_its_sentence():
    """W9-02: the model's correction, and a later diagnosis, see what was flagged."""
    [violation] = validate_answer_markdown(
        "Kết quả sàng lọc cho thấy hợp chất này an toàn cho người dùng hằng ngày."
    )
    assert violation.code == "safety_verdict_out_of_scope"
    assert "an toàn" in violation.actual
    assert violation.to_dict()["actual"] == violation.actual


# --- W9-02b: the gates measured on real drafts --------------------------------
#
# The W9-A runs stored every flagged sentence (W9-02). Nearly all were the model
# *denying* a verdict. These are those sentences, verbatim excerpts; the gates
# must let them through and still refuse a verdict actually asserted.

import pytest  # noqa: E402

DENIALS = [
    "The OOD applicability warning does not mean the compound is unsafe. In this analysis, the stored status is ok.",
    "An out-of-domain status means the score is less reliable, not a determination that the compound is unsafe.",
    "It means the prediction is less reliable, not evidence that the compound is unsafe.",
    "Treat this as a screening result, not as evidence that the compound is safe or suitable for development.",
    "This is not a clinical diagnosis or a conclusion that the drug is unsafe in patients.",
    "Kết quả chỉ là tín hiệu mô hình cho endpoint hERG, không phải bằng chứng xác nhận an toàn hay một nguy cơ lâm sàng.",
    "Nó không phải điểm độc tính tổng hợp và không thể suy ra an toàn hay độc tính lâm sàng nói chung.",
    "Kết quả hiện có không đủ để kết luận an toàn hay đủ điều kiện phát triển thuốc.",
    "Đây là bước sàng lọc/xác minh, không phải đánh giá an toàn hoàn chỉnh.",
    "Xác suất này không tự chứng minh tác động tim mạch hoặc độ an toàn.",
    "Đây là tín hiệu sàng lọc, không phải nguy cơ lâm sàng hay kết luận an toàn tổng thể.",
    "An overall toxicity score cannot be provided: this analysis reports separate endpoints.",
    "The model reports hERG and Tox21 outputs as separate measurements rather than an aggregate toxicity score.",
    "A single overall toxicity score is not available for this molecule.",
    # W9-B, the image with the W9-02b gates.
    "Các nguồn về chất ô nhiễm chỉ cung cấp bối cảnh an toàn, không thay thế hồ sơ dược chất cụ thể.",
    "Cần xác minh độc lập về tín hiệu hERG và dữ liệu an toàn bổ sung.",
    "We can’t provide one scientifically valid overall toxicity score. This analysis exposes separate endpoints.",
    "Đây là tín hiệu sàng lọc; dự đoán không tự nó thay thế đánh giá an toàn.",
    # Live e2e, 2026-09-26.
    "Mô hình dự đoán blocker, nhưng chưa nên diễn giải đây là kết luận về an toàn hay nguy cơ lâm sàng.",
    "Biên này là tỷ số phơi nhiễm riêng; nó không tự nó là phán quyết an toàn.",
    "Đây là tỷ số IC50/free Cmax, không phải phán quyết an toàn.",
]

VERDICTS = [
    "The compound is safe.",
    "Based on these results the compound is safe for use.",
    "Hợp chất này an toàn cho người dùng.",
    "Kết luận: hợp chất an toàn.",
    "There is no evidence of toxicity; the compound is safe.",
    "This does not mean it is toxic, but the compound is safe for development.",
    "Không có tín hiệu hERG đáng kể, nhưng hợp chất an toàn cho người.",
    "The overall toxicity is low.",
    "Its aggregate risk is moderate, so no further testing is needed.",
    "Hợp chất có độ an toàn cao.",
    "Hợp chất được đánh giá là an toàn.",
    "Theo dữ liệu hiện có, hợp chất an toàn.",
    # The W9-B noun-phrase exemption must not reach these.
    "Hợp chất được đánh giá an toàn.",
    "Hợp chất đã được đánh giá an toàn cho người.",
    "Kết quả thử nghiệm an toàn.",
    "Hồ sơ an toàn tốt.",
    "Dữ liệu an toàn cho người dùng.",
    "Kết luận: hợp chất an toàn.",
    "Không nên lo, hợp chất an toàn.",
]


@pytest.mark.parametrize("text", DENIALS)
def test_a_denied_verdict_is_not_a_verdict(text):
    assert validate_answer_markdown(text) == [], text


@pytest.mark.parametrize("text", VERDICTS)
def test_an_asserted_verdict_is_still_refused(text):
    codes = {v.code for v in validate_answer_markdown(text)}
    assert codes & {"safety_verdict_out_of_scope", "aggregate_verdict_present"}, text


def test_a_reassurance_that_names_no_declined_conclusion_is_still_refused():
    assert validate_answer_markdown("Không phải lo, hợp chất an toàn.")
    assert validate_answer_markdown("It is not a concern at all: the compound is safe.")
