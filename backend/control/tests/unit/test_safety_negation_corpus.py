"""P1-5: a limitation that denies a claim is not that claim.

The hard safety gates have to refuse "this compound is safe for humans" and
must not refuse "this is not a clinical risk" — the second sentence is the
product doing its job. The negation handling this needs already exists in
``prohibited_claims``; what did not exist is a bilingual corpus pinning both
halves, so a future pattern tightened against a false negative cannot quietly
reintroduce the false positive.

Every row here is a sentence a real answer would plausibly contain.
"""
from __future__ import annotations

import pytest

from toxagent.validation.prohibited_claims import validate_answer_markdown

# --- must pass: a limitation, a denial, a scoped statement ------------------

ACCEPTABLE = [
    # Vietnamese negations, the case the audit named.
    "Kết quả này không phải nguy cơ lâm sàng.",
    "Đây không phải là bằng chứng về độc tính lâm sàng.",
    "Mô hình không đưa ra kết luận an toàn cho người.",
    "Sàng lọc không cung cấp điểm độc tính tổng thể.",
    # English negations.
    "This is not a clinical toxicity finding.",
    "Screening does not provide an overall toxicity score.",
    "The model cannot say whether this compound is safe.",
    "No aggregate risk score is produced by this product.",
    # Ordinary scoped statements that merely contain a loaded word.
    "The hERG endpoint estimates channel block, not clinical outcome.",
    "Attribution shows model sensitivity, not a chemical mechanism.",
    "Xác suất chẹn kênh hERG là 0,731.",
]

# --- must be refused: the assertions the product never makes ---------------

PROHIBITED = [
    ("The compound is safe for humans.", "safety_verdict_out_of_scope"),
    ("Hợp chất này an toàn cho người.", "safety_verdict_out_of_scope"),
    ("This molecule is considered safe.", "safety_verdict_out_of_scope"),
    ("The overall toxicity risk is low.", "aggregate_verdict_present"),
    ("An aggregate risk score of 0.4 was computed.", "aggregate_verdict_present"),
]


@pytest.mark.parametrize("text", ACCEPTABLE)
def test_a_denial_or_a_limitation_is_not_a_prohibited_claim(text: str) -> None:
    violations = validate_answer_markdown(text)
    assert violations == [], f"{text!r} was refused: {[v.code for v in violations]}"


@pytest.mark.parametrize("text,code", PROHIBITED)
def test_the_hard_gate_still_refuses_the_assertion(text: str, code: str) -> None:
    codes = [v.code for v in validate_answer_markdown(text)]
    assert code in codes, f"{text!r} was accepted; got {codes}"


def test_a_denial_inside_a_longer_answer_does_not_poison_a_real_violation() -> None:
    """Both sentences in one answer: the denial passes, the assertion does not."""
    text = (
        "This is not a clinical risk assessment. "
        "The compound is safe for humans."
    )
    codes = [v.code for v in validate_answer_markdown(text)]
    assert "safety_verdict_out_of_scope" in codes


def test_a_bare_negation_does_not_excuse_a_safety_verdict() -> None:
    """The exemption is for the product declining to conclude, not for any
    sentence containing "no": "there is no doubt the compound is safe" is the
    exact claim this gate exists to refuse."""
    codes = [v.code for v in validate_answer_markdown("There is no doubt the compound is safe.")]
    assert "safety_verdict_out_of_scope" in codes


def test_endpoint_substitution_is_caught_on_the_claim_that_makes_it() -> None:
    """Scoped to a claim rather than to the prose, because the gate needs the
    field_path to know that an hERG number is being described as clinical
    toxicity. Markdown alone does not carry that."""
    from toxagent.validation.prohibited_claims import validate_claim_wording
    from toxagent.validation.answer.candidate_wire import ClaimCandidate

    claim = ClaimCandidate(
        claim_id="clm_" + "0" * 32,
        kind="scientific",
        text="This compound is clinically toxic.",
        observation_id="obs_" + "0" * 32,
        field_path="predictions.herg.probability_blocker",
    )
    assert [v.code for v in validate_claim_wording(claim)] == [
        "endpoint_substitution_language"
    ]

    denied = ClaimCandidate(
        claim_id="clm_" + "1" * 32,
        kind="scientific",
        text="An hERG signal is not clinical toxicity.",
        observation_id="obs_" + "0" * 32,
        field_path="predictions.herg.probability_blocker",
    )
    assert validate_claim_wording(denied) == []


def test_the_negation_window_does_not_reach_across_a_sentence_boundary() -> None:
    """"...is not established. The compound is safe" must still be refused —
    the negation belongs to the previous sentence."""
    text = "Clinical toxicity is not established. The compound is safe."
    codes = [v.code for v in validate_answer_markdown(text)]
    assert "safety_verdict_out_of_scope" in codes
