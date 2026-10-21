"""Required limitations and their triggers (plan section 9.4).

A limitation is not a disclaimer bolted onto the end of an answer. Each one is
triggered by something the answer actually did: interpreting a probability,
naming the applicability status, citing an attribution, working around an
endpoint the build does not serve. The renderer may merge two of them into one
sentence; it may not drop the code, because the code is what an audit reads.
"""
from __future__ import annotations

from typing import Any, Final, Iterable, Mapping

from ..domain.answer import ClaimKind, LimitationCode

CODE = LimitationCode

TEXT: Final[Mapping[LimitationCode, Mapping[str, str]]] = {
    CODE.UNCALIBRATED_PROBABILITY: {
        "en": (
            "This probability is a model score, not a calibrated clinical risk. It ranks "
            "molecules; it does not estimate how often the effect occurs in people."
        ),
        "vi": (
            "Xác suất này là điểm số của mô hình, không phải nguy cơ lâm sàng đã hiệu chuẩn. "
            "Nó dùng để xếp hạng phân tử, không ước lượng tần suất xảy ra trên người."
        ),
    },
    CODE.APPLICABILITY_IS_RULE_BASED: {
        "en": (
            "The applicability check is a rule-based element whitelist, not a learned "
            "out-of-distribution test; 'ok' does not mean the molecule resembles the "
            "training data."
        ),
        "vi": (
            "Kiểm tra applicability là quy tắc whitelist nguyên tố, không phải bộ phát hiện "
            "ngoài phân phối đã học; 'ok' không có nghĩa phân tử giống dữ liệu huấn luyện."
        ),
    },
    CODE.ATTRIBUTION_NOT_CAUSALITY: {
        "en": (
            "Attribution shows which input tokens moved this model's score for this one "
            "endpoint. It is not evidence of a chemical mechanism."
        ),
        "vi": (
            "Attribution cho biết token đầu vào nào làm thay đổi điểm của mô hình cho đúng "
            "một endpoint. Đây không phải bằng chứng về cơ chế hóa học."
        ),
    },
    CODE.ENDPOINT_UNAVAILABLE: {
        "en": (
            "One or more requested endpoints are not served by this deployment. No other "
            "endpoint was substituted for them."
        ),
        "vi": (
            "Một hoặc nhiều endpoint được yêu cầu không được phục vụ ở deployment này. "
            "Không có endpoint nào được dùng thay thế."
        ),
    },
    CODE.EVIDENCE_SCOPE_LIMITED: {
        "en": (
            "The literature search covered one provider and the records it returned; "
            "full texts were not read. Absence of evidence here is not evidence of absence."
        ),
        "vi": (
            "Tìm kiếm tài liệu chỉ qua một provider và các bản ghi provider trả về; "
            "không đọc toàn văn. Việc không tìm thấy bằng chứng không phải bằng chứng phủ định."
        ),
    },
    CODE.SCREENING_NOT_SAFETY_ASSESSMENT: {
        "en": (
            "These are screening signals proposing what to verify next. They are not a "
            "safety assessment and carry no regulatory standing."
        ),
        "vi": (
            "Đây là tín hiệu sàng lọc để đề xuất bước kiểm chứng tiếp theo, không phải "
            "đánh giá an toàn và không có giá trị pháp lý."
        ),
    },
    CODE.ACTIVITY_INTERVAL_IS_MARGINAL_COVERAGE: {
        "en": (
            "The activity interval covers the stated share of validation residuals across "
            "many candidates. It is not a guarantee for this one candidate."
        ),
        "vi": (
            "Khoảng activity bao phủ tỷ lệ đã nêu của sai số trên tập validation, tính "
            "trên nhiều ứng viên. Đây không phải bảo đảm cho riêng ứng viên này."
        ),
    },
    CODE.PIC50_NOT_THERAPEUTIC_EFFICACY: {
        "en": (
            "A pIC50 is an estimated in-vitro potency against one target in one assay "
            "context. It says nothing about therapeutic efficacy."
        ),
        "vi": (
            "pIC50 là ước lượng hoạt lực in vitro trên một target trong một bối cảnh "
            "assay. Nó không nói gì về hiệu quả điều trị."
        ),
    },
    CODE.ACTIVITY_MODEL_NOT_ADMITTED: {
        "en": (
            "The activity model behind this estimate has not passed admission, so the "
            "estimate is not validated for decisions."
        ),
        "vi": (
            "Mô hình activity tạo ra ước lượng này chưa qua admission, nên ước lượng "
            "chưa được kiểm định để dùng cho quyết định."
        ),
    },
    CODE.POLICY_DECISION_NOT_A_MEASUREMENT: {
        "en": (
            "The prioritisation is the output of a fixed rule applied to the estimate, "
            "not a second measurement of anything."
        ),
        "vi": (
            "Mức ưu tiên là đầu ra của một quy tắc cố định áp lên ước lượng, không phải "
            "một phép đo thứ hai."
        ),
    },
    CODE.TOXICITY_NOT_RUN_NOT_NEGATIVE: {
        "en": (
            "The toxicity stage did not run for this candidate. That is the absence of a "
            "result, not a negative toxicity result."
        ),
        "vi": (
            "Bước đánh giá độc tính chưa chạy cho ứng viên này. Đó là việc chưa có kết "
            "quả, không phải kết quả độc tính âm tính."
        ),
    },
}


def text_for(code: LimitationCode, language: str = "en") -> str:
    entry = TEXT[code]
    return entry.get(language, entry["en"])


def required_for_analysis(
    *, has_probability: bool, applicability_status: str | None, unavailable_endpoints: Iterable[str]
) -> tuple[str, ...]:
    """What an observation built from a prediction must carry with it, so that a
    projection cannot hand a model a number without its caveat."""
    required: list[str] = []
    if has_probability:
        required.append(CODE.UNCALIBRATED_PROBABILITY.value)
    if applicability_status is not None:
        required.append(CODE.APPLICABILITY_IS_RULE_BASED.value)
    if tuple(unavailable_endpoints):
        required.append(CODE.ENDPOINT_UNAVAILABLE.value)
    return tuple(required)


#: Field-path fragments that mean "this claim interpreted a probability".
_PROBABILITY_MARKERS: Final[tuple[str, ...]] = (
    "probability_blocker", "probability_activity", "probability_clinical_toxicity", "probability",
)

#: Carried by an observation and required of every claim that cites it,
#: whatever field the claim read: each one says what the whole artifact is.
OBSERVATION_WIDE: Final[frozenset[str]] = frozenset({
    CODE.ENDPOINT_UNAVAILABLE.value,
    CODE.ATTRIBUTION_NOT_CAUSALITY.value,
    CODE.PIC50_NOT_THERAPEUTIC_EFFICACY.value,
    CODE.ACTIVITY_MODEL_NOT_ADMITTED.value,
    CODE.POLICY_DECISION_NOT_A_MEASUREMENT.value,
    CODE.TOXICITY_NOT_RUN_NOT_NEGATIVE.value,
})

#: Carried by an observation but about one part of it: required of a claim
#: that reads a field matching one of the fragments, or that names no field at
#: all and so cannot be shown not to have used it.
FIELD_SCOPED: Final[Mapping[str, tuple[str, ...]]] = {
    CODE.ACTIVITY_INTERVAL_IS_MARGINAL_COVERAGE.value: ("interval", "uncertainty"),
}

#: Observation kinds whose fields are not the toxicity predictor's, so the
#: predictor's path triggers do not apply to them: an activity estimate's
#: ``applicability`` is a ligand/target domain, not the element whitelist.
_NOT_TOXICITY_PREDICTOR: Final[frozenset[str]] = frozenset(
    {"activity_estimate", "prioritisation_decision", "screening_state"}
)


def required_for_answer(
    claims: Iterable[Any],
    *,
    observation_limitations: Mapping[str, tuple[str, ...]] | None = None,
    observation_kinds: Mapping[str, str] | None = None,
    cited_evidence: bool = False,
    has_recommendation: bool = False,
) -> frozenset[str]:
    """The limitation codes an answer must carry, derived from what it claimed.

    Deriving rather than declaring is the point: a model cannot avoid a caveat
    by omitting it from its candidate, because the trigger is the claim it made.

    Accepts either domain ``Claim`` objects or the pre-commit wire
    ``ClaimCandidate`` — both carry ``kind`` (as an enum or a plain string),
    ``field_path`` and ``observation_id``, and this runs against the wire shape
    *before* a candidate has passed validation, since the required set is part
    of what validation checks.
    """
    required: set[str] = set()
    observation_limitations = observation_limitations or {}
    observation_kinds = observation_kinds or {}

    for claim in claims:
        path = claim.field_path or ""
        kind = getattr(claim.kind, "value", claim.kind)
        predictor_fields = (
            observation_kinds.get(claim.observation_id or "") not in _NOT_TOXICITY_PREDICTOR
        )
        if predictor_fields:
            if kind == ClaimKind.NUMERIC.value and any(m in path for m in _PROBABILITY_MARKERS):
                required.add(CODE.UNCALIBRATED_PROBABILITY.value)
            if path.startswith("applicability"):
                required.add(CODE.APPLICABILITY_IS_RULE_BASED.value)
            if "attribution" in path or "tokens" in path:
                required.add(CODE.ATTRIBUTION_NOT_CAUSALITY.value)
        if claim.observation_id:
            # endpoint_unavailable and attribution_not_causality are genuinely
            # observation-wide — they say something about the analysis this
            # observation *is*, not about one field of it (an attribution
            # observation exists only to report per-token contribution, so any
            # claim citing it is an attribution claim, unlike an analysis
            # observation, which mixes label/probability/applicability in one
            # place). uncalibrated_probability and applicability_is_rule_based
            # stay field-triggered above, or a claim that only ever mentions the
            # hERG label would incorrectly need to disclose an applicability
            # caveat it never touched.
            #
            # Found live 2026-09-06 (progress log section 14.4): a scientific
            # claim citing an attribution observation_id with no field_path at
            # all (SCIENTIFIC is not in FIELD_BACKED_KINDS, so field_path is
            # optional) silently escaped the "attribution"/"tokens" substring
            # check above — this observation-keyed check is the actual source
            # of truth, not a substring guess at wording the model chose.
            #
            # The activity artifacts' limitations follow the same split: what
            # the artifact is (a pIC50, a rule's output, a stage that did not
            # run) is observation-wide; the interval caveat is about the
            # interval, so it is required of a claim that reads it.
            cited_limitations = observation_limitations.get(claim.observation_id, ())
            required.update(code for code in cited_limitations if code in OBSERVATION_WIDE)
            for code, fragments in FIELD_SCOPED.items():
                if code in cited_limitations and (
                    not path or any(fragment in path for fragment in fragments)
                ):
                    required.add(code)

    if cited_evidence:
        required.add(CODE.EVIDENCE_SCOPE_LIMITED.value)
    if has_recommendation:
        required.add(CODE.SCREENING_NOT_SAFETY_ASSESSMENT.value)
    return frozenset(required)
