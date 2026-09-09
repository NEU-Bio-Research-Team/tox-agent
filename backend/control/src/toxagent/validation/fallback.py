"""The deterministic fallback answer (plan section 9.5).

Reached only after a model has spent both its candidate attempts and still
produced something that fails validation. The fallback is server-authored: it
contains no model text at all, only facts read directly from the prediction
observation that is already on record for this session, each with its field
path and its required limitation attached exactly as any other claim would
need. There is no reflection loop and no third attempt — plan section 9.5 is
explicit that this is where a run stops.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from ..domain.answer import Claim, ClaimKind, GroundedAnswer, Limitation, LimitationCode
from ..domain.ids import CLAIM, new_id
from ..domain.observation import Observation, ObservationKind
from .limitations import required_for_answer, text_for

#: (endpoint key in the predictor payload, probability field, label field)
_PRIMARY_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("herg", "probability_blocker", "label"),
    ("clintox", "probability_clinical_toxicity", "label"),
)

_MESSAGE = {
    "en": (
        "I could not produce a fully grounded answer to that question. Below are the "
        "recorded prediction results this session has on file; each figure links to its "
        "source observation. Please rephrase the question, or ask about one of these "
        "figures directly."
    ),
    "vi": (
        "Tôi chưa thể tạo câu trả lời có căn cứ đầy đủ cho câu hỏi này. Dưới đây là kết quả "
        "dự đoán đã ghi nhận trong phiên này; mỗi con số đều liên kết tới observation nguồn. "
        "Vui lòng diễn đạt lại câu hỏi, hoặc hỏi trực tiếp về một trong các con số này."
    ),
}

_NO_DATA_MESSAGE = {
    "en": (
        "I could not produce a fully grounded answer to that question, and this session has "
        "no recorded prediction to fall back on. Please run an analysis first."
    ),
    "vi": (
        "Tôi chưa thể tạo câu trả lời có căn cứ đầy đủ cho câu hỏi này, và phiên này chưa có "
        "kết quả dự đoán nào để tham chiếu. Vui lòng chạy phân tích trước."
    ),
}


def build_fallback_answer(
    *,
    session_id: str,
    run_id: str,
    observations: Sequence[Observation],
    language: str,
    now: datetime,
) -> GroundedAnswer:
    prediction = next(
        (o for o in observations if o.kind is ObservationKind.PREDICTION), None
    )
    if prediction is None:
        return GroundedAnswer.create(
            session_id=session_id, run_id=run_id,
            answer_markdown=_NO_DATA_MESSAGE.get(language, _NO_DATA_MESSAGE["en"]),
            claims=(), now=now, is_fallback=True,
        )

    claims = _primary_claims(prediction)
    required = required_for_answer(
        claims, observation_limitations={prediction.id: prediction.required_limitations}
    )
    limitations = tuple(
        Limitation(LimitationCode(code), text_for(LimitationCode(code), language))
        for code in sorted(required)
    )
    return GroundedAnswer.create(
        session_id=session_id, run_id=run_id,
        answer_markdown=_MESSAGE.get(language, _MESSAGE["en"]),
        claims=claims, limitations=limitations, now=now, is_fallback=True,
    )


def _primary_claims(observation: Observation) -> tuple[Claim, ...]:
    predictions = observation.canonical_payload.get("predictions", {})
    claims: list[Claim] = []
    for endpoint, probability_field, label_field in _PRIMARY_FIELDS:
        section = predictions.get(endpoint)
        if not section:
            continue
        probability_path = f"predictions.{endpoint}.{probability_field}"
        probability = float(section[probability_field])
        claims.append(
            Claim(
                claim_id=new_id(CLAIM), kind=ClaimKind.NUMERIC,
                text=f"Predicted {endpoint} {probability_field.replace('_', ' ')} is "
                     f"{probability:.3f}.",
                observation_id=observation.id, field_path=probability_path,
                source_value=probability, rendered_value=f"{probability:.3f}", transform="round:3",
            )
        )
        label_path = f"predictions.{endpoint}.{label_field}"
        label = section[label_field]
        claims.append(
            Claim(
                claim_id=new_id(CLAIM), kind=ClaimKind.CLASSIFICATION,
                text=f"The {endpoint} label is {label!r}.",
                observation_id=observation.id, field_path=label_path,
                source_value=label, rendered_value=str(label), transform="identity",
            )
        )
    return tuple(claims)
