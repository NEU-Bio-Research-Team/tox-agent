"""Compile semantic model output into deterministic grounded candidates.

The draft names sources and presentation intent only.  Claim identifiers,
canonical values, arithmetic and numeric rendering are server-owned.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Mapping

from ..domain.ids import CLAIM, new_id
from ..domain.observation import Observation
from ..validation.wire import (
    ClaimCandidate,
    GroundedAnswerCandidate,
    LimitationCandidate,
    RecommendationCandidate,
)


@dataclass(frozen=True, slots=True)
class SemanticClaim:
    kind: str
    text_template: str
    observation_id: str | None = None
    field_path: str | None = None
    transform: str = "identity"
    citation_ids: tuple[str, ...] = ()
    input_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticAnswerDraft:
    sections: tuple[tuple[str, tuple[SemanticClaim, ...]], ...]
    limitations: tuple[tuple[str, str], ...] = ()
    next_steps: tuple[str, ...] = ()


def _format(value: float, transform: str) -> str:
    if transform == "identity":
        return str(value)
    if transform.startswith(("round:", "percent:")):
        digits = int(transform.split(":", 1)[1])
        scaled = value * 100 if transform.startswith("percent:") else value
        quantum = Decimal(1).scaleb(-digits)
        text = format(Decimal(str(scaled)).quantize(quantum, rounding=ROUND_HALF_UP), f".{digits}f")
        return text + ("%" if transform.startswith("percent:") else "")
    return str(value)


class AnswerCompiler:
    def compile(self, draft: SemanticAnswerDraft, *, observations: Mapping[str, Observation]) -> GroundedAnswerCandidate:
        compiled: list[ClaimCandidate] = []
        paragraphs: list[str] = []
        for heading, semantic_claims in draft.sections:
            paragraphs.append(f"## {heading}")
            for semantic in semantic_claims:
                claim_id = new_id(CLAIM)
                source_value = None
                rendered_value = None
                observation_id = semantic.observation_id
                field_path = semantic.field_path
                input_claim_ids: list[str] = []
                if semantic.kind in {"numeric", "classification"}:
                    if not observation_id or not field_path or observation_id not in observations:
                        raise ValueError("field-backed semantic claims require a known observation and field path")
                    source_value = observations[observation_id].value_at(field_path)
                    if semantic.kind == "numeric":
                        if isinstance(source_value, bool) or not isinstance(source_value, (int, float)):
                            raise ValueError(f"{field_path!r} is not numeric")
                        rendered_value = _format(float(source_value), semantic.transform)
                elif semantic.transform in {"difference", "ratio"}:
                    if len(semantic.input_indices) != 2 or any(i >= len(compiled) for i in semantic.input_indices):
                        raise ValueError("derived claims require two earlier input indices")
                    inputs = [compiled[i] for i in semantic.input_indices]
                    values = [float(item.source_value) for item in inputs]
                    if semantic.transform == "ratio" and values[1] == 0:
                        raise ValueError("cannot compile ratio with zero denominator")
                    source_value = values[0] - values[1] if semantic.transform == "difference" else values[0] / values[1]
                    rendered_value = str(source_value)
                    input_claim_ids = [item.claim_id for item in inputs]
                text = semantic.text_template.replace("{value}", rendered_value or "")
                compiled.append(ClaimCandidate(
                    claim_id=claim_id, kind=semantic.kind, text=text,
                    observation_id=observation_id, field_path=field_path,
                    source_value=source_value, rendered_value=rendered_value,
                    transform=semantic.transform, citation_ids=list(semantic.citation_ids),
                    input_claim_ids=input_claim_ids,
                ))
                paragraphs.append(f"- {text}")
        return GroundedAnswerCandidate(
            answer_markdown="\n\n".join(paragraphs), claims=compiled,
            limitations=[LimitationCandidate(code=code, text=text) for code, text in draft.limitations],
            recommended_next_steps=[RecommendationCandidate(text=text) for text in draft.next_steps],
        )
