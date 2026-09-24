"""The answer validator (plan sections 8.4, 9): candidate in, decision out.

This is the single place all of the deterministic checks are applied, in the
order plan section 8.4 lists them: schema (already done by the wire model by
the time a candidate reaches here), reference/ACL resolution, numeric and
classification validation, required limitations, prohibited wording. Passing
all of it is what constructs the domain ``GroundedAnswer`` — there is no path
that stores a candidate without going through every check.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from ..domain.answer import (
    Claim,
    ClaimKind,
    GroundedAnswer,
    Limitation,
    LimitationCode,
    RecommendedNextStep,
)
from ..domain.errors import Violation
from ..domain.evidence import EvidenceRecord
from ..domain.observation import Observation, ObservationKind
from .citations import validate_basis, validate_citations, validate_recommendation_basis
from .classification import validate_classification
from .coverage import validate_markdown_numeric_coverage, validate_no_uncited_links
from .limitations import required_for_answer, text_for
from .numeric import validate_derived_numeric, validate_field_backed_numeric
from .prohibited_claims import (
    validate_answer_markdown,
    validate_claim_wording,
    validate_no_hitcount_severity,
)
from .wire import GroundedAnswerCandidate

_KNOWN_LIMITATION_CODES = frozenset(c.value for c in LimitationCode)
_DERIVED_TRANSFORMS = frozenset({"difference", "ratio"})
_BASIS_REQUIRED_KINDS = frozenset({"scientific", "comparison"})


@dataclass(frozen=True)
class AnswerValidationResult:
    violations: tuple[Violation, ...]
    answer: GroundedAnswer | None = None

    @property
    def ok(self) -> bool:
        return not self.violations and self.answer is not None


def validate_candidate(
    candidate: GroundedAnswerCandidate,
    *,
    session_id: str,
    run_id: str,
    candidate_generation: int,
    observations_by_id: Mapping[str, Observation],
    evidence_by_id: Mapping[str, EvidenceRecord],
    language: str,
    now: datetime,
    read_evidence_ids: frozenset[str] = frozenset(),
) -> AnswerValidationResult:
    violations: list[Violation] = []

    by_id = {c.claim_id: c for c in candidate.claims}
    duplicate_ids = {cid for cid in by_id if [c.claim_id for c in candidate.claims].count(cid) > 1}
    if duplicate_ids:
        violations.append(
            Violation("duplicate_claim_id", f"claim id(s) repeated: {sorted(duplicate_ids)}", path="claims")
        )

    for claim in candidate.claims:
        observation = observations_by_id.get(claim.observation_id) if claim.observation_id else None
        is_derived = claim.transform in _DERIVED_TRANSFORMS

        if claim.kind == "numeric":
            violations += (
                validate_derived_numeric(claim, by_id) if is_derived
                else validate_field_backed_numeric(claim, observation)
            )
        elif claim.kind == "classification":
            if is_derived:
                violations.append(
                    Violation(
                        "claim_transform_invalid_for_kind",
                        "a classification claim cannot use a difference/ratio transform",
                        path=f"claims[{claim.claim_id}].transform",
                    )
                )
            else:
                violations += validate_classification(claim, observation)
        elif claim.kind in _BASIS_REQUIRED_KINDS:
            if is_derived:
                violations += validate_derived_numeric(claim, by_id)
            # An attribution observation is entirely and only about
            # attribution (application/recognize_structure.py's counterpart,
            # tools/definitions/analysis.py's `attribution()`, never mixes in
            # other fields) — citing it at all is a well-formed basis, field_
            # path or not, exactly like limitations.py's
            # ATTRIBUTION_NOT_CAUSALITY is observation-wide rather than
            # field-triggered. Found live 2026-09-06 (progress log section
            # 14.6): get_attribution's model_view hands the model
            # observation_id and top_tokens, never a citable field_path for
            # "which tokens drove this" — a scientific claim answering exactly
            # that question had no field_path to name and no evidence
            # citation to offer either, so every attempt failed
            # claim_has_no_basis before the limitation-derivation fix in
            # limitations.py (section 14.4) ever had a candidate to apply to.
            has_observation_basis = bool(
                observation is not None
                and (
                    (claim.field_path and observation.has(claim.field_path))
                    or observation.kind is ObservationKind.ATTRIBUTION
                )
            )
            violations += validate_basis(claim, has_observation_basis=has_observation_basis)
        # limitation/recommendation kinds carry no field- or citation-basis
        # requirement of their own; they are caveats and proposals, not facts.

        violations += validate_citations(claim, evidence_by_id, read_evidence_ids=read_evidence_ids)
        violations += validate_claim_wording(claim)

    violations += validate_answer_markdown(candidate.answer_markdown)
    violations += validate_no_hitcount_severity(candidate.claims, candidate.answer_markdown)
    violations += validate_no_uncited_links(candidate.answer_markdown)
    violations += validate_markdown_numeric_coverage(candidate.answer_markdown, candidate.claims)

    known_claim_ids = frozenset(by_id)
    for index, step in enumerate(candidate.recommended_next_steps):
        violations += validate_recommendation_basis(index, step.basis_claim_ids, known_claim_ids)

    violations += _validate_limitations(
        candidate, observations_by_id=observations_by_id
    )

    if violations:
        return AnswerValidationResult(violations=tuple(violations))

    try:
        answer = _build_answer(candidate, session_id, run_id, candidate_generation, language, now)
    except ValueError as exc:
        # A residual domain invariant this validator did not anticipate. Still
        # a correctable violation, never a 500 back to the caller.
        return AnswerValidationResult(
            violations=(Violation("candidate_malformed", str(exc)),)
        )
    return AnswerValidationResult(violations=(), answer=answer)


def _validate_limitations(
    candidate: GroundedAnswerCandidate, *, observations_by_id: Mapping[str, Observation]
) -> list[Violation]:
    violations: list[Violation] = []
    for index, limitation in enumerate(candidate.limitations):
        if limitation.code not in _KNOWN_LIMITATION_CODES:
            violations.append(
                Violation(
                    "unknown_limitation_code", f"{limitation.code!r} is not a declared limitation code",
                    path=f"limitations[{index}].code",
                )
            )

    observation_limitations = {
        obs_id: obs.required_limitations for obs_id, obs in observations_by_id.items()
    }
    cited_evidence = any(claim.citation_ids for claim in candidate.claims)
    has_recommendation = bool(candidate.recommended_next_steps) or any(
        claim.kind == "recommendation" for claim in candidate.claims
    )
    required = required_for_answer(
        candidate.claims, observation_limitations=observation_limitations,
        cited_evidence=cited_evidence, has_recommendation=has_recommendation,
    )
    declared = {l.code for l in candidate.limitations}
    missing = required - declared
    if missing:
        violations.append(
            Violation(
                "missing_required_limitation",
                f"this answer requires limitation(s) it did not declare: {sorted(missing)}",
                path="limitations", expected=sorted(missing),
            )
        )
    return violations


def _build_answer(
    candidate: GroundedAnswerCandidate,
    session_id: str,
    run_id: str,
    candidate_generation: int,
    language: str,
    now: datetime,
) -> GroundedAnswer:
    claims = tuple(
        Claim(
            claim_id=c.claim_id, kind=ClaimKind(c.kind), text=c.text,
            observation_id=c.observation_id, field_path=c.field_path,
            source_value=c.source_value, rendered_value=c.rendered_value,
            transform=c.transform, citation_ids=tuple(c.citation_ids),
            input_claim_ids=tuple(c.input_claim_ids),
        )
        for c in candidate.claims
    )
    limitations = tuple(
        Limitation(
            code=LimitationCode(l.code), text=l.text or text_for(LimitationCode(l.code), language)
        )
        for l in candidate.limitations
    )
    steps = tuple(
        RecommendedNextStep(text=s.text, basis_claim_ids=tuple(s.basis_claim_ids))
        for s in candidate.recommended_next_steps
    )
    return GroundedAnswer.create(
        session_id=session_id, run_id=run_id, answer_markdown=candidate.answer_markdown,
        claims=claims, limitations=limitations, recommended_next_steps=steps,
        candidate_generation=candidate_generation, now=now,
    )
