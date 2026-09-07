"""Classification claim validation (plan section 9.2).

A classification claim's ``source_value`` must equal the canonical enum value
exactly — the raw string the predictor emitted, not an alias. ``non_blocker``
does not become ``safe`` here; that substitution is caught as a *wording*
problem by ``prohibited_claims``, because it is a property of the claim's
prose, not of whether its stored value matches its source.
"""
from __future__ import annotations

from ..domain.errors import Violation
from ..domain.fieldpath import FieldPathError
from ..domain.observation import Observation
from .wire import ClaimCandidate


def validate_classification(
    claim: ClaimCandidate, observation: Observation | None
) -> list[Violation]:
    path = f"claims[{claim.claim_id}]"
    if observation is None:
        return [
            Violation(
                "claim_observation_not_found",
                f"observation {claim.observation_id!r} does not exist in this session",
                path=f"{path}.observation_id",
            )
        ]
    if not claim.field_path:
        return [
            Violation("claim_field_path_missing", "a classification claim must name a field_path", path)
        ]

    try:
        source_value = observation.value_at(claim.field_path)
    except FieldPathError as exc:
        return [
            Violation(
                "claim_field_path_unresolvable", str(exc), path=f"{path}.field_path",
                actual=claim.field_path,
            )
        ]

    if isinstance(source_value, (int, float)) and not isinstance(source_value, bool):
        return [
            Violation(
                "claim_field_not_classification",
                f"{claim.field_path} is a numeric field; use kind=numeric",
                path=f"{path}.field_path",
            )
        ]

    violations: list[Violation] = []
    if claim.source_value != source_value:
        violations.append(
            Violation(
                "claim_source_value_mismatch",
                "claimed source_value is not the exact canonical value",
                path=f"{path}.source_value", expected=source_value, actual=claim.source_value,
            )
        )
    if claim.rendered_value is not None and claim.rendered_value != str(source_value):
        # The canonical claim stores the raw enum; a display alias belongs to
        # the renderer, never to the stored claim (plan section 9.2).
        violations.append(
            Violation(
                "claim_rendered_value_is_an_alias",
                "rendered_value must be the raw canonical value, not a display alias",
                path=f"{path}.rendered_value", expected=str(source_value), actual=claim.rendered_value,
            )
        )
    return violations
