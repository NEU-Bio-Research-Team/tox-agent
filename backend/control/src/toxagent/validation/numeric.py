"""Numeric claim validation (plan section 9.1).

The rule in one sentence: a number in an accepted answer equals the canonical
field it cites, exactly, under a declared and checkable transform. There is no
unit conversion, no "close enough" beyond the stated rounding tolerance, and no
string-replacement fix-up of a wrong answer — a claim that fails is returned as
a violation for the model to correct, never silently repaired.
"""
from __future__ import annotations

import math
import re
from typing import Mapping

from ..domain.errors import Violation
from ..domain.fieldpath import FieldPathError
from ..domain.observation import Observation
from .wire import ClaimCandidate

#: The canonical shape of a numeric ``rendered_value`` (plan sections 5.7, 9.1.5,
#: DEC-08): a single number, an optional leading sign, one optional decimal
#: separator that may be a dot or a Vietnamese comma, an optional trailing '%'.
#: No thousands separator ("1,731" must not be read as 1731), no spaces, no
#: parenthetical, no words. Display phrasing such as "0,0315 (3,15%)" belongs in
#: the claim's ``text``; ``rendered_value`` is the one number the transform
#: produced, so that it stays mechanically checkable against the source.
_CANONICAL_NUMBER = re.compile(r"^-?\d+(?:[.,]\d+)?%?$")

#: ADR 0005. The message a model gets back when it submits a compound render;
#: it names the exact fix rather than only reporting "unparseable".
_RENDERED_VALUE_RULE = (
    "rendered_value must be a single number such as '0.731' or '0,731' (a "
    "Vietnamese decimal comma is allowed) with an optional trailing '%'. Move "
    "any display phrasing like '0,0315 (3,15%)' into the claim's text; "
    "rendered_value carries only the number the transform produced."
)


def parse_rendered_number(rendered: str) -> float:
    """Parse a canonical numeric ``rendered_value``: dot or Vietnamese comma
    decimal, optional sign, optional trailing '%'. Anything else — a thousands
    separator, a trailing unit, a "0,0315 (3,15%)" compound — raises with the
    correction rule, never a silent best-effort extraction (plan section 9.1:
    "Không dùng regex/string replacement để sửa answer đã sai")."""
    text = rendered.strip()
    if not _CANONICAL_NUMBER.match(text):
        raise ValueError(f"{_RENDERED_VALUE_RULE} Got: {rendered!r}")
    if text.endswith("%"):
        text = text[:-1]
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError as exc:  # pragma: no cover - regex already constrains this
        raise ValueError(f"{_RENDERED_VALUE_RULE} Got: {rendered!r}") from exc


def round_tolerance(n: int) -> float:
    """Plan section 9.1: abs(rendered - source) <= 0.5 * 10^-n + 1e-12."""
    return 0.5 * (10.0 ** -n) + 1e-12


def validate_field_backed_numeric(
    claim: ClaimCandidate, observation: Observation | None
) -> list[Violation]:
    """A ``numeric`` claim that cites one observation field directly."""
    violations: list[Violation] = []
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
        return [Violation("claim_field_path_missing", "a numeric claim must name a field_path", path)]

    try:
        source_value = observation.value_at(claim.field_path)
    except FieldPathError as exc:
        return [
            Violation(
                "claim_field_path_unresolvable", str(exc), path=f"{path}.field_path",
                expected=None, actual=claim.field_path,
            )
        ]

    if not isinstance(source_value, (int, float)) or isinstance(source_value, bool):
        return [
            Violation(
                "claim_field_not_numeric",
                f"{claim.field_path} is not a numeric field",
                path=f"{path}.field_path", actual=source_value,
            )
        ]
    source_value = float(source_value)

    if claim.source_value is None:
        violations.append(
            Violation("claim_source_value_missing", "a numeric claim must state source_value", path)
        )
    elif not _numbers_equal(claim.source_value, source_value):
        violations.append(
            Violation(
                "claim_source_value_mismatch",
                f"claimed source_value does not match the observation",
                path=f"{path}.source_value", expected=source_value, actual=claim.source_value,
            )
        )

    violations.extend(_validate_rendered(claim, source_value, path))
    return violations


def validate_derived_numeric(
    claim: ClaimCandidate, resolved_inputs: Mapping[str, "ClaimCandidate"]
) -> list[Violation]:
    """A ``difference``/``ratio`` claim computed from earlier claims in the same
    candidate (plan section 9.1: "deterministic difference/ratio có input claim
    ids khai báo"). Inputs must themselves be well-formed numeric claims; this
    does not re-derive their correctness against an observation — that already
    happened when each was validated on its own.
    """
    path = f"claims[{claim.claim_id}]"
    if len(claim.input_claim_ids) != 2:
        return [
            Violation(
                "claim_derived_inputs_invalid",
                f"a {claim.transform} claim needs exactly two input_claim_ids, got "
                f"{len(claim.input_claim_ids)}",
                path=f"{path}.input_claim_ids",
            )
        ]
    inputs = []
    for claim_id in claim.input_claim_ids:
        source = resolved_inputs.get(claim_id)
        if source is None or source.source_value is None:
            return [
                Violation(
                    "claim_derived_input_missing",
                    f"input claim {claim_id!r} does not exist or has no numeric source_value",
                    path=f"{path}.input_claim_ids",
                )
            ]
        inputs.append(float(source.source_value))

    a, b = inputs
    if claim.transform == "difference":
        expected = a - b
    else:
        if b == 0:
            return [
                Violation(
                    "claim_derived_division_by_zero", "a ratio claim's second input is zero",
                    path=f"{path}.input_claim_ids",
                )
            ]
        expected = a / b

    violations: list[Violation] = []
    if claim.source_value is None or not _numbers_equal(claim.source_value, expected):
        violations.append(
            Violation(
                "claim_source_value_mismatch",
                f"the declared {claim.transform} does not match its inputs",
                path=f"{path}.source_value", expected=expected, actual=claim.source_value,
            )
        )
    violations.extend(_validate_rendered_exact(claim, expected, path))
    return violations


def _validate_rendered(claim: ClaimCandidate, source_value: float, path: str) -> list[Violation]:
    if claim.rendered_value is None:
        return [Violation("claim_rendered_value_missing", "a numeric claim must render a value", path)]
    try:
        rendered_number = parse_rendered_number(claim.rendered_value)
    except ValueError as exc:
        return [Violation("claim_rendered_value_unparseable", str(exc), path=f"{path}.rendered_value")]

    if claim.transform == "identity":
        target = source_value
        tolerance = 1e-9
    elif claim.transform.startswith("round:"):
        n = int(claim.transform.split(":")[1])
        target = source_value
        tolerance = round_tolerance(n)
    elif claim.transform.startswith("percent:"):
        n = int(claim.transform.split(":")[1])
        target = source_value * 100.0
        tolerance = round_tolerance(n)
    else:
        # difference/ratio never reach here; they go through
        # validate_derived_numeric, which has its own rendering check.
        return []

    if abs(rendered_number - target) > tolerance:
        return [
            Violation(
                "claim_rendered_value_mismatch",
                f"rendered value is outside the tolerance for {claim.transform}",
                path=f"{path}.rendered_value", expected=target, actual=rendered_number,
            )
        ]
    return []


def _validate_rendered_exact(claim: ClaimCandidate, expected: float, path: str) -> list[Violation]:
    if claim.rendered_value is None:
        return []
    try:
        rendered_number = parse_rendered_number(claim.rendered_value)
    except ValueError as exc:
        return [Violation("claim_rendered_value_unparseable", str(exc), path=f"{path}.rendered_value")]
    if abs(rendered_number - expected) > 1e-9:
        return [
            Violation(
                "claim_rendered_value_mismatch", f"rendered value does not match the computed {claim.transform}",
                path=f"{path}.rendered_value", expected=expected, actual=rendered_number,
            )
        ]
    return []


def _numbers_equal(claimed: object, actual: float, *, tolerance: float = 1e-9) -> bool:
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)):
        return False
    if math.isnan(actual):
        return False
    return abs(float(claimed) - actual) <= tolerance
