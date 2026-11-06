"""Verification by claim type (#72, architecture proposal §9.1–9.2).

Before this, every numeric, classification, scientific and comparison claim
went to the LLM reviewer, which then judged things code can settle: whether a
margin of 30 follows from 300 nM over 10 nM, whether a field lookup returned
the field. Live (arm E, 2026-10-29) that reviewer blocked correct derived
claims in 3 of 4 rounds. §9.1 gives each of six claim types its own path, and
says numeric calculations are checked by code, not re-derived by the reviewer:

| type | path |
|---|---|
| artifact fact | code: the hard validator already proved the value equals the cited field |
| extracted numeric fact | code when every number in the claim is the cited record's measured value (unit-normalised); otherwise review |
| derived numeric claim | code: the approved calculation is re-run from its recorded inputs |
| scope relation | review, with the scope rule |
| scientific assessment | review, with the premise/counter-evidence rule |
| recommendation | structural: it must name the claims it rests on; recorded, not gated |

What the reviewer still sees, it sees as §9.2 frames it: the claim as observed
(X, "the answer says ..."), what the cited source states (Y), the rule for that
claim type, and the scope — together, in one entry.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ...domain import exposure_margin
from ...domain.fieldpath import FieldPathError
from ...domain.measurement_value import concentration_factors

ARTIFACT_FACT = "artifact_fact"
EXTRACTED_NUMERIC = "extracted_numeric"
DERIVED_NUMERIC = "derived_numeric"
SCOPE_RELATION = "scope_relation"
SCIENTIFIC_ASSESSMENT = "scientific_assessment"
RECOMMENDATION = "recommendation"

CLAIM_TYPES = (
    ARTIFACT_FACT, EXTRACTED_NUMERIC, DERIVED_NUMERIC, SCOPE_RELATION,
    SCIENTIFIC_ASSESSMENT, RECOMMENDATION,
)

#: What a reviewer judges for each type it receives. A type code settles
#: never reaches the reviewer, so it has no rule here.
RULES: dict[str, str] = {
    EXTRACTED_NUMERIC: (
        "The number must be the value the source states for this compound, endpoint and "
        "assay, with its comparator (=, <, >) and unit; a bound reported as a measurement, "
        "or another assay's value, is not supported."
    ),
    DERIVED_NUMERIC: (
        "Judge only whether the stated inputs are the ones the claim names; the arithmetic "
        "has been checked by code."
    ),
    SCOPE_RELATION: (
        "The relation (supports, contradicts, same as, differs from) must hold within the "
        "scope the sources share: same compound or variant, endpoint, assay/readout, species "
        "and conditions. A relation that crosses scopes is partially_supported at best."
    ),
    SCIENTIFIC_ASSESSMENT: (
        "The assessment must follow from its premises as the sources state them, within "
        "their scope; counter-evidence among the sources listed must be acknowledged; "
        "assumptions must be stated. Outside knowledge does not fill a gap."
    ),
}

#: A number with an optional molar unit: "300 nM", "0,3 µM", "30-fold".
_NUMBER = re.compile(
    r"(?<![\w.,])(\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*(pM|nM|µM|μM|uM|mM|M)?(?![\w])"
)
_TO_NM = concentration_factors()


@dataclass(frozen=True)
class ClaimCheck:
    claim_id: str
    claim_type: str
    #: ``code`` (settled here), ``review`` (goes to the reviewer) or
    #: ``structural`` (recorded, never gating).
    path: str
    verdict: str | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        item = {"claim_id": self.claim_id, "claim_type": self.claim_type, "path": self.path,
                "verdict": self.verdict, "reason": self.reason}
        if self.details:
            item["details"] = self.details
        return item


def _kind(claim) -> str:
    return getattr(claim.kind, "value", claim.kind)


def recompute_margin(payload: dict[str, Any]) -> tuple[bool, str]:
    """Re-run the exposure margin from its recorded inputs; (agrees, why)."""
    inputs = payload.get("inputs") or {}
    try:
        ic50 = inputs["ic50"]
        cmax = inputs["cmax"]
        fu = inputs.get("fraction_unbound") or None
        result = exposure_margin.compute(
            exposure_margin.Concentration(float(ic50["value"]), ic50["unit"], str(ic50.get("source_ref", ""))),
            exposure_margin.Concentration(float(cmax["value"]), cmax["unit"], str(cmax.get("source_ref", ""))),
            fraction_unbound=float(fu["value"]) if fu else None,
            fu_source_ref=fu.get("source_ref") if fu else None,
        )
    except (KeyError, TypeError, ValueError, exposure_margin.InvalidMarginInput) as exc:
        return False, f"the recorded inputs do not recompute: {exc}"
    recorded = payload.get("margin")
    if not isinstance(recorded, (int, float)) or not math.isclose(
        result["margin"], float(recorded), rel_tol=1e-6
    ):
        return False, f"recomputed {result['margin']} from the inputs; recorded {recorded}"
    return True, (
        f"{payload.get('formula')}: recomputed {result['margin']:g} from "
        f"IC50 {ic50['value']} {ic50['unit']} and Cmax {cmax['value']} {cmax['unit']}"
    )


#: Approved calculations code can re-run, by method version.
RECOMPUTE = {exposure_margin.METHOD_VERSION: recompute_margin}


def numbers_in(text: str) -> list[tuple[Decimal, str | None]]:
    found = []
    for raw, unit in _NUMBER.findall(text or ""):
        try:
            found.append((Decimal(raw.replace(",", ".")), unit.replace("μ", "µ") if unit else None))
        except ArithmeticError:
            continue
    return found


def _nanomolar(value: Decimal, unit: str | None) -> Decimal | None:
    if unit is None:
        return None
    factor = _TO_NM.get(unit.replace("μ", "µ"))
    return value * factor if factor is not None else None


def numbers_match_record(text: str, measurement: dict[str, Any]) -> tuple[bool, str]:
    """Whether every concentration in ``text`` is the record's measured value.

    A unitless number (a year, a count, "30-fold") is not a concentration and
    is left alone; a claim with no concentration at all is not settled here.
    """
    value, unit = measurement.get("value"), measurement.get("unit")
    if not isinstance(value, (int, float)) or measurement.get("state") != "measured":
        return False, "the record states no measured value to compare with"
    relation = measurement.get("relation") or "="
    record_nm = _nanomolar(Decimal(repr(float(value))), unit)
    stated = [(n, u) for n, u in numbers_in(text) if u is not None]
    if not stated:
        return False, "the claim states no concentration to compare"
    for number, stated_unit in stated:
        nm = _nanomolar(number, stated_unit)
        if record_nm is None or nm is None or not math.isclose(float(nm), float(record_nm), rel_tol=1e-6):
            return False, f"{number} {stated_unit} is not the record's {relation} {value} {unit}"
    if relation != "=" and relation not in (text or ""):
        return False, f"the record's value is a bound ({relation} {value} {unit}); the claim drops it"
    return True, f"equals the record's measured {relation} {value} {unit}"


async def plan_checks(uow, *, session_id: str, answer) -> list[ClaimCheck]:
    """Every checkable claim of ``answer``, with its type and its path."""
    checks: list[ClaimCheck] = []
    ids = {c.claim_id for c in answer.claims}
    observations: dict[str, Any] = {}
    for claim in answer.claims:
        kind = _kind(claim)
        if kind == "limitation":
            continue
        if kind == "recommendation":
            basis = [cid for cid in claim.input_claim_ids if cid in ids]
            checks.append(ClaimCheck(
                claim.claim_id, RECOMMENDATION, "structural",
                verdict="supported" if basis else None,
                reason=("rests on " + ", ".join(basis)) if basis
                else "names no claim it rests on; its basis, scope and what would change it "
                     "should be stated",
            ))
            continue
        observation = None
        if claim.observation_id:
            if claim.observation_id not in observations:
                observations[claim.observation_id] = await uow.observations.get(
                    claim.observation_id, session_id=session_id
                )
            observation = observations[claim.observation_id]
        if observation is not None and claim.field_path and kind in {"numeric", "classification"}:
            try:
                observation.value_at(claim.field_path)
            except FieldPathError:
                observation = None
        if observation is not None and claim.field_path and kind in {"numeric", "classification"}:
            if observation.kind.value == "calculation":
                payload = observation.canonical_payload or {}
                method = payload.get("method_version")
                if method in RECOMPUTE:
                    agrees, why = RECOMPUTE[method](payload)
                    checks.append(ClaimCheck(
                        claim.claim_id, DERIVED_NUMERIC, "code",
                        verdict="supported" if agrees else "not_supported", reason=why,
                        details={"method_version": method, "observation_id": observation.id},
                    ))
                else:
                    checks.append(ClaimCheck(claim.claim_id, DERIVED_NUMERIC, "review",
                                             reason=f"no code recomputation for {method!r}"))
                continue
            checks.append(ClaimCheck(
                claim.claim_id, ARTIFACT_FACT, "code", verdict="supported",
                reason=(f"the value is {observation.id}.{claim.field_path}, checked field by "
                        "field by the hard validator"),
            ))
            continue
        if kind == "numeric" and claim.citation_ids:
            settled = None
            for evidence_id in claim.citation_ids:
                record = await uow.evidence.get(evidence_id, session_id=session_id)
                if record is None:
                    continue
                from ...research.measurement import is_measured_value, measured_value

                if not is_measured_value(record.normalized_facts):
                    continue
                agrees, why = numbers_match_record(claim.text, measured_value(record.normalized_facts))
                if agrees:
                    settled = ClaimCheck(claim.claim_id, EXTRACTED_NUMERIC, "code",
                                         verdict="supported", reason=why,
                                         details={"evidence_id": evidence_id})
                    break
            checks.append(settled or ClaimCheck(claim.claim_id, EXTRACTED_NUMERIC, "review"))
            continue
        if kind in {"comparison", "classification"}:
            checks.append(ClaimCheck(claim.claim_id, SCOPE_RELATION, "review"))
        elif kind == "numeric":
            checks.append(ClaimCheck(claim.claim_id, EXTRACTED_NUMERIC, "review"))
        else:
            checks.append(ClaimCheck(claim.claim_id, SCIENTIFIC_ASSESSMENT, "review"))
    return checks


def review_claim_ids(checks: list[ClaimCheck]) -> list[str]:
    return [c.claim_id for c in checks if c.path == "review"]


def code_verdicts(checks: list[ClaimCheck]) -> list[dict[str, Any]]:
    """Code-settled verdicts, in the reviewer's record shape, marked as code's."""
    return [
        {"claim_id": c.claim_id, "verdict": c.verdict, "reason": c.reason,
         "checked_by": "code", "claim_type": c.claim_type}
        for c in checks if c.path == "code" and c.verdict is not None
    ]
