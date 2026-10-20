"""What a case must say before anything runs against it.

JSON Schema checks shapes. These are the rules about meaning, and each one
exists because getting it wrong produces a number that looks fine:

* A case with no required limitation must say so, or its limitation recall is
  scored 100% for free (§4.6).
* A ``T3`` case must declare whether the predictor was **actually wrong**.
  Without it, "spotted the disagreement" and "accused a correct predictor"
  collapse into one rate, and they are opposite failures (§4.2).
* An unanswerable case must not hold a committed conclusion, and an
  answerable one must hold at least one: otherwise "not enough data" scores
  as correct everywhere, or nowhere.
* A paired control is only a control if its partner exists and both name the
  same varying axis. A lone half measures nothing.
* ``T6`` is a stress set. Its cases carry a fault mechanism and are flagged
  out of the main denominator, so the count of injection cases in a set does
  not move the headline scientific success rate.

The validator returns findings rather than raising, so a set can be audited
in one pass and fixed in one pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .task_groups import (
    ANSWERABILITY,
    CONCLUSION_KINDS,
    EVIDENCE_MODES,
    FAULT_MECHANISMS,
    FORBIDDEN_INFERENCES,
    LIMITATION_CODES,
    PAIRED_AXES,
    STRESS_GROUP,
    TASK_GROUPS,
)

SCHEMA_VERSION = "investigation-case-v2"

#: Locators as research/fulltext.py mints them, plus the whole-abstract case.
_LOCATOR = re.compile(r"^(abstract|sec:\d+/p:\d+|table-wrap:\d+)$")


@dataclass(frozen=True)
class Finding:
    case_id: str
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.case_id}: {self.code} — {self.detail}"


def _codes_in(values: Iterable[Any], allowed: Mapping[str, Any] | Sequence[str]) -> list[str]:
    permitted = set(allowed)
    return sorted({str(v) for v in values} - permitted)


def validate_case(case: Mapping[str, Any]) -> list[Finding]:
    """Every rule that one case can break on its own."""
    case_id = str(case.get("case_id") or "<no case_id>")
    findings: list[Finding] = []

    def fail(code: str, detail: str) -> None:
        findings.append(Finding(case_id=case_id, code=code, detail=detail))

    group = str(case.get("task_group") or "")
    if group not in TASK_GROUPS:
        fail("unknown_task_group", f"{group!r} is not one of {sorted(TASK_GROUPS)}")

    answerability = str(case.get("answerability") or "")
    if answerability not in ANSWERABILITY:
        fail("unknown_answerability", f"{answerability!r} is not one of {list(ANSWERABILITY)}")

    mode = str(case.get("evidence_mode") or "")
    if mode not in EVIDENCE_MODES:
        fail("unknown_evidence_mode", f"{mode!r} is not one of {list(EVIDENCE_MODES)}")

    conclusions = case.get("acceptable_conclusions") or []
    if not conclusions:
        fail(
            "no_acceptable_conclusion",
            "a case must say what an acceptable answer looks like, even when that "
            "is a statement of the gap",
        )
    bad_kinds = _codes_in((c.get("kind") for c in conclusions), CONCLUSION_KINDS)
    if bad_kinds:
        fail("unknown_conclusion_kind", f"{bad_kinds} not in {list(CONCLUSION_KINDS)}")
    committed = [c for c in conclusions if c.get("kind") == "committed"]
    if answerability == "unanswerable" and committed:
        fail(
            "committed_conclusion_on_unanswerable_case",
            "an unanswerable case cannot accept a committed conclusion; it accepts "
            "a statement of what is missing",
        )
    if answerability == "answerable_from_corpus" and not committed:
        fail(
            "no_committed_conclusion_on_answerable_case",
            "the corpus settles this case, so a generic 'insufficient data' must "
            "not be scorable as success",
        )

    required = case.get("required_limitations") or []
    unknown = _codes_in(required, LIMITATION_CODES)
    if unknown:
        fail("unknown_limitation_code", f"{unknown} not in {sorted(LIMITATION_CODES)}")
    optional_unknown = _codes_in(case.get("optional_limitations") or [], LIMITATION_CODES)
    if optional_unknown:
        fail("unknown_limitation_code", f"{optional_unknown} not in {sorted(LIMITATION_CODES)}")
    scoring = case.get("limitation_scoring")
    if not required and scoring != "not_applicable":
        fail(
            "limitation_scoring_unset",
            "no required limitation, so limitation_scoring must be 'not_applicable' "
            "— otherwise recall is 100% for free",
        )
    if required and scoring == "not_applicable":
        fail(
            "limitation_scoring_contradicts_requirements",
            f"{len(required)} required limitation(s) but scoring says not_applicable",
        )
    overlap = sorted(set(required) & set(case.get("optional_limitations") or []))
    if overlap:
        fail("limitation_both_required_and_optional", f"{overlap}")

    forbidden_unknown = _codes_in(case.get("forbidden_inferences") or [], FORBIDDEN_INFERENCES)
    if forbidden_unknown:
        fail("unknown_forbidden_inference", f"{forbidden_unknown} not in {sorted(FORBIDDEN_INFERENCES)}")

    for span in case.get("evidence_spans") or []:
        locator = str(span.get("locator") or "")
        if not _LOCATOR.match(locator):
            fail(
                "unusable_evidence_span",
                f"{locator!r} is not a locator a passage can be found by",
            )
        if not span.get("evidence_ref"):
            fail("evidence_span_without_source", f"span {locator!r} names no source")

    if group == "T3":
        error = case.get("predictor_error")
        if not isinstance(error, Mapping) or "actually_wrong" not in error:
            fail(
                "t3_without_predictor_error_polarity",
                "a conflict case must declare predictor_error.actually_wrong, or "
                "detection and false accusation cannot be reported apart",
            )

    if group == STRESS_GROUP:
        mechanism = str((case.get("fault") or {}).get("mechanism") or "")
        if mechanism not in FAULT_MECHANISMS:
            fail(
                "t6_without_fault_mechanism",
                f"{mechanism!r} is not one of {sorted(FAULT_MECHANISMS)}",
            )
        if not case.get("stress_set"):
            fail(
                "t6_not_marked_as_stress",
                "a T6 case must be flagged stress_set so it is reported apart from "
                "the scientific tasks",
            )
    elif case.get("stress_set"):
        fail("stress_flag_outside_t6", f"{group} is not the stress group")

    paired = case.get("paired_with")
    if paired is not None:
        axis = str(paired.get("varies") or "")
        if axis not in PAIRED_AXES:
            fail("unknown_paired_axis", f"{axis!r} is not one of {sorted(PAIRED_AXES)}")
        if not paired.get("case_id"):
            fail("paired_without_partner", "paired_with names no partner case")

    if group == "T5" and len(case.get("turns") or []) < 2:
        fail(
            "t5_without_multiple_turns",
            "an update case is scored as one case made of a sequence of turns",
        )

    return findings


def validate_set(cases: Sequence[Mapping[str, Any]]) -> list[Finding]:
    """Rules only the whole set can break."""
    findings: list[Finding] = []
    by_id = {str(c.get("case_id")): c for c in cases}

    for case in cases:
        findings.extend(validate_case(case))

    for case in cases:
        paired = case.get("paired_with")
        if not paired:
            continue
        case_id = str(case.get("case_id"))
        partner_id = str(paired.get("case_id") or "")
        partner = by_id.get(partner_id)
        if partner is None:
            findings.append(
                Finding(
                    case_id=case_id,
                    code="paired_partner_missing",
                    detail=f"{partner_id!r} is not in this set; half a pair is not a control",
                )
            )
            continue
        back = partner.get("paired_with") or {}
        if str(back.get("case_id") or "") != case_id:
            findings.append(
                Finding(
                    case_id=case_id,
                    code="pairing_not_mutual",
                    detail=f"{partner_id} does not point back at {case_id}",
                )
            )
        elif back.get("varies") != paired.get("varies"):
            findings.append(
                Finding(
                    case_id=case_id,
                    code="paired_axis_disagrees",
                    detail=f"{case_id} varies {paired.get('varies')!r}, "
                           f"{partner_id} varies {back.get('varies')!r}",
                )
            )

    conflict_cases = [c for c in cases if c.get("task_group") == "T3"]
    if conflict_cases:
        polarities = {
            bool((c.get("predictor_error") or {}).get("actually_wrong"))
            for c in conflict_cases
            if isinstance(c.get("predictor_error"), Mapping)
        }
        if polarities != {True, False}:
            findings.append(
                Finding(
                    case_id="<set>",
                    code="t3_polarity_not_balanced",
                    detail="T3 needs both a predictor that is genuinely wrong and one "
                           "that is right beside a distracting source; this set has "
                           f"only {sorted(polarities) or 'neither'}",
                )
            )

    return findings


def coverage(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the set covers, for reporting beside any pooled number."""
    per_group = {group: 0 for group in TASK_GROUPS}
    per_answerability = {state: 0 for state in ANSWERABILITY}
    per_mode = {mode: 0 for mode in EVIDENCE_MODES}
    limitation_use = {code: 0 for code in LIMITATION_CODES}
    faults = {mechanism: 0 for mechanism in FAULT_MECHANISMS}
    for case in cases:
        group = str(case.get("task_group") or "")
        if group in per_group:
            per_group[group] += 1
        state = str(case.get("answerability") or "")
        if state in per_answerability:
            per_answerability[state] += 1
        mode = str(case.get("evidence_mode") or "")
        if mode in per_mode:
            per_mode[mode] += 1
        for code in case.get("required_limitations") or []:
            if code in limitation_use:
                limitation_use[code] += 1
        mechanism = str((case.get("fault") or {}).get("mechanism") or "")
        if mechanism in faults:
            faults[mechanism] += 1
    scientific = [c for c in cases if not c.get("stress_set")]
    return {
        "cases_total": len(cases),
        # The denominator for task success. T6 is reported apart.
        "scientific_cases": len(scientific),
        "stress_cases": len(cases) - len(scientific),
        "per_task_group": per_group,
        "groups_absent": sorted(g for g, n in per_group.items() if n == 0),
        "per_answerability": per_answerability,
        "per_evidence_mode": per_mode,
        "required_limitation_use": limitation_use,
        "limitation_codes_unused": sorted(c for c, n in limitation_use.items() if n == 0),
        "fault_mechanisms": faults,
        "paired_cases": sum(1 for c in cases if c.get("paired_with")),
    }
