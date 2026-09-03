"""Applicability assessment by element rules.

Ported from ``backend/ood_guard.py``. Renamed on purpose: the original called
itself an OOD detector and returned ``ood_risk: "LOW"``, which reads as evidence
that a molecule is inside the training distribution. It is an element whitelist.
It can flag an unusual element; it cannot certify that anything is in-domain.

``method`` is carried in the response so the limitation travels with the result.
"""
from __future__ import annotations

from ..domain.molecule import Molecule
from ..domain.prediction import ApplicabilityAssessment

METHOD = "element_rules_v1"

COMMON_ELEMENTS = frozenset({"C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H"})

# Elements essentially unrepresented in the ClinTox/hERG/Tox21 training sets;
# a prediction for a molecule containing one is extrapolation.
HIGH_RISK_ELEMENTS = frozenset({"Pt", "Bi", "Au", "As", "Hg", "Pb", "Cd", "Sn", "Sb"})


def assess(molecule: Molecule) -> ApplicabilityAssessment:
    high_risk = sorted(molecule.elements & HIGH_RISK_ELEMENTS)
    uncommon = sorted(molecule.elements - COMMON_ELEMENTS - HIGH_RISK_ELEMENTS)

    reasons: list[str] = []
    if high_risk:
        reasons.append(
            f"contains element(s) with essentially no training-set support: {', '.join(high_risk)}"
        )
    if uncommon:
        reasons.append(f"contains uncommon element(s): {', '.join(uncommon)}")

    if high_risk:
        status = "out_of_domain"
    elif uncommon:
        status = "limited"
    else:
        status = "ok"
        reasons.append(
            "all elements are common in the training sets; this rule cannot confirm "
            "distributional similarity beyond element composition"
        )

    return ApplicabilityAssessment(status=status, method=METHOD, reasons=tuple(reasons))
