from __future__ import annotations

from typing import Any, Dict, List

from rdkit import Chem

# Organic drug-like element set mirrored from graph featurization assumptions.
COMMON_ELEMENTS = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H"}
# Elements that commonly indicate higher OOD risk for this model family.
OOD_RISK_ELEMENTS = {"Pt", "Bi", "Au", "As", "Hg", "Pb", "Cd", "Sn", "Sb"}


def _sorted(values: List[str]) -> List[str]:
    return sorted(values, key=lambda item: (item not in OOD_RISK_ELEMENTS, item))


def check_ood_risk(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "ood_risk": "PARSE_ERROR",
            "flag": True,
            "reason": "SMILES parse failed; reliability cannot be assessed.",
            "rare_elements": [],
            "high_risk_elements": [],
            "recommendation": "Validate molecular structure before interpreting toxicity scores.",
        }

    atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
    rare = atoms - COMMON_ELEMENTS
    high_risk = atoms & OOD_RISK_ELEMENTS

    if high_risk:
        high_risk_elements = _sorted(list(high_risk))
        rare_elements = _sorted(list(rare))
        return {
            "ood_risk": "HIGH",
            "flag": True,
            "reason": (
                "Contains organometallic or uncommon elements "
                f"{', '.join(high_risk_elements)} that are weakly represented in training data."
            ),
            "rare_elements": rare_elements,
            "high_risk_elements": high_risk_elements,
            "recommendation": (
                "Route to expert review and cross-check with external safety datasets "
                "(for example PubChem BioAssay, ECHA, ChEMBL) before final decision."
            ),
        }

    if rare:
        rare_elements = _sorted(list(rare))
        return {
            "ood_risk": "MEDIUM",
            "flag": True,
            "reason": (
                "Contains uncommon elements "
                f"{', '.join(rare_elements)} outside the model's main organic feature set."
            ),
            "rare_elements": rare_elements,
            "high_risk_elements": [],
            "recommendation": "Treat prediction as screening-only and request orthogonal validation assays.",
        }

    return {
        "ood_risk": "LOW",
        "flag": False,
        "reason": "Element profile is within common organic training distribution.",
        "rare_elements": [],
        "high_risk_elements": [],
        "recommendation": None,
    }
