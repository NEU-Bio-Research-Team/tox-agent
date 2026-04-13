from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except Exception:
    Chem = None
    DataStructs = None
    AllChem = None


def canonicalize_smiles(smiles: str) -> Dict[str, Any]:
    if not smiles or not smiles.strip():
        return {
            "valid": False,
            "canonical_smiles": None,
            "error": "smiles_empty",
            "atom_count": None,
        }

    if Chem is None:
        return {
            "valid": False,
            "canonical_smiles": None,
            "error": "rdkit_not_installed",
            "atom_count": None,
        }

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return {
            "valid": False,
            "canonical_smiles": None,
            "error": f"rdkit_parse_failed: {smiles}",
            "atom_count": None,
        }

    return {
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "error": None,
        "atom_count": mol.GetNumAtoms(),
    }


def fingerprint_from_smiles(
    smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
):
    if Chem is None or AllChem is None:
        return None

    validated = canonicalize_smiles(smiles)
    if not validated.get("valid"):
        return None

    mol = Chem.MolFromSmiles(str(validated["canonical_smiles"]))
    if mol is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(fp_a, fp_b) -> Optional[float]:
    if fp_a is None or fp_b is None or DataStructs is None:
        return None

    try:
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return None
