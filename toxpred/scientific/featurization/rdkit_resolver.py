"""SMILES parsing and canonicalisation.

Canonicalisation happens once, at the edge, and every downstream consumer uses
the canonical form — so a cache key, a model input and a response field cannot
disagree about which string represents the molecule.
"""
from __future__ import annotations

from rdkit import Chem, RDLogger

from ...domain.molecule import InvalidSmilesError, Molecule

RDLogger.DisableLog("rdApp.*")

MAX_SMILES_LENGTH = 1000


def resolve(smiles: str) -> Molecule:
    if smiles is None:
        raise InvalidSmilesError("", "input is null")
    stripped = smiles.strip()
    if not stripped:
        raise InvalidSmilesError(smiles, "input is empty")
    if len(stripped) > MAX_SMILES_LENGTH:
        raise InvalidSmilesError(
            smiles, f"input exceeds {MAX_SMILES_LENGTH} characters ({len(stripped)})"
        )

    mol = Chem.MolFromSmiles(stripped)
    if mol is None:
        raise InvalidSmilesError(smiles, "RDKit could not parse the structure")
    if mol.GetNumAtoms() == 0:
        # RDKit returns an empty molecule rather than None for some inputs.
        raise InvalidSmilesError(smiles, "structure contains no atoms")

    return Molecule(
        input_smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol),
        elements=frozenset(a.GetSymbol() for a in mol.GetAtoms()),
        num_atoms=mol.GetNumAtoms(),
    )
