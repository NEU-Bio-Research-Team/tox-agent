"""Deterministic token-to-atom-and-bond alignment for explanation v2.

RDKit owns molecular indices.  This walker only associates SMILES syntax with
those indices; in particular ring closures are resolved through parser state,
never with a regex that can confuse a digit with an atom index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .token_atom_align import AtomAlignment, align_tokens_to_atoms

STRUCTURE_ORDER_VERSION = "rdkit-structure-order-v2"
_BOND_CHARS = frozenset("-=#:/$\\")


@dataclass(frozen=True)
class BondSpan:
    bond_index: int
    begin_atom_index: int
    end_atom_index: int
    bond_type: str
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StructureAlignment:
    atoms: AtomAlignment
    bonds: tuple[BondSpan, ...]
    token_bonds: tuple[tuple[int, ...], ...]


def _atom_end(smiles: str, start: int) -> int:
    if smiles[start] == "[":
        close = smiles.find("]", start)
        return len(smiles) if close < 0 else close + 1
    if smiles[start:start + 2] in ("Cl", "Br"):
        return start + 2
    return start + 1


def _is_atom_start(smiles: str, index: int) -> bool:
    char = smiles[index]
    return char == "[" or char.isalpha()


def align_tokens_to_structure(
    canonical_smiles: str, token_char_spans: Sequence[tuple[int, int]],
) -> StructureAlignment:
    """Return positional RDKit atom/bond ownership for every token span.

    Only syntax that actually has a token span becomes direct bond attribution.
    Implicit bonds remain in the returned topology, but their presentation
    intensity is derived separately by the caller from adjacent atoms.
    """
    from rdkit import Chem

    atoms = align_tokens_to_atoms(canonical_smiles, token_char_spans)
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {canonical_smiles!r}")

    n = len(canonical_smiles)
    atom_index = -1
    current: int | None = None
    branch_stack: list[int | None] = []
    rings: dict[str, tuple[int, tuple[int, int] | None]] = {}
    pending_bond: tuple[int, int] | None = None
    spans: dict[int, list[tuple[int, int]]] = {}

    def record(first: int, second: int, syntax: tuple[int, int] | None) -> None:
        bond = mol.GetBondBetweenAtoms(first, second)
        if bond is None:
            raise ValueError(f"SMILES walk could not resolve bond {first}-{second}")
        if syntax is not None:
            spans.setdefault(bond.GetIdx(), []).append(syntax)

    i = 0
    while i < n:
        char = canonical_smiles[i]
        if _is_atom_start(canonical_smiles, i):
            atom_index += 1
            if atom_index >= mol.GetNumAtoms():
                raise ValueError("SMILES atom walk exceeded RDKit atom count")
            if current is not None:
                record(current, atom_index, pending_bond)
            current = atom_index
            pending_bond = None
            i = _atom_end(canonical_smiles, i)
            continue
        if char == "(":
            branch_stack.append(current)
        elif char == ")":
            if not branch_stack:
                raise ValueError("unbalanced SMILES branch")
            current = branch_stack.pop()
        elif char in _BOND_CHARS:
            pending_bond = (i, i + 1)
        elif char.isdigit() or (char == "%" and i + 2 < n and canonical_smiles[i + 1:i + 3].isdigit()):
            if current is None:
                raise ValueError("ring closure has no current atom")
            end = i + 3 if char == "%" else i + 1
            key = canonical_smiles[i:end]
            if key in rings:
                other, first_syntax = rings.pop(key)
                record(other, current, (i, end))
                if pending_bond is not None:
                    record(other, current, pending_bond)
                if first_syntax is not None:
                    record(other, current, first_syntax)
            else:
                rings[key] = (current, pending_bond)
            pending_bond = None
            i = end
            continue
        i += 1
    if rings or branch_stack or atom_index + 1 != mol.GetNumAtoms():
        raise ValueError("SMILES topology walk disagrees with RDKit")

    bonds = tuple(
        BondSpan(
            bond_index=bond.GetIdx(), begin_atom_index=bond.GetBeginAtomIdx(),
            end_atom_index=bond.GetEndAtomIdx(), bond_type=str(bond.GetBondType()),
            spans=tuple(spans.get(bond.GetIdx(), [])),
        )
        for bond in mol.GetBonds()
    )
    token_bonds = tuple(
        tuple(
            bond.bond_index for bond in bonds
            if any(start < end_token and token_start < end for start, end in bond.spans)
        )
        for token_start, end_token in token_char_spans
    )
    return StructureAlignment(atoms=atoms, bonds=bonds, token_bonds=token_bonds)
