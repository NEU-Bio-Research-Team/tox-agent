"""Deterministic SMILES <-> BPE-token <-> heavy-atom alignment (plan section 5.1).

ChemBERTa attributes importance to byte-level BPE tokens. Chemists reason about
atoms and substructures, not tokens, so an explainer needs to project token
importances onto atom indices. This module is the projection, and it is a pure
function: given a canonical SMILES string and the character spans of each token,
it says which heavy atoms each token covers.

The atom index contract (D-XAI-4): ``atom_index`` is the 0-based position of a
heavy atom in ``canonical_smiles`` counting left to right. RDKit assigns atom
indices in exactly that order when it parses the string, so ``atom_index`` also
indexes ``Chem.MolFromSmiles(canonical_smiles)``. The frontend walks the same
string the same way, so indices line up without shipping a coordinate set; a
mismatch is a loud version-string check, never a silently wrong highlight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

#: Bumped only if the walk below changes in a way that moves atom indices. The
#: control plane echoes it and the frontend refuses to highlight on a mismatch.
ATOM_ORDER_VERSION = "rdkit-output-order-v1"

#: Organic-subset atoms that may appear outside brackets. Two-letter symbols are
#: matched first so ``Cl`` is one atom, not a chlorine-less carbon plus an "l".
_TWO_LETTER = ("Cl", "Br")
_ONE_LETTER = frozenset("BCNOPSFIbcnosp")


@dataclass(frozen=True)
class AtomSpan:
    atom_index: int
    symbol: str
    start: int
    end: int


@dataclass(frozen=True)
class AtomAlignment:
    canonical_smiles: str
    atom_spans: tuple[AtomSpan, ...]
    #: ``token_atoms[i]`` is the tuple of atom indices token ``i`` overlaps,
    #: possibly empty (bond symbols, ring digits, parentheses, stereo marks).
    token_atoms: tuple[tuple[int, ...], ...]


def heavy_atom_char_spans(smiles: str) -> list[tuple[int, int]]:
    """Character span ``[start, end)`` of every heavy atom's element symbol, in
    string order. Ring-closure digits, bond symbols, parentheses, ``%NN``,
    stereo ``@`` / ``/`` / ``\\`` and, inside a bracket atom, the brackets,
    charge and hydrogen count are all skipped — they belong to no atom.
    """
    spans: list[tuple[int, int]] = []
    i, n = 0, len(smiles)
    while i < n:
        ch = smiles[i]
        if ch == "[":
            close = smiles.find("]", i)
            if close == -1:
                close = n - 1
            # Skip an optional isotope number, then take the element symbol:
            # an upper-case letter with an optional lower-case letter, or a
            # single aromatic lower-case letter.
            k = i + 1
            while k < close and smiles[k].isdigit():
                k += 1
            if k < close and smiles[k].isalpha():
                if (
                    smiles[k].isupper()
                    and k + 1 < close
                    and smiles[k + 1].islower()
                ):
                    spans.append((k, k + 2))
                else:
                    spans.append((k, k + 1))
            i = close + 1
            continue
        if smiles[i : i + 2] in _TWO_LETTER:
            spans.append((i, i + 2))
            i += 2
            continue
        if ch in _ONE_LETTER:
            spans.append((i, i + 1))
            i += 1
            continue
        i += 1
    return spans


def align_tokens_to_atoms(
    canonical_smiles: str, token_char_spans: Sequence[tuple[int, int]]
) -> AtomAlignment:
    """Project token character spans onto heavy-atom indices.

    ``token_char_spans`` come from the tokenizer's ``return_offsets_mapping``.
    A token whose span overlaps an atom's element-symbol span maps to that
    atom; a token overlapping several atoms maps to all of them (the caller
    splits its importance); a token overlapping none maps to nothing.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {canonical_smiles!r}")

    spans = heavy_atom_char_spans(canonical_smiles)
    if mol.GetNumAtoms() != len(spans):
        raise ValueError(
            "heavy-atom count disagreement between RDKit "
            f"({mol.GetNumAtoms()}) and the SMILES walk ({len(spans)}) for "
            f"{canonical_smiles!r}; refusing to align rather than mislabel atoms"
        )

    atom_spans = tuple(
        AtomSpan(
            atom_index=idx,
            symbol=mol.GetAtomWithIdx(idx).GetSymbol(),
            start=start,
            end=end,
        )
        for idx, (start, end) in enumerate(spans)
    )

    token_atoms: list[tuple[int, ...]] = []
    for tstart, tend in token_char_spans:
        hit = tuple(
            idx
            for idx, (astart, aend) in enumerate(spans)
            if astart < tend and tstart < aend  # half-open interval overlap
        )
        token_atoms.append(hit)

    return AtomAlignment(
        canonical_smiles=canonical_smiles,
        atom_spans=atom_spans,
        token_atoms=tuple(token_atoms),
    )
