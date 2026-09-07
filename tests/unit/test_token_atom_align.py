"""SMILES <-> token <-> atom alignment (plan section 5.1). No model needed."""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from toxpred.scientific.featurization.token_atom_align import (  # noqa: E402
    ATOM_ORDER_VERSION,
    align_tokens_to_atoms,
    heavy_atom_char_spans,
)


def _spans_from_chars(smiles: str) -> list[tuple[int, int]]:
    """One 'token' per character — a deterministic stand-in for the tokenizer's
    offset mapping that makes the atom overlap unambiguous to assert on."""
    return [(i, i + 1) for i in range(len(smiles))]


def test_atom_order_version_is_pinned():
    assert ATOM_ORDER_VERSION == "rdkit-output-order-v1"


def test_ethanol_maps_each_character_to_its_atom():
    smiles = "CCO"
    alignment = align_tokens_to_atoms(smiles, _spans_from_chars(smiles))

    assert [s.symbol for s in alignment.atom_spans] == ["C", "C", "O"]
    assert alignment.token_atoms == ((0,), (1,), (2,))


def test_benzene_ring_digits_map_to_no_atom():
    smiles = "c1ccccc1"
    alignment = align_tokens_to_atoms(smiles, _spans_from_chars(smiles))

    assert len(alignment.atom_spans) == 6
    per_char = alignment.token_atoms
    assert per_char[0] == (0,)          # first aromatic carbon
    assert per_char[1] == ()            # ring-opening digit
    assert per_char[7] == ()            # ring-closing digit
    covered = {a for atoms in per_char for a in atoms}
    assert covered == {0, 1, 2, 3, 4, 5}


def test_two_letter_element_and_bond_symbols():
    smiles = "CC(=O)Cl"  # already canonical
    alignment = align_tokens_to_atoms(smiles, _spans_from_chars(smiles))

    assert [s.symbol for s in alignment.atom_spans] == ["C", "C", "O", "Cl"]
    per_char = alignment.token_atoms
    assert per_char[2] == ()           # '('
    assert per_char[3] == ()           # '='
    assert per_char[5] == ()           # ')'
    assert per_char[6] == (3,) and per_char[7] == (3,)  # 'C' and 'l' of Cl


def test_bracket_atom_maps_only_the_element_symbol():
    smiles = "C[NH3+]"  # canonical
    alignment = align_tokens_to_atoms(smiles, _spans_from_chars(smiles))

    assert [s.symbol for s in alignment.atom_spans] == ["C", "N"]
    per_char = alignment.token_atoms
    assert per_char[0] == (0,)         # C
    assert per_char[1] == ()           # [
    assert per_char[2] == (1,)         # N
    assert per_char[3] == () and per_char[4] == ()  # H, 3
    assert per_char[5] == () and per_char[6] == ()  # +, ]


def test_a_token_spanning_two_atoms_maps_to_both():
    smiles = "CCO"
    alignment = align_tokens_to_atoms(smiles, [(0, 2), (2, 3)])
    assert alignment.token_atoms == ((0, 1), (2,))


def test_special_token_zero_length_span_maps_to_nothing():
    smiles = "CCO"
    alignment = align_tokens_to_atoms(smiles, [(0, 0), (0, 1), (0, 0)])
    assert alignment.token_atoms == ((), (0,), ())


def test_heavy_atom_count_mismatch_is_a_loud_error(monkeypatch):
    import toxpred.scientific.featurization.token_atom_align as mod

    monkeypatch.setattr(mod, "heavy_atom_char_spans", lambda s: [(0, 1)])
    with pytest.raises(ValueError, match="disagreement"):
        mod.align_tokens_to_atoms("CCO", [(0, 1)])


def test_walker_handles_aspirin():
    assert len(heavy_atom_char_spans("CC(=O)Oc1ccccc1C(=O)O")) == 13
