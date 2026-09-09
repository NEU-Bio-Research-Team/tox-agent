"""A molecule written as more than one component still aligns (K10).

`align_tokens_to_structure` walks the canonical SMILES and asks RDKit for the
bond between each atom and the one before it. It had no case for `.`, the
component separator, so after a `.` it asked for a bond between the last atom
of one fragment and the first atom of the next, found none, and raised
`ValueError`.

`/v1/explanations` does not catch that, so the endpoint failed outright for
any salt — an ordinary way to write a drug. Four of the forty-two molecules in
the golden panel are like this: sodium salicylate, diphenhydramine
hydrochloride, cisplatin, ferrocene. Found by running the XAI benchmark over
the panel, which is what a benchmark harness is for.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from toxpred.scientific.featurization.token_structure_align import (  # noqa: E402
    align_tokens_to_structure,
)


def spans_for(smiles: str) -> list[tuple[int, int]]:
    """One span per character: enough to exercise the walk without a tokenizer."""
    return [(index, index + 1) for index in range(len(smiles))]


MULTI_COMPONENT = [
    pytest.param("O=C([O-])c1ccccc1O.[Na+]", 11, id="sodium-salicylate"),
    pytest.param("CN(C)CCOC(c1ccccc1)c1ccccc1.Cl", 20, id="diphenhydramine-hcl"),
    pytest.param("N.N.[Cl][Pt][Cl]", 5, id="cisplatin"),
    pytest.param("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1", 11, id="ferrocene"),
    pytest.param("CCO.O", 4, id="minimal"),
]


@pytest.mark.parametrize("smiles,atom_count", MULTI_COMPONENT)
def test_a_multi_component_smiles_aligns_instead_of_raising(smiles, atom_count):
    alignment = align_tokens_to_structure(smiles, spans_for(smiles))
    assert len(alignment.atoms.atom_spans) == atom_count


@pytest.mark.parametrize("smiles,_atom_count", MULTI_COMPONENT)
def test_no_bond_is_invented_across_a_component_break(smiles, _atom_count):
    """The failure was asking for a bond that does not exist. The fix must not
    become inventing one: RDKit's own bond list is the reference."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    alignment = align_tokens_to_structure(smiles, spans_for(smiles))
    assert len(alignment.bonds) == molecule.GetNumBonds()
    real = {
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()
    }
    assert {(bond.begin_atom_index, bond.end_atom_index) for bond in alignment.bonds} == real


def test_a_branch_open_across_a_component_break_does_not_leak():
    """`)` after a `.` used to pop a branch belonging to the previous
    component. The walk's own consistency check would then disagree with
    RDKit, which is the loud failure — but only by luck of ordering."""
    smiles = "CC(C)O.CC(C)O"
    alignment = align_tokens_to_structure(smiles, spans_for(smiles))
    assert len(alignment.atoms.atom_spans) == 8


def test_a_single_component_molecule_is_unaffected():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    alignment = align_tokens_to_structure(smiles, spans_for(smiles))
    assert len(alignment.atoms.atom_spans) == 13
    assert len(alignment.bonds) == 13
