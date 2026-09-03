"""SMILES resolution: canonicalise once, reject invalid input loudly."""
import pytest

from toxpred.domain.molecule import InvalidSmilesError
from toxpred.scientific.applicability import assess
from toxpred.scientific.featurization.rdkit_resolver import MAX_SMILES_LENGTH, resolve


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"),
    ("OC(=O)c1ccccc1OC(C)=O", "CC(=O)Oc1ccccc1C(=O)O"),   # same molecule, written differently
    ("CCO", "CCO"),
    ("OCC", "CCO"),
])
def test_equivalent_inputs_canonicalise_to_one_string(smiles, expected):
    assert resolve(smiles).canonical_smiles == expected


def test_input_smiles_is_preserved_alongside_the_canonical_form():
    molecule = resolve("OCC")
    assert molecule.input_smiles == "OCC"
    assert molecule.canonical_smiles == "CCO"


@pytest.mark.parametrize("bad", ["", "   ", "not_a_smiles", "C(C", "c1ccccc", "[Xx]", "C1CC"])
def test_invalid_input_raises_rather_than_predicting_zero(bad):
    with pytest.raises(InvalidSmilesError):
        resolve(bad)


def test_overlong_input_is_rejected_before_parsing():
    with pytest.raises(InvalidSmilesError, match="exceeds"):
        resolve("C" * (MAX_SMILES_LENGTH + 1))


def test_elements_are_reported():
    assert resolve("CC(=O)Oc1ccccc1C(=O)O").elements == frozenset({"C", "O"})


# --- applicability ---------------------------------------------------------

def test_common_organic_molecule_is_in_scope():
    assessment = assess(resolve("CC(=O)Oc1ccccc1C(=O)O"))
    assert assessment.status == "ok"
    assert assessment.method == "element_rules_v1"


def test_ok_status_does_not_claim_distributional_support():
    reasons = " ".join(assess(resolve("CCO")).reasons)
    assert "cannot confirm" in reasons


@pytest.mark.parametrize("smiles,status", [
    ("N.N.Cl[Pt]Cl", "out_of_domain"),
    ("O=[As]O[As]=O", "out_of_domain"),
    ("CC[Pb](CC)(CC)CC", "out_of_domain"),
    ("C[Se]CC[C@H](N)C(=O)O", "limited"),
    ("[cH-]1cccc1.[cH-]1cccc1.[Fe+2]", "limited"),
])
def test_unusual_elements_are_flagged(smiles, status):
    assert assess(resolve(smiles)).status == status
