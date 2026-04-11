from rdkit import Chem
from rdkit.Chem import Descriptors

def get_molecule_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "weight": Descriptors.MolWt(mol),
            "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "valid": True
        }
    return {"valid": False}