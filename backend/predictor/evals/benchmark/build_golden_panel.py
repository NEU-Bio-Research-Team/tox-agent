#!/usr/bin/env python3
"""Build the golden SMILES panel for predictor regression (plan Phase 0, step 5).

Every structure is validated and canonicalised with RDKit before it is written,
so the panel cannot ship a mis-typed structure.
"""
from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

OUT = Path(__file__).resolve().parent / "fixtures" / "golden_panel.json"

# (id, smiles, group, note)
CASES: list[tuple[str, str, str, str]] = [
    # --- known hERG blockers (positive controls for the hERG head) -----------
    ("herg_pos_astemizole", "COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1", "herg_positive", "withdrawn, hERG"),
    ("herg_pos_terfenadine", "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1", "herg_positive", "withdrawn, hERG"),
    ("herg_pos_cisapride", "COc1cc(C(=O)NC2CCN(CCCOc3ccc(F)cc3)CC2OC)c(N)cc1Cl", "herg_positive", "withdrawn, hERG"),
    ("herg_pos_dofetilide", "CN(CCOc1ccc(NS(C)(=O)=O)cc1)CCc1ccc(NS(C)(=O)=O)cc1", "herg_positive", "class III antiarrhythmic"),
    ("herg_pos_sotalol", "CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1", "herg_positive", "QT prolongation"),
    ("herg_pos_haloperidol", "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1", "herg_positive", "antipsychotic"),
    ("herg_pos_thioridazine", "CSc1ccc2Sc3ccccc3N(CCC3CCCCN3C)c2c1", "herg_positive", "antipsychotic"),
    ("herg_pos_verapamil", "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC", "herg_positive", "calcium blocker"),
    ("herg_pos_amiodarone", "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1", "herg_positive", "iodinated"),
    ("herg_pos_pimozide", "O=C1Nc2ccccc2N1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1", "herg_positive", "antipsychotic"),

    # --- low-liability / widely used drugs ----------------------------------
    ("safe_aspirin", "CC(=O)Oc1ccccc1C(=O)O", "low_liability", "NSAID"),
    ("safe_caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "low_liability", "xanthine"),
    ("safe_ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "low_liability", "NSAID"),
    ("safe_acetaminophen", "CC(=O)Nc1ccc(O)cc1", "low_liability", "analgesic"),
    ("safe_ethanol", "CCO", "low_liability", "small molecule"),
    ("safe_metformin", "CN(C)C(=N)NC(=N)N", "low_liability", "biguanide"),
    ("safe_ascorbic_acid", "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O", "low_liability", "vitamin C"),
    ("safe_glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "low_liability", "sugar"),
    ("safe_penicillin_g", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O", "low_liability", "beta-lactam"),
    ("safe_paracetamol_dup", "CC(=O)Nc1ccc(O)cc1", "duplicate", "same as acetaminophen — determinism check"),

    # --- reference / known-toxic compounds ----------------------------------
    ("ref_thalidomide", "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1", "reference", "teratogen"),
    ("ref_benzene", "c1ccccc1", "reference", "carcinogen"),
    ("ref_aflatoxin_b1", "COc1cc2c(c3c1C1C=COC1O3)C(=O)CC2", "reference", "mycotoxin, fused"),
    ("ref_nicotine", "CN1CCC[C@H]1c1cccnc1", "reference", "alkaloid, stereo"),
    ("ref_bisphenol_a", "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", "reference", "endocrine, Tox21 NR-ER"),
    ("ref_estradiol", "C[C@]12CC[C@H]3[C@@H](CC[C@H]4Cc5ccc(O)cc5[C@H]34)[C@@H]1CC[C@@H]2O", "reference", "Tox21 NR-ER agonist"),
    ("ref_dexamethasone", "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO", "reference", "steroid"),
    ("ref_rotenone", "COc1cc2c(cc1OC)C(=O)C1COc3cc(OC)c(OC)cc3C1O2", "reference", "SR-MMP mitochondrial"),

    # --- stereochemistry / salts / charged ----------------------------------
    ("stereo_quinidine", "COc1ccc2nccc([C@@H](O)[C@H]3C[C@@H]4CCN3C[C@@H]4C=C)c2c1", "stereochemistry", "explicit stereo"),
    ("stereo_quinidine_flat", "COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1", "stereochemistry", "same skeleton, no stereo"),
    ("stereo_ibuprofen_s", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "stereochemistry", "S-enantiomer"),
    ("salt_sodium_salicylate", "O=C([O-])c1ccccc1O.[Na+]", "salt", "multi-fragment"),
    ("salt_diphenhydramine_hcl", "CN(C)CCOC(c1ccccc1)c1ccccc1.Cl", "salt", "HCl salt"),
    ("charged_choline", "C[N+](C)(C)CCO", "charged", "permanent cation"),
    ("charged_glycine_zwitterion", "[NH3+]CC(=O)[O-]", "charged", "zwitterion"),

    # --- rare elements / organometallic (applicability boundary) ------------
    ("rare_cisplatin", "N.N.Cl[Pt]Cl", "rare_element", "Pt complex"),
    ("rare_arsenic_trioxide", "O=[As]O[As]=O", "rare_element", "As"),
    ("rare_ferrocene", "[cH-]1cccc1.[cH-]1cccc1.[Fe+2]", "rare_element", "Fe sandwich"),
    ("rare_selenomethionine", "C[Se]CC[C@H](N)C(=O)O", "rare_element", "Se"),
    ("rare_tetraethyllead", "CC[Pb](CC)(CC)CC", "rare_element", "Pb"),

    # --- size boundary -------------------------------------------------------
    ("size_long_chain", "C" * 120, "long_input", "120 carbons, exceeds max_length=128 tokens"),
    ("size_cyclosporine_like", "CC[C@H]1NC(=O)[C@@H](NC(=O)[C@H](C)NC(=O)[C@@H](C)NC(=O)[C@H](CC(C)C)NC1=O)CC(C)C", "long_input", "peptide macrocycle"),
]

# Inputs that must be rejected — deliberately not RDKit-parseable.
INVALID_CASES: list[tuple[str, str, str]] = [
    ("invalid_garbage", "not_a_smiles", "non-chemical text"),
    ("invalid_unclosed_ring", "c1ccccc", "unclosed aromatic ring"),
    ("invalid_unbalanced_paren", "C(C", "unbalanced parenthesis"),
    ("invalid_bad_valence", "C(C)(C)(C)(C)C", "pentavalent carbon"),
    ("invalid_unknown_element", "[Xx]", "unknown element symbol"),
    ("invalid_open_ring", "C1CC", "unclosed ring bond"),
]

# RDKit happily returns an empty molecule for these, so they must be rejected by
# the request envelope, not by the chemistry layer. Kept in the panel so the API
# contract test can assert a typed 400 rather than an empty prediction.
ENVELOPE_REJECT_CASES: list[tuple[str, str, str]] = [
    ("envelope_empty", "", "empty string — RDKit returns an empty mol"),
    ("envelope_whitespace", "   ", "whitespace only"),
]


def main() -> int:
    valid, failures = [], []
    for case_id, smi, group, note in CASES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append((case_id, smi))
            continue
        valid.append({
            "id": case_id,
            "smiles": smi,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "group": group,
            "note": note,
            "num_atoms": mol.GetNumAtoms(),
            "elements": sorted({a.GetSymbol() for a in mol.GetAtoms()}),
        })

    if failures:
        print("REFUSING TO WRITE — these structures did not parse:")
        for case_id, smi in failures:
            print(f"  {case_id}: {smi}")
        return 1

    invalid = []
    for case_id, smi, note in INVALID_CASES:
        if Chem.MolFromSmiles(smi) is not None:
            print(f"REFUSING TO WRITE — {case_id} was expected to be invalid but parsed: {smi}")
            return 1
        invalid.append({"id": case_id, "smiles": smi, "group": "invalid", "note": note})

    envelope = []
    for case_id, smi, note in ENVELOPE_REJECT_CASES:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.GetNumAtoms() != 0:
            print(f"REFUSING TO WRITE — {case_id} produced a non-empty molecule: {smi}")
            return 1
        envelope.append({"id": case_id, "smiles": smi, "group": "envelope_reject", "note": note})

    panel = {
        "schema_version": 1,
        "description": "Golden SMILES panel for ToxAgent predictor regression.",
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "n_envelope_reject": len(envelope),
        "valid": valid,
        "invalid": invalid,
        "envelope_reject": envelope,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(panel, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} — {len(valid)} valid, {len(invalid)} invalid, {len(envelope)} envelope-reject")
    groups: dict[str, int] = {}
    for row in valid:
        groups[row["group"]] = groups.get(row["group"], 0) + 1
    for g, n in sorted(groups.items()):
        print(f"  {g:18s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
