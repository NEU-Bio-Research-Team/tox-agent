"""Structure standardization for the bioactivity data contract.

Every record keeps three identities (data contract, section 3.4):

  1. ``submitted_smiles``      -- exactly what ChEMBL returned, never mutated
  2. ``standardized_smiles``   -- parent structure, stereochemistry preserved
  3. ``connectivity_key``      -- InChIKey skeleton block, used for split grouping

The third exists because stereoisomers and salts of the same scaffold are near
duplicates. Grouping splits on stereo-insensitive connectivity keeps a compound's
enantiomer from landing in test while the compound itself trains, which would
inflate every cold-chemistry claim in the benchmark.

Isotopes and metal-containing structures are *flagged*, not silently repaired:
dropping them quietly would misstate the applicability domain, and "fixing" them
would invent a structure the assay never measured.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

LOG = logging.getLogger(__name__)

STANDARDIZER_VERSION = "toxact-standardizer-v1"

#: Organic-chemistry element set. Anything outside it marks the record as
#: outside the model's applicability domain rather than excluding it outright.
ORGANIC_ELEMENTS = frozenset(
    {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
)

#: Counterions that routinely appear as the salt form of an otherwise organic
#: drug. These are *not* treated as coordination metals: the measured entity is
#: the organic fragment, and discarding the counterion is the correct parent
#: choice. Keeping this set separate from the metal check is what stops sodium
#: aspirinate from being rejected alongside cisplatin.
SALT_COUNTERION_ELEMENTS = frozenset(
    {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Al", "Zn"}
)


@dataclass(frozen=True)
class StandardizedMolecule:
    submitted_smiles: str
    standardized_smiles: str | None
    inchikey: str | None
    connectivity_key: str | None
    murcko_scaffold: str | None
    num_heavy_atoms: int | None
    mol_weight: float | None
    has_isotope: bool
    has_nonorganic_element: bool
    has_coordination_metal: bool
    had_salt_counterion: bool
    was_mixture: bool
    charge_normalized: bool
    status: str
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Pipeline:
    """Holds the RDKit standardization objects, which are costly to rebuild."""

    def __init__(self) -> None:
        self.largest_fragment = rdMolStandardize.LargestFragmentChooser()
        self.uncharger = rdMolStandardize.Uncharger()
        # Default transform set; this constructor takes no parameters object.
        self.normalizer = rdMolStandardize.Normalizer()


_PIPELINE: _Pipeline | None = None


def _pipeline() -> _Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = _Pipeline()
    return _PIPELINE


def _fail(smiles: str, reason: str) -> StandardizedMolecule:
    return StandardizedMolecule(
        submitted_smiles=smiles,
        standardized_smiles=None,
        inchikey=None,
        connectivity_key=None,
        murcko_scaffold=None,
        num_heavy_atoms=None,
        mol_weight=None,
        has_isotope=False,
        has_nonorganic_element=False,
        has_coordination_metal=False,
        had_salt_counterion=False,
        was_mixture=False,
        charge_normalized=False,
        status="rejected",
        reason=reason,
    )


def standardize_smiles(smiles: str | None) -> StandardizedMolecule:
    """Standardize one SMILES string, reporting why it failed if it did."""
    if not smiles or not str(smiles).strip():
        return _fail(smiles or "", "empty_smiles")

    smiles = str(smiles).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _fail(smiles, "unparseable")
    if mol.GetNumAtoms() == 0:
        return _fail(smiles, "no_atoms")

    pipeline = _pipeline()

    # Flags describe the SUBMITTED entity, so they are computed before any
    # fragment is discarded. Doing this afterwards misreports metal complexes:
    # for cisplatin (N.N.Cl[Pt]Cl) the largest fragment by atom count is
    # ammonia, which would both lose the platinum and clear the flag.
    submitted_elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    has_isotope = any(atom.GetIsotope() != 0 for atom in mol.GetAtoms())
    # A coordination metal is a non-organic element that is not a routine salt
    # counterion. Sodium aspirinate and cisplatin both contain a metal, but only
    # one of them has an organic fragment that *is* the measured compound.
    coordination_metals = (
        submitted_elements - ORGANIC_ELEMENTS - SALT_COUNTERION_ELEMENTS
    )
    has_coordination_metal = bool(coordination_metals)
    had_salt_counterion = bool(submitted_elements & SALT_COUNTERION_ELEMENTS)

    try:
        # Normalize functional groups to a canonical form (nitro, azide, ...)
        # before any fragment choice, so fragment comparison is consistent.
        mol = pipeline.normalizer.normalize(mol)
        mol = rdMolStandardize.Cleanup(mol)

        was_mixture = len(Chem.GetMolFrags(mol)) > 1
        # Largest-fragment choice assumes one fragment is the measured compound
        # and the rest are counterions. That assumption does not hold for metal
        # coordination complexes, so those keep their full structure and are
        # flagged out of the applicability domain instead of being "repaired".
        if was_mixture and not has_coordination_metal:
            mol = pipeline.largest_fragment.choose(mol)

        charge_before = Chem.GetFormalCharge(mol)
        uncharged = pipeline.uncharger.uncharge(mol)
        if uncharged is not None:
            mol = uncharged
        charge_normalized = Chem.GetFormalCharge(mol) != charge_before

        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit raises a range of sanitization errors
        return _fail(smiles, f"standardization_error: {type(exc).__name__}")

    if mol.GetNumAtoms() == 0:
        return _fail(smiles, "empty_after_standardization")

    # Stereochemistry is kept: it is part of the measured entity.
    standardized = Chem.MolToSmiles(mol, isomericSmiles=True)
    if not standardized:
        return _fail(smiles, "no_canonical_smiles")

    try:
        inchikey = Chem.MolToInchiKey(mol) or None
    except Exception:
        inchikey = None
    # The InChIKey's first block encodes connectivity only -- no stereo, no
    # protonation -- which is exactly the grouping granularity splits need.
    connectivity_key = inchikey.split("-")[0] if inchikey else None

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = None

    # `ok` means "usable as a model input". `flagged` means the structure was
    # standardized faithfully but sits outside the applicability domain the
    # panel was built for; extraction keeps these out of HQ-Exact and reports
    # them in the filter ledger rather than dropping them without trace.
    # Judge the PARENT, not the submitted mixture: once a sodium counterion is
    # stripped, what remains is an ordinary organic compound and belongs in
    # HQ-Exact. A coordination complex keeps its metal and stays flagged.
    parent_elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    has_nonorganic_element = bool(parent_elements - ORGANIC_ELEMENTS)

    status = "flagged" if (has_nonorganic_element or has_isotope) else "ok"
    reason = None
    if has_coordination_metal:
        reason = "coordination_metal:" + ",".join(sorted(coordination_metals))
    elif has_nonorganic_element:
        reason = "nonorganic_element:" + ",".join(
            sorted(parent_elements - ORGANIC_ELEMENTS)
        )
    elif has_isotope:
        reason = "isotopic_label"

    return StandardizedMolecule(
        submitted_smiles=smiles,
        standardized_smiles=standardized,
        inchikey=inchikey,
        connectivity_key=connectivity_key,
        murcko_scaffold=scaffold or None,
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        mol_weight=round(float(Descriptors.MolWt(mol)), 4),
        has_isotope=has_isotope,
        has_nonorganic_element=has_nonorganic_element,
        has_coordination_metal=has_coordination_metal,
        had_salt_counterion=had_salt_counterion,
        was_mixture=was_mixture,
        charge_normalized=charge_normalized,
        status=status,
        reason=reason,
    )


def standardize_many(
    smiles_list: list[str],
) -> tuple[dict[str, StandardizedMolecule], dict[str, int]]:
    """Standardize a list of SMILES once per unique string.

    ChEMBL repeats the same `canonical_smiles` across many activity rows, so
    caching by input string avoids re-running RDKit tens of thousands of times.
    Returns the cache plus a tally of rejection reasons for the manifest.
    """
    cache: dict[str, StandardizedMolecule] = {}
    reasons: dict[str, int] = {}
    for smiles in smiles_list:
        key = str(smiles).strip() if smiles else ""
        if key in cache:
            continue
        result = standardize_smiles(key)
        cache[key] = result
        if result.status != "ok":
            reasons[result.reason or "unknown"] = reasons.get(result.reason or "unknown", 0) + 1
    return cache, reasons
