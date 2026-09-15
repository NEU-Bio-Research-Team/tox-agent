"""Molecular featurization for the classical bioactivity baselines.

Two representations, cached per unique SMILES because the same compound recurs
across tasks:

  ECFP4       2048-bit Morgan fingerprint, radius 2 -- the standard QSAR
              baseline representation and a genuinely hard one to beat on
              activity cliffs
  descriptors ~10 interpretable physicochemical properties, used as an
              auxiliary block rather than on their own

The feature schema is versioned. A model artifact records the version it was
trained with, so a schema change cannot silently invalidate a checkpoint.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

LOG = logging.getLogger(__name__)

FEATURE_SCHEMA_VERSION = "toxact-bioactivity-ecfp4-v1"

ECFP_RADIUS = 2
ECFP_BITS = 2048

DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
)


def _generator():
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=ECFP_RADIUS, fpSize=ECFP_BITS
    )


class ContextEncoder:
    """One-hot encoder for assay context, fitted on the training split.

    Assay context is conditioned on rather than split by: a model for
    (target, IC50) sees binding and cell-based rows together, but is told which
    is which. Pooling them without this flag would make protocol differences
    look like chemistry.

    The vocabulary is learned at fit time; a context unseen in training encodes
    as all-zeros plus an explicit `unknown` flag, so an unfamiliar protocol is
    visible to the model rather than silently mapped onto a familiar one.
    """

    def __init__(self) -> None:
        from bioactivity.ingest.task_keys import CONTEXT_FIELDS

        self.fields = CONTEXT_FIELDS
        self.vocab: dict[str, list[str]] = {}
        self._fitted = False

    @property
    def n_features(self) -> int:
        if not self._fitted:
            return 0
        # One column per known level, plus an unknown flag per field, plus the
        # variant-construct indicator.
        return sum(len(v) + 1 for v in self.vocab.values()) + 1

    def fit(self, rows) -> "ContextEncoder":
        levels: dict[str, set[str]] = {field: set() for field in self.fields}
        for row in rows:
            for field in self.fields:
                levels[field].add(str(row.get(field) or "NA"))
        self.vocab = {field: sorted(values) for field, values in levels.items()}
        self._fitted = True
        return self

    def transform(self, rows) -> np.ndarray:
        from bioactivity.ingest.task_keys import is_variant

        if not self._fitted:
            raise RuntimeError("ContextEncoder.transform before fit")

        matrix = np.zeros((len(rows), self.n_features), dtype=np.float32)
        for index, row in enumerate(rows):
            offset = 0
            for field in self.fields:
                levels = self.vocab[field]
                value = str(row.get(field) or "NA")
                if value in levels:
                    matrix[index, offset + levels.index(value)] = 1.0
                else:
                    matrix[index, offset + len(levels)] = 1.0  # unknown flag
                offset += len(levels) + 1
            matrix[index, offset] = 1.0 if is_variant(row) else 0.0
        return matrix


class Featurizer:
    """Caches features per SMILES string."""

    def __init__(self, *, use_descriptors: bool = False) -> None:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        self.use_descriptors = use_descriptors
        self._generator = _generator()
        self._cache: dict[str, np.ndarray | None] = {}

    @property
    def n_features(self) -> int:
        return ECFP_BITS + (len(DESCRIPTOR_NAMES) if self.use_descriptors else 0)

    def _descriptors(self, mol) -> np.ndarray:
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

        values = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            mol.GetNumHeavyAtoms(),
        ]
        array = np.asarray(values, dtype=np.float32)
        # RDKit can emit inf/nan for pathological structures; a feature matrix
        # with a nan in it fails silently in some estimators and loudly in
        # others, so it is normalized here once.
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def featurize_one(self, smiles: str) -> np.ndarray | None:
        if smiles in self._cache:
            return self._cache[smiles]

        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self._cache[smiles] = None
            return None

        fingerprint = self._generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        if self.use_descriptors:
            features = np.concatenate([fingerprint, self._descriptors(mol)])
        else:
            features = fingerprint
        self._cache[smiles] = features
        return features

    def featurize(self, smiles_list: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (matrix, valid_mask). Invalid SMILES are masked, not dropped.

        Returning a mask rather than a filtered matrix keeps the caller's row
        alignment intact -- silently dropping rows here would misalign
        predictions against labels downstream.
        """
        matrix = np.zeros((len(smiles_list), self.n_features), dtype=np.float32)
        valid = np.zeros(len(smiles_list), dtype=bool)
        for row, smiles in enumerate(smiles_list):
            features = self.featurize_one(smiles)
            if features is not None:
                matrix[row] = features
                valid[row] = True
        if not valid.all():
            LOG.warning("%d/%d SMILES could not be featurized",
                        int((~valid).sum()), len(smiles_list))
        return matrix, valid
