"""Frozen-reference ECFP applicability-domain baseline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SimilarityAssessment:
    nearest_tanimoto: float
    top_k_average: float
    distance_percentile: float
    label: str
    reference_sha256: str
    method: str = "ecfp4_tanimoto_v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method, "nearest_tanimoto": self.nearest_tanimoto,
            "top_k_average": self.top_k_average, "distance_percentile": self.distance_percentile,
            "label": self.label, "reference_sha256": self.reference_sha256,
        }


class EcfpSimilarityDomain:
    def __init__(self, training_smiles: Sequence[str], *, limited_cutoff: float,
                 out_of_domain_cutoff: float, top_k: int = 5) -> None:
        if not training_smiles or not 0 <= out_of_domain_cutoff <= limited_cutoff <= 1:
            raise ValueError("invalid reference set or calibrated similarity cutoffs")
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self._fingerprints = []
        canonical: list[str] = []
        for smiles in training_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"invalid training reference SMILES: {smiles!r}")
            canonical.append(Chem.MolToSmiles(mol, canonical=True))
            self._fingerprints.append(self._generator.GetFingerprint(mol))
        self._limited = limited_cutoff
        self._ood = out_of_domain_cutoff
        self._top_k = max(1, min(top_k, len(self._fingerprints)))
        payload = json.dumps(sorted(canonical), separators=(",", ":"))
        self.reference_sha256 = hashlib.sha256(payload.encode()).hexdigest()

    def assess(self, canonical_smiles: str) -> SimilarityAssessment:
        from rdkit import Chem, DataStructs

        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            raise ValueError("invalid query SMILES")
        query = self._generator.GetFingerprint(mol)
        similarities = sorted(
            (float(value) for value in DataStructs.BulkTanimotoSimilarity(query, self._fingerprints)),
            reverse=True,
        )
        nearest = similarities[0]
        top_average = sum(similarities[: self._top_k]) / self._top_k
        # Percentile of distance among query-to-reference distances.  The
        # baseline is explicit and stable; a release may replace it only with
        # a hash-pinned calibration distribution.
        distances = [1 - value for value in similarities]
        query_distance = 1 - nearest
        percentile = sum(distance <= query_distance for distance in distances) / len(distances)
        label = "in_domain" if nearest >= self._limited else (
            "limited" if nearest >= self._ood else "out_of_domain"
        )
        return SimilarityAssessment(nearest, top_average, percentile, label, self.reference_sha256)
