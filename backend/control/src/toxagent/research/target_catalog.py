"""The targets a subject-bound search can be about, versioned (AG-EVIDENCE).

A search about a screening is a search about a ligand on a *target*, and
whether a paper is about that target is decided here, not by the words of the
query: the query is the model's, the subject is the server's. A target the
catalog does not hold is not guessed at -- it is reported as
``not_in_catalog``, and nothing found for it is promoted as direct evidence.

Each entry names what the literature calls the target (the vocabulary a
relevance check looks for), the identifiers that pin it, which toxicity
endpoint it is the target of when there is one (KCNH2 is hERG), and whether
its live ChEMBL mapping has been checked against the service. An unverified
live mapping is not used for a live lookup; frozen corpora are read the same
way either way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

CATALOG_VERSION = "benchmark-targets-v1"


@dataclass(frozen=True, slots=True)
class CatalogTarget:
    target_id: str
    name: str
    uniprot_id: str
    #: Terms a paper about this target contains. Not synonyms for the model.
    vocabulary: tuple[str, ...]
    chembl_target_id: str | None = None
    #: The toxicity endpoint this target is the molecular target of, if any.
    toxicity_endpoint: str | None = None
    #: The ChEMBL id was checked against the live service.
    live_mapping_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id, "name": self.name, "uniprot_id": self.uniprot_id,
            "chembl_target_id": self.chembl_target_id,
            "toxicity_endpoint": self.toxicity_endpoint,
            "live_mapping_verified": self.live_mapping_verified,
            "catalog_version": CATALOG_VERSION,
        }


TARGETS: Mapping[str, CatalogTarget] = {
    target.target_id: target
    for target in (
        CatalogTarget(
            target_id="KCNH2", name="Potassium voltage-gated channel subfamily H member 2 (hERG)",
            uniprot_id="Q12809",
            vocabulary=("herg", "kcnh2", "kv11.1", "ikr", "rapid delayed rectifier",
                        "potassium channel", "qt prolongation", "patch clamp"),
            chembl_target_id="CHEMBL240", toxicity_endpoint="herg",
            # research/providers/chembl.py: checked live on 2026-09-26.
            live_mapping_verified=True,
        ),
        CatalogTarget(
            target_id="EGFR", name="Epidermal growth factor receptor", uniprot_id="P00533",
            vocabulary=("egfr", "epidermal growth factor receptor", "erbb1", "her1",
                        "erbb-1"),
            chembl_target_id="CHEMBL203",
        ),
        CatalogTarget(
            target_id="ABL1", name="Tyrosine-protein kinase ABL1", uniprot_id="P00519",
            vocabulary=("abl1", "abl kinase", "bcr-abl", "bcr abl", "abelson"),
            chembl_target_id="CHEMBL1862",
        ),
    )
}


def lookup(target_id: str | None, *, uniprot_id: str | None = None) -> CatalogTarget | None:
    """The catalog entry for a screening's target. A UniProt id that disagrees
    with the entry's is not the same protein, whatever the target id says."""
    if not target_id:
        return None
    entry = TARGETS.get(target_id.strip().upper())
    if entry is None:
        return None
    if uniprot_id and uniprot_id != entry.uniprot_id:
        return None
    return entry


def describe() -> dict[str, Any]:
    return {
        "catalog_version": CATALOG_VERSION,
        "targets": {k: v.to_dict() for k, v in sorted(TARGETS.items())},
    }
