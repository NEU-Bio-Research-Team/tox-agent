"""Target annotation: protein family and gene symbol for human targets.

The panel rule requires "at least four protein families", so family is part of
the selection criteria and must be derived reproducibly rather than hand-typed.

ChEMBL stores the family hierarchy in `protein_classification` (905 rows) and
links it from `target_component`. Both are small enough to page in full, which
avoids one request per target. `protein_class_desc` already encodes the full
path from the root, double-space separated, so the level-1 family is its first
segment -- no tree walk is needed, though we cross-check against the hierarchy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from bioactivity.ingest.chembl_client import ChemblClient

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetAnnotation:
    target_chembl_id: str
    pref_name: str
    uniprot: str
    gene_symbol: str
    protein_family: str
    protein_class_path: str


def load_protein_classes(client: ChemblClient) -> dict[int, tuple[str, str]]:
    """Map `protein_class_id` -> (level-1 family, full path).

    Returns family as ChEMBL's own level-1 label (enzyme, membrane receptor,
    transporter, ion channel, ...) which is the coarsest consistently populated
    level of the hierarchy.
    """
    records = client.fetch_all("protein_classification", {})
    by_id: dict[int, dict[str, Any]] = {
        int(r["protein_class_id"]): r for r in records
    }

    classes: dict[int, tuple[str, str]] = {}
    for class_id, rec in by_id.items():
        desc = (rec.get("protein_class_desc") or "").strip()
        # Path segments are separated by two spaces; single spaces occur inside
        # a segment ("membrane receptor", "protein kinase").
        segments = [s for s in desc.split("  ") if s]
        if segments:
            family = segments[0]
        else:
            family = _walk_to_level1(class_id, by_id)
        classes[class_id] = (family, desc)

    LOG.info(
        "loaded %d protein classes spanning %d level-1 families",
        len(classes),
        len({f for f, _ in classes.values()}),
    )
    return classes


def _walk_to_level1(class_id: int, by_id: dict[int, dict[str, Any]]) -> str:
    """Fallback for rows with no `protein_class_desc`: climb to class_level 1."""
    seen: set[int] = set()
    current: int | None = class_id
    while current is not None and current not in seen:
        seen.add(current)
        rec = by_id.get(current)
        if rec is None:
            break
        if int(rec.get("class_level") or 0) == 1:
            return str(rec.get("pref_name") or "unclassified")
        parent = rec.get("parent_id")
        current = int(parent) if parent is not None else None
    return "unclassified"


def load_human_components(
    client: ChemblClient,
) -> dict[int, dict[str, Any]]:
    """Map `component_id` -> accession, gene symbol and protein class ids.

    `sequence` is deliberately excluded: it is large, and V1 fixed-target models
    use a learned target embedding rather than the sequence itself.
    """
    records = client.fetch_all(
        "target_component",
        {"organism": "Homo sapiens"},
        fields=[
            "component_id",
            "accession",
            "protein_classifications",
            "target_component_synonyms",
        ],
    )
    components: dict[int, dict[str, Any]] = {}
    for rec in records:
        class_ids = [
            int(c["protein_classification_id"])
            for c in rec.get("protein_classifications") or []
            if c.get("protein_classification_id") is not None
        ]
        components[int(rec["component_id"])] = {
            "accession": rec.get("accession") or "",
            "gene_symbol": _gene_symbol(rec),
            "class_ids": class_ids,
        }
    LOG.info("loaded %d human target components", len(components))
    return components


def _gene_symbol(component: dict[str, Any]) -> str:
    """Preferred HGNC-style gene symbol, falling back to any reported symbol."""
    fallback = ""
    for syn in component.get("target_component_synonyms") or []:
        syn_type = syn.get("syn_type")
        name = syn.get("component_synonym") or ""
        if syn_type == "GENE_SYMBOL":
            return name
        if syn_type == "GENE_SYMBOL_OTHER" and not fallback:
            fallback = name
    return fallback


def annotate_targets(
    client: ChemblClient,
    targets: list[dict[str, Any]],
) -> dict[str, TargetAnnotation]:
    """Attach family, gene symbol and UniProt accession to each target.

    `targets` must carry `target_chembl_id`, `pref_name` and `target_components`
    as returned by the `target` endpoint.
    """
    classes = load_protein_classes(client)
    components = load_human_components(client)

    annotations: dict[str, TargetAnnotation] = {}
    unclassified = 0
    for target in targets:
        component_ids = [
            int(c["component_id"])
            for c in target.get("target_components") or []
            if c.get("component_id") is not None
        ]

        family, path, accession, gene = "unclassified", "", "", ""
        for component_id in component_ids:
            component = components.get(component_id)
            if component is None:
                continue
            accession = accession or component["accession"]
            gene = gene or component["gene_symbol"]
            for class_id in component["class_ids"]:
                resolved = classes.get(class_id)
                if resolved and resolved[0]:
                    family, path = resolved
                    break
            if family != "unclassified":
                break

        if family == "unclassified":
            unclassified += 1

        target_id = target["target_chembl_id"]
        annotations[target_id] = TargetAnnotation(
            target_chembl_id=target_id,
            pref_name=target.get("pref_name") or "",
            uniprot=accession,
            gene_symbol=gene,
            protein_family=family,
            protein_class_path=path,
        )

    LOG.info(
        "annotated %d targets (%d unclassified)", len(annotations), unclassified
    )
    return annotations
