"""Atom-level explanation (plan section 5.1).

Wraps :class:`AttributionService`. The attribution service still owns the
gradient computation and its ``completed`` / ``partial`` / ``failed`` /
timeout semantics; this layer only projects the per-token importances onto
heavy-atom indices via the deterministic SMILES walk in
``scientific.featurization.token_atom_align``.

The service stays numeric-only. It returns no image and imports no plotting
library — the 2D highlighted depiction is the frontend's job (D-XAI-3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..scientific.featurization.token_atom_align import (
    ATOM_ORDER_VERSION,
)
from ..scientific.featurization.token_structure_align import (
    STRUCTURE_ORDER_VERSION,
    align_tokens_to_structure,
)
from .depiction import xai_svg
from .attribution import AttributionService

TOKEN_ALIGN_METHOD = "token_structure_align_v2"


@dataclass(frozen=True)
class ExplainService:
    attribution: AttributionService

    def explain(
        self, smiles: str, endpoint: str, task: str | None = None,
        method: str = "grad_x_input",
    ) -> dict[str, Any]:
        raw = (
            self.attribution.attribute(smiles, endpoint, task)
            if method == "grad_x_input"
            else self.attribution.attribute(smiles, endpoint, task, method=method)
        )

        if raw.get("status") == "failed":
            # No tokens, no probability — pass the failure through unchanged
            # apart from the atom-level fields, which are simply empty.
            return {
                "status": "failed",
                "endpoint": endpoint,
                "task": task,
                "input_smiles": raw.get("input_smiles", smiles),
                "canonical_smiles": raw.get("canonical_smiles"),
                "atom_order_version": ATOM_ORDER_VERSION,
                "structure_order_version": STRUCTURE_ORDER_VERSION,
                "probability": None,
                "atoms": [],
                "bonds": [],
                "depiction_svg": None,
                "depiction": None,
                "unmapped_importance": None,
                "tokens": raw.get("tokens", []),
                "method": f"{TOKEN_ALIGN_METHOD}",
                "metadata": {
                    "error": raw.get("error"),
                    "message": raw.get("message"),
                    "duration_ms": raw.get("duration_ms"),
                    "deterministic": True,
                },
            }

        tokens = raw["tokens"]
        canonical = raw["canonical_smiles"]
        alignment = align_tokens_to_structure(
            canonical, [tuple(token["offsets"]) for token in tokens]
        )

        atom_importance = [0.0] * len(alignment.atoms.atom_spans)
        bond_importance = [0.0] * len(alignment.bonds)
        atom_signed = [0.0] * len(alignment.atoms.atom_spans)
        bond_signed = [0.0] * len(alignment.bonds)
        unmapped = 0.0
        unmapped_signed = 0.0
        for token, atom_indices, bond_indices in zip(tokens, alignment.atoms.token_atoms, alignment.token_bonds):
            importance = float(token["importance"])
            signed_contribution = float(token.get("signed_contribution", importance))
            targets = len(atom_indices) + len(bond_indices)
            if targets:
                share = importance / targets
                signed_share = signed_contribution / targets
                for atom_index in atom_indices:
                    atom_importance[atom_index] += share
                    atom_signed[atom_index] += signed_share
                for bond_index in bond_indices:
                    bond_importance[bond_index] += share
                    bond_signed[bond_index] += signed_share
            else:
                unmapped += importance
                unmapped_signed += signed_contribution

        total = sum(atom_importance) + sum(bond_importance) + unmapped
        denominator = total or 1.0
        atoms = [
            {
                "atom_index": span.atom_index,
                "symbol": span.symbol,
                "importance": atom_importance[span.atom_index],
                "magnitude": atom_importance[span.atom_index],
                "signed_contribution": atom_signed[span.atom_index],
                "relative_importance": atom_importance[span.atom_index] / denominator,
            }
            for span in alignment.atoms.atom_spans
        ]
        bonds = []
        for span in alignment.bonds:
            direct = bond_importance[span.bond_index]
            adjacent = (atom_importance[span.begin_atom_index] + atom_importance[span.end_atom_index]) / 2
            bonds.append({
                "bond_index": span.bond_index,
                "begin_atom_index": span.begin_atom_index,
                "end_atom_index": span.end_atom_index,
                "bond_type": span.bond_type,
                "importance": direct,
                "magnitude": direct,
                "signed_contribution": bond_signed[span.bond_index],
                "relative_importance": direct / denominator,
                "display_importance": (direct if direct else adjacent) / denominator,
                "source": "explicit_token" if direct else "adjacent_atom_derived",
            })
        try:
            depiction_svg, depiction = xai_svg(canonical, atoms, bonds)
        except Exception as exc:  # a numeric artifact remains complete without a drawable SVG
            depiction_svg, depiction = None, {"error": type(exc).__name__}

        metadata = raw.get("metadata", {})
        return {
            "status": raw["status"],  # completed | partial
            "endpoint": endpoint,
            "task": task,
            "input_smiles": raw["input_smiles"],
            "canonical_smiles": canonical,
            "atom_order_version": ATOM_ORDER_VERSION,
            "structure_order_version": STRUCTURE_ORDER_VERSION,
            "probability": raw["probability"],
            "atoms": atoms,
            "bonds": bonds,
            "depiction_svg": depiction_svg,
            "depiction": depiction,
            "unmapped_importance": unmapped / denominator,
            "unmapped_signed_contribution": unmapped_signed,
            "signed_contribution_total": sum(atom_signed) + sum(bond_signed) + unmapped_signed,
            "tokens": tokens,
            "method": f"{metadata.get('method', 'unknown')}+{TOKEN_ALIGN_METHOD}",
            "metadata": {
                "model_id": metadata.get("model_id"),
                "target": metadata.get("target", "logit"),
                "mapping_version": metadata.get("mapping_version"),
                "deterministic": True,
                "duration_ms": metadata.get("duration_ms"),
                "note": metadata.get("note"),
            },
        }
