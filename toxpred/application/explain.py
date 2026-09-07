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
    align_tokens_to_atoms,
)
from .attribution import AttributionService

TOKEN_ALIGN_METHOD = "token_atom_align_v1"


@dataclass(frozen=True)
class ExplainService:
    attribution: AttributionService

    def explain(
        self, smiles: str, endpoint: str, task: str | None = None
    ) -> dict[str, Any]:
        raw = self.attribution.attribute(smiles, endpoint, task)

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
                "probability": None,
                "atoms": [],
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
        alignment = align_tokens_to_atoms(
            canonical, [tuple(token["offsets"]) for token in tokens]
        )

        atom_importance = [0.0] * len(alignment.atom_spans)
        unmapped = 0.0
        for token, atom_indices in zip(tokens, alignment.token_atoms):
            importance = float(token["importance"])
            if atom_indices:
                share = importance / len(atom_indices)
                for atom_index in atom_indices:
                    atom_importance[atom_index] += share
            else:
                unmapped += importance

        total = sum(atom_importance) + unmapped
        denominator = total or 1.0
        atoms = [
            {
                "atom_index": span.atom_index,
                "symbol": span.symbol,
                "importance": atom_importance[span.atom_index],
                "relative_importance": atom_importance[span.atom_index] / denominator,
            }
            for span in alignment.atom_spans
        ]

        metadata = raw.get("metadata", {})
        return {
            "status": raw["status"],  # completed | partial
            "endpoint": endpoint,
            "task": task,
            "input_smiles": raw["input_smiles"],
            "canonical_smiles": canonical,
            "atom_order_version": ATOM_ORDER_VERSION,
            "probability": raw["probability"],
            "atoms": atoms,
            "unmapped_importance": unmapped / denominator,
            "tokens": tokens,
            "method": f"{metadata.get('method', 'unknown')}+{TOKEN_ALIGN_METHOD}",
            "metadata": {
                "model_id": metadata.get("model_id"),
                "deterministic": True,
                "duration_ms": metadata.get("duration_ms"),
                "note": metadata.get("note"),
            },
        }
