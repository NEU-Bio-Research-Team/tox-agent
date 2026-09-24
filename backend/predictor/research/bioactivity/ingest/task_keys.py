"""The two task granularities, and why they are different.

`task_key` (target, standard_type, assay context) is the AGGREGATION unit. Two
measurements may only be averaged together if they share all three, because a
binding assay on isolated protein and a cell-based functional readout are not
replicates of one another.

`task_unit_key` (target, standard_type) is the MODELLING and REPORTING unit.
Aggregation granularity is not the right modelling granularity: in ChEMBL 37 the
panel's 27 (target, standard_type) units fragment into 295 assay contexts, 153
of which hold fewer than 50 rows. Fitting one model per context sends most of
them to a median fallback, which measures the fallback rather than the method.

Assay context is therefore not discarded -- it is passed to the model as
conditioning features (see `models.featurize.context_features`), which is the
assay-aware approach the plan prescribes: share the encoder, condition on the
context. What must never happen is pooling across `standard_type`, and that
stays separated at both granularities.
"""

from __future__ import annotations

from typing import Any

#: Categorical context axes handed to the model as features.
CONTEXT_FIELDS = ("assay_type", "bao_label")


def task_key(row: dict[str, Any]) -> str:
    """Aggregation unit: target, measurement type and full assay context."""
    return row["task_key"]


def task_unit_key(row: dict[str, Any]) -> str:
    """Modelling and reporting unit: target and measurement type only.

    Must match the key the split manifest's per-task counts and eligibility
    gate are built on, or the macro silently scores nothing.
    """
    return f"{row['target_chembl_id']}|{row['standard_type']}"


def is_variant(row: dict[str, Any]) -> bool:
    """True when the assay used a mutant construct rather than wild type."""
    mutation = (row.get("assay_variant_mutation") or "WT").strip()
    return mutation not in ("", "WT")
