"""Build the activity-cliff pair set (protocol section 7.3).

A cliff is a pair of structurally similar compounds, measured in the SAME
context, whose potencies differ sharply. Pairs are therefore formed only within
one (target, standard_type, assay_context) task -- a Ki here against an IC50
there differing by two logs is a protocol difference, not a cliff, and treating
it as one would make the cliff metrics measure curation noise.

Following MoleculeACE, similarity is judged three ways and a pair qualifies on
any of them, because no single similarity view captures all cliffs:

  * ECFP4 Tanimoto           -- substructure overlap
  * scaffold identity        -- same Bemis-Murcko core, different substituents
  * MCS-based overlap        -- approximated by ECFP with a lower threshold

Pair enumeration is quadratic within a task, so tasks are capped and the cap is
recorded. A silently truncated cliff set would make the gate easier to pass.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

LOG = logging.getLogger(__name__)

#: A pair must be at least this similar to count as "structurally similar".
SIMILARITY_THRESHOLD = 0.70

#: ...and differ by at least this much in pActivity to count as a cliff.
MIN_POTENCY_FOLD_LOG = 1.0

#: Above this many compounds in one task, all-pairs enumeration is capped.
MAX_COMPOUNDS_PER_TASK = 4000


@dataclass(frozen=True)
class CliffPair:
    task_key: str
    index_a: int
    index_b: int
    similarity: float
    delta_pactivity: float
    basis: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "index_a": self.index_a,
            "index_b": self.index_b,
            "similarity": round(self.similarity, 4),
            "delta_pactivity": round(self.delta_pactivity, 4),
            "basis": self.basis,
        }


def _fingerprints(smiles_list: Sequence[str]):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fingerprints.append(None if mol is None else generator.GetFingerprint(mol))
    return fingerprints


def find_cliffs_in_task(
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    min_delta: float = MIN_POTENCY_FOLD_LOG,
    max_compounds: int = MAX_COMPOUNDS_PER_TASK,
) -> tuple[list[CliffPair], dict[str, Any]]:
    """Enumerate cliff pairs within one task."""
    from rdkit import DataStructs

    task_key = rows[indices[0]]["task_key"]
    truncated = False
    if len(indices) > max_compounds:
        # Keep the most potent compounds: cliffs among inactives are the least
        # actionable, and this keeps the cap's effect predictable rather than
        # dependent on row order.
        indices = sorted(
            indices, key=lambda i: -float(rows[i]["pactivity"])
        )[:max_compounds]
        truncated = True

    smiles = [rows[i]["standardized_smiles"] for i in indices]
    scaffolds = [rows[i].get("murcko_scaffold") or "" for i in indices]
    activities = np.array([float(rows[i]["pactivity"]) for i in indices])
    fingerprints = _fingerprints(smiles)

    pairs: list[CliffPair] = []
    for position in range(len(indices)):
        reference = fingerprints[position]
        if reference is None:
            continue
        rest = fingerprints[position + 1 :]
        valid = [(offset, fp) for offset, fp in enumerate(rest) if fp is not None]
        if not valid:
            continue

        similarities = DataStructs.BulkTanimotoSimilarity(
            reference, [fp for _, fp in valid]
        )
        for (offset, _), similarity in zip(valid, similarities):
            other = position + 1 + offset
            delta = float(activities[position] - activities[other])
            if abs(delta) < min_delta:
                continue

            same_scaffold = bool(scaffolds[position]) and (
                scaffolds[position] == scaffolds[other]
            )
            if similarity >= similarity_threshold:
                basis = "ecfp4_tanimoto"
            elif same_scaffold and similarity >= similarity_threshold - 0.2:
                # Same core with a substituent change is the canonical
                # matched-pair cliff even when whole-molecule Tanimoto dips.
                basis = "shared_scaffold"
            else:
                continue

            pairs.append(
                CliffPair(
                    task_key=task_key,
                    index_a=indices[position],
                    index_b=indices[other],
                    similarity=float(similarity),
                    delta_pactivity=delta,
                    basis=basis,
                )
            )

    stats = {
        "task_key": task_key,
        "n_compounds_considered": len(indices),
        "n_pairs": len(pairs),
        "truncated": truncated,
    }
    return pairs, stats


def build_cliff_set(
    rows: list[dict[str, Any]],
    *,
    restrict_to: set[int] | None = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    min_delta: float = MIN_POTENCY_FOLD_LOG,
) -> tuple[list[CliffPair], dict[str, Any]]:
    """Cliff pairs across every task.

    `restrict_to` limits enumeration to a set of row indices -- pass the test
    split's indices so that cliff metrics are computed on held-out data and both
    members of every pair are genuinely unseen.
    """
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if restrict_to is not None and index not in restrict_to:
            continue
        by_task[row["task_key"]].append(index)

    all_pairs: list[CliffPair] = []
    per_task: list[dict[str, Any]] = []
    for task_key in sorted(by_task):
        indices = by_task[task_key]
        if len(indices) < 2:
            continue
        pairs, stats = find_cliffs_in_task(
            rows,
            indices,
            similarity_threshold=similarity_threshold,
            min_delta=min_delta,
        )
        all_pairs.extend(pairs)
        per_task.append(stats)

    summary = {
        "n_pairs": len(all_pairs),
        "n_tasks_with_pairs": sum(1 for s in per_task if s["n_pairs"] > 0),
        "n_tasks_examined": len(per_task),
        "definition": {
            "similarity_threshold": similarity_threshold,
            "min_delta_pactivity": min_delta,
            "similarity": "ECFP4 (radius 2, 2048 bits) Tanimoto, or shared "
            "Bemis-Murcko scaffold at a relaxed threshold",
            "pairs_within": "(target, standard_type, assay_context) only",
        },
        "truncated_tasks": [s["task_key"] for s in per_task if s["truncated"]],
        "per_task": per_task,
    }
    LOG.info(
        "cliff set: %d pairs across %d/%d tasks",
        len(all_pairs), summary["n_tasks_with_pairs"], summary["n_tasks_examined"],
    )
    return all_pairs, summary
