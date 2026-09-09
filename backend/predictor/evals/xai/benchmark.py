"""Does an explanation mean anything? (K10)

The predictor returns per-token and per-atom attributions and labels them
`deterministic: true`. Nothing measured whether they are deterministic, whether
they survive the same molecule being typed differently, or whether the atoms
they point at are the atoms the prediction actually depends on. A number
labelled "importance" that fails any of those is a picture, not an
explanation, and the plan is explicit that a gradient-based attribution must
not be presented as causality.

Three properties, each with its own failure mode:

*Determinism.* Two attributions of the same input must be identical. The
metadata already claims this; a claim nothing checks is a claim.

*Invariance to spelling.* One molecule has many valid SMILES. The service
canonicalises before doing anything, so an attribution must not change when
the input is re-spelled. If it does, an explanation shown to a user depends on
how they happened to type the molecule.

*Faithfulness.* Deleting the atoms an attribution calls important must move
the probability more than deleting the same number of arbitrary atoms. This is
the deletion metric; it is a comparison against a random control, never an
absolute score, because "the probability moved" on its own says nothing.

    benchmark.py --out results.json [--limit N] [--endpoint herg]

Needs the real weights (`MODELS_ROOT`). What it produces is a measurement on
one panel with one method, not a claim that the explanations are good.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / "benchmark" / "fixtures" / "golden_panel.json"

#: Fixed, because a benchmark whose control group changes between runs cannot
#: be compared with the previous run.
SEED = 20260909


@dataclass
class MoleculeResult:
    id: str
    smiles: str
    endpoint: str
    task: str | None = None
    deterministic: bool | None = None
    spelling_invariant: bool | None = None
    spelling_max_delta: float | None = None
    #: Probability change from deleting the top-k attributed atoms.
    top_k_delta: float | None = None
    #: The same, deleting k arbitrary atoms. The comparison is the metric.
    random_delta: float | None = None
    faithful: bool | None = None
    skipped: str | None = None


@dataclass
class Summary:
    endpoint: str
    task: str | None
    model_id: str | None
    molecules_attempted: int = 0
    determinism_checked: int = 0
    determinism_failures: list[str] = field(default_factory=list)
    spelling_checked: int = 0
    spelling_failures: list[str] = field(default_factory=list)
    faithfulness_checked: int = 0
    faithfulness_wins: int = 0
    median_top_k_delta: float | None = None
    median_random_delta: float | None = None
    skipped: dict[str, int] = field(default_factory=dict)


def panel(limit: int | None) -> list[dict[str, Any]]:
    entries = json.loads(PANEL.read_text())["valid"]
    return entries[:limit] if limit else entries


def _token_signature(result: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (token["position"], token["token"], token["signed_contribution"], token["magnitude"])
        for token in result["tokens"]
    ]


def _atom_signature(result: dict[str, Any]) -> list[tuple[int, float]]:
    """Atom-level, so it can be compared across two spellings of one molecule.

    Token positions are a property of the string; atom indices are a property
    of the canonical structure, which is the whole point of the alignment step
    the explain service performs.
    """
    return [(atom["atom_index"], round(atom["signed_contribution"], 9)) for atom in result["atoms"]]


def _delete_atoms(smiles: str, indices: set[int]) -> str | None:
    """The molecule without those atoms, or None if that is not one molecule.

    Deletion can disconnect a ring system or leave something RDKit will not
    sanitise. A fragment is a different molecule, so comparing its probability
    to the original would measure the fragmentation rather than the
    attribution; those cases are excluded rather than scored.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    editable = Chem.RWMol(molecule)
    for index in sorted(indices, reverse=True):
        editable.RemoveAtom(index)
    try:
        fragment = editable.GetMol()
        Chem.SanitizeMol(fragment)
    except Exception:  # noqa: BLE001 — an unsanitisable edit is a skip, not a result
        return None
    result = Chem.MolToSmiles(fragment)
    if not result or "." in result:
        # Disconnected: two molecules, not one with atoms removed.
        return None
    return result


def _deletable(smiles: str) -> list[int]:
    """Atoms that can be removed one at a time leaving a single molecule.

    Drug-like molecules are mostly fused rings, so removing three arbitrary
    heavy atoms nearly always fragments them — a first run of this benchmark
    skipped every molecule in the panel for exactly that reason, which is a
    measurement of ring topology and not of the attribution.

    Restricting both the treatment and the control to this set makes the
    question well posed: among the atoms this molecule can lose, do the ones
    the attribution ranks highest move the probability more than the others?
    A candidate set shared by both arms is what keeps that a comparison rather
    than two unrelated numbers.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return []
    return [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if _delete_atoms(smiles, {atom.GetIdx()}) is not None
    ]


def _respell(smiles: str, rng: random.Random) -> str | None:
    """A different, valid SMILES for the same molecule."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    for _ in range(8):
        candidate = Chem.MolToSmiles(molecule, canonical=False, doRandom=True)
        if candidate != smiles:
            return candidate
    return None


def run(
    *, endpoint: str, task: str | None, limit: int | None, top_k: int, device: str
) -> tuple[Summary, list[MoleculeResult]]:
    from toxpred.application.explain import ExplainService
    from toxpred.application.attribution import AttributionService
    from toxpred.scientific.bootstrap import build_registry

    registry = build_registry(device=device)
    explain = ExplainService(AttributionService(registry))
    provider = registry.resolve(capability=endpoint)
    summary = Summary(endpoint=endpoint, task=task, model_id=provider.model_id)
    rng = random.Random(SEED)
    results: list[MoleculeResult] = []
    top_deltas: list[float] = []
    random_deltas: list[float] = []

    for entry in panel(limit):
        result = MoleculeResult(
            id=entry["id"], smiles=entry["canonical_smiles"], endpoint=endpoint, task=task
        )
        summary.molecules_attempted += 1
        first = explain.explain(result.smiles, endpoint, task)
        if first.get("status") == "failed":
            result.skipped = f"explain_failed:{first.get('metadata', {}).get('error')}"
            summary.skipped[result.skipped] = summary.skipped.get(result.skipped, 0) + 1
            results.append(result)
            continue

        # 1. Determinism.
        second = explain.explain(result.smiles, endpoint, task)
        result.deterministic = _token_signature(first) == _token_signature(second)
        summary.determinism_checked += 1
        if not result.deterministic:
            summary.determinism_failures.append(result.id)

        # 2. Invariance to spelling.
        respelled = _respell(result.smiles, rng)
        if respelled is None:
            result.spelling_invariant = None
        else:
            other = explain.explain(respelled, endpoint, task)
            if other.get("status") == "failed":
                result.spelling_invariant = None
            else:
                a, b = dict(_atom_signature(first)), dict(_atom_signature(other))
                if set(a) != set(b):
                    result.spelling_invariant = False
                    result.spelling_max_delta = float("inf")
                else:
                    result.spelling_max_delta = max(
                        (abs(a[i] - b[i]) for i in a), default=0.0
                    )
                    # Same forward and backward pass on the same canonical
                    # structure: this is exact equality, not a tolerance.
                    result.spelling_invariant = result.spelling_max_delta == 0.0
                summary.spelling_checked += 1
                if not result.spelling_invariant:
                    summary.spelling_failures.append(result.id)

        # 3. Faithfulness by deletion, against a random control.
        atoms = first["atoms"]
        candidates = set(_deletable(result.smiles))
        deletable = [atom for atom in atoms if atom["atom_index"] in candidates]
        k = min(top_k, max(1, len(deletable) // 4))
        if len(deletable) < 2 * k:
            # Not enough removable atoms to form both arms without overlap.
            result.skipped = "too_few_deletable_atoms"
            summary.skipped[result.skipped] = summary.skipped.get(result.skipped, 0) + 1
            results.append(result)
            continue
        ranked = sorted(deletable, key=lambda a: -abs(a["signed_contribution"]))
        top_indices = {atom["atom_index"] for atom in ranked[:k]}
        # The control is drawn from the *lowest*-ranked deletable atoms'
        # neighbourhood at random, never from the top-k, so the two arms are
        # disjoint and the comparison is not diluted by overlap.
        pool = [atom["atom_index"] for atom in deletable if atom["atom_index"] not in top_indices]
        random_indices = set(rng.sample(pool, k)) if len(pool) >= k else None

        baseline = first["probability"]
        top_smiles = _delete_atoms(result.smiles, top_indices)
        random_smiles = _delete_atoms(result.smiles, random_indices) if random_indices else None
        if top_smiles is None or random_smiles is None:
            result.skipped = "deletion_left_no_single_molecule"
            summary.skipped[result.skipped] = summary.skipped.get(result.skipped, 0) + 1
            results.append(result)
            continue

        top_after = explain.explain(top_smiles, endpoint, task)
        random_after = explain.explain(random_smiles, endpoint, task)
        if top_after.get("status") == "failed" or random_after.get("status") == "failed":
            result.skipped = "deleted_molecule_not_predictable"
            summary.skipped[result.skipped] = summary.skipped.get(result.skipped, 0) + 1
            results.append(result)
            continue

        result.top_k_delta = abs(baseline - top_after["probability"])
        result.random_delta = abs(baseline - random_after["probability"])
        result.faithful = result.top_k_delta > result.random_delta
        summary.faithfulness_checked += 1
        summary.faithfulness_wins += int(result.faithful)
        top_deltas.append(result.top_k_delta)
        random_deltas.append(result.random_delta)
        results.append(result)

    if top_deltas:
        summary.median_top_k_delta = statistics.median(top_deltas)
        summary.median_random_delta = statistics.median(random_deltas)
    return summary, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="herg", choices=["herg", "tox21"])
    parser.add_argument("--task", default=None, help="required for tox21")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.endpoint == "tox21" and not args.task:
        parser.error("tox21 needs --task: the twelve assays are independent")

    summary, results = run(
        endpoint=args.endpoint, task=args.task, limit=args.limit,
        top_k=args.top_k, device=args.device,
    )
    document = {
        "schema_version": "xai-benchmark-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "top_k": args.top_k,
        "summary": asdict(summary),
        "molecules": [asdict(result) for result in results],
    }
    rendered = json.dumps(document, indent=2)
    if args.out:
        args.out.write_text(rendered + "\n")
        print(f"wrote {args.out}")
    print(json.dumps(document["summary"], indent=2))

    # A failure of determinism or of spelling invariance is a defect: those are
    # properties the service claims. Faithfulness is a measurement, reported
    # rather than gated, because the threshold at which an attribution method
    # is "good enough" is a scientific judgement and not this script's to make.
    return 1 if summary.determinism_failures or summary.spelling_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
