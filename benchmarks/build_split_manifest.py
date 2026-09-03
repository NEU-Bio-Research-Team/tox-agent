#!/usr/bin/env python3
"""Freeze the evaluation split (plan Phase 6, steps 2 and 5).

Run once. It reproduces the exact split the serving checkpoint was trained with
— scaffold, seed 42, the values recorded in
``models/pretrained_2head_herg_chemberta_model/config.yaml`` — and writes the
test-set molecules to a manifest with a content hash.

Everything downstream reads the manifest. Nothing re-splits at benchmark time,
so the failure in ``backend/data.py`` cannot recur: when DeepChem is missing it
falls back to PyTDC and silently swaps the scaffold split for a random 80/10/10,
which would put training molecules in the test set.

If a loader is unavailable this script fails. It never substitutes a split.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "manifests" / "eval-split-v1.json"
CACHE_DIR = str(ROOT / "data")
SPLIT_TYPE = "scaffold"
SEED = 42


def canonical(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def build_tox21() -> dict:
    from backend.data import get_task_names, load_tox21

    train, val, test = load_tox21(
        cache_dir=CACHE_DIR, split_type=SPLIT_TYPE, seed=SEED, enforce_workspace_mode=False
    )
    tasks = list(get_task_names("tox21"))
    rows, dropped = [], 0
    for _, row in test.iterrows():
        smi = canonical(str(row["smiles"]))
        if smi is None:
            dropped += 1
            continue
        # NaN means the assay was not run for this molecule — kept as null and
        # masked at scoring time, never coerced to 0.
        labels = {
            t: (None if row[t] != row[t] else int(row[t])) for t in tasks
        }
        rows.append({"canonical_smiles": smi, "labels": labels})
    return {
        "dataset": "tox21",
        "tasks": tasks,
        "split_type": SPLIT_TYPE,
        "seed": SEED,
        "loader": "backend.data.load_tox21 (DeepChem ScaffoldSplitter)",
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "unparseable_dropped": dropped,
        "test": rows,
    }


def build_herg() -> dict:
    from tdc.single_pred import Tox

    data = Tox(name="hERG_Karim", path=CACHE_DIR)
    split = data.get_split(method=SPLIT_TYPE, seed=SEED)
    rows, dropped = [], 0
    for _, row in split["test"].iterrows():
        smi = canonical(str(row["Drug"]))
        if smi is None:
            dropped += 1
            continue
        rows.append({"canonical_smiles": smi, "label": int(row["Y"])})
    return {
        "dataset": "herg_karim",
        "split_type": SPLIT_TYPE,
        "seed": SEED,
        "loader": "tdc.single_pred.Tox('hERG_Karim').get_split (PyTDC scaffold)",
        "counts": {k: len(v) for k, v in split.items()},
        "unparseable_dropped": dropped,
        "test": rows,
    }


def main() -> int:
    print(f"building frozen split (type={SPLIT_TYPE}, seed={SEED}) ...")
    manifest = {
        "schema_version": 1,
        "purpose": "Frozen evaluation split for the ToxPred scientific benchmark.",
        "note": (
            "Test molecules only. Reproduces the split the serving checkpoint was "
            "trained with, so the benchmark cannot score a molecule the model saw."
        ),
        "datasets": {},
    }
    for name, build in (("tox21", build_tox21), ("herg", build_herg)):
        print(f"  {name} ...", end=" ", flush=True)
        try:
            manifest["datasets"][name] = build()
        except Exception as exc:  # noqa: BLE001 — a missing loader must be loud
            print(f"FAILED: {type(exc).__name__}: {exc}")
            print("\nRefusing to write a partial manifest. Install the loader and rerun.")
            return 1
        d = manifest["datasets"][name]
        print(f"{len(d['test'])} test molecules (dropped {d['unparseable_dropped']})")

    payload = json.dumps(manifest, indent=2, sort_keys=True)
    manifest["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"  content_sha256 = {manifest['content_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
