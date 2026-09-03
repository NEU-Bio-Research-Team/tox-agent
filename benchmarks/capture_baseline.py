#!/usr/bin/env python3
"""Capture the pre-refactor baseline (plan Phase 0, steps 3-7).

Records the runtime fingerprint, every serving artifact's SHA-256, and the RAW
probabilities of the currently-servable models over the golden panel. Raw floats
are stored, not labels, so a refactor can be checked for numeric parity rather
than for agreement after thresholding.

Run:  python benchmarks/capture_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent
MANIFEST = OUT_DIR / "manifests" / "baseline-e6882b2.json"
GOLDEN = OUT_DIR / "golden" / "baseline_predictions.json"
PANEL = OUT_DIR / "fixtures" / "golden_panel.json"

SERVING_ARTIFACTS = {
    "clintox-smilesgnn": ROOT / "models" / "smilesgnn_model",
    "herg-tox21-chemberta": ROOT / "models" / "pretrained_2head_herg_chemberta_model",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_dir(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    files = {}
    for f in sorted(path.rglob("*")):
        if f.is_file():
            files[str(f.relative_to(path))] = {"sha256": sha256(f), "bytes": f.stat().st_size}
    return {"present": True, "path": str(path.relative_to(ROOT)), "files": files}


def runtime_fingerprint() -> dict:
    def ver(mod: str) -> str | None:
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    import torch

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = None

    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "numpy": ver("numpy"),
        "transformers": ver("transformers"),
        "rdkit": ver("rdkit"),
        "torch_geometric": ver("torch_geometric"),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def capture_chemberta(panel: dict) -> dict:
    """Raw hERG + Tox21 head outputs. No thresholding, no label derivation."""
    import numpy as np
    import torch
    from backend.inference import load_pretrained_dual_head_bundle

    model_dir = SERVING_ARTIFACTS["herg-tox21-chemberta"]
    t0 = time.perf_counter()
    bundle = load_pretrained_dual_head_bundle(model_dir, device="cpu")
    load_s = time.perf_counter() - t0

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    task_names = list(bundle["task_names"])
    max_length = int(bundle.get("max_length", 128))
    model.eval()

    rows, latencies = {}, []
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    for case in panel["valid"]:
        mol = Chem.MolFromSmiles(case["smiles"])
        canonical = Chem.MolToSmiles(mol)
        t0 = time.perf_counter()
        enc = tokenizer(
            [canonical], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        with torch.inference_mode():
            heads = model.forward_heads(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            )
            herg = torch.sigmoid(heads["herg_logits"]).cpu().numpy().reshape(-1)
            tox21 = torch.sigmoid(heads["tox21_logits"]).cpu().numpy().reshape(-1)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        rows[case["id"]] = {
            "input_smiles": case["smiles"],
            "canonical_smiles": canonical,
            "n_tokens": int(enc["input_ids"].shape[1]),
            "truncated": bool(enc["input_ids"].shape[1] >= max_length),
            "herg_probability_blocker": float(herg[0]),
            "tox21_probability_activity": {
                task: float(tox21[i]) for i, task in enumerate(task_names)
            },
        }

    lat = np.asarray(latencies)
    return {
        "model_id": "herg-tox21-chemberta-v1",
        "loaded": True,
        "load_seconds": round(load_s, 3),
        "task_names": task_names,
        "max_length": max_length,
        "artifact_herg_threshold": float(bundle.get("herg_threshold")),
        "artifact_tox21_thresholds": {
            k: float(v) for k, v in (bundle.get("tox21_thresholds") or {}).items()
        } if isinstance(bundle.get("tox21_thresholds"), dict) else None,
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "n": int(lat.size),
        },
        "predictions": rows,
    }


def capture_clintox(panel: dict) -> dict:
    """Attempt the ClinTox SMILES-GNN path. Records the failure verbatim if it
    cannot load — a missing baseline is a Phase 0 stop condition, not a warning
    to be swallowed."""
    from backend.inference import load_model

    try:
        load_model(
            SERVING_ARTIFACTS["clintox-smilesgnn"],
            ROOT / "config" / "smilesgnn_config.yaml",
            device="cpu",
            enforce_workspace_mode=False,
        )
    except Exception as exc:  # noqa: BLE001 - the message is the deliverable
        return {
            "model_id": "clintox-smilesgnn-v1",
            "loaded": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "predictions": {},
        }
    return {"model_id": "clintox-smilesgnn-v1", "loaded": True, "predictions": {}}


def main() -> int:
    panel = json.loads(PANEL.read_text())
    print(f"panel: {panel['n_valid']} valid cases")

    manifest = {
        "schema_version": 1,
        "purpose": "Pre-refactor baseline for the predictor-only migration.",
        "runtime": runtime_fingerprint(),
        "artifacts": {k: fingerprint_dir(v) for k, v in SERVING_ARTIFACTS.items()},
        "panel_sha256": hashlib.sha256(PANEL.read_bytes()).hexdigest(),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")

    print("capturing clintox-smilesgnn ...")
    clintox = capture_clintox(panel)
    print(f"  loaded={clintox['loaded']}"
          + ("" if clintox["loaded"] else f" — {clintox['error_type']}"))

    print("capturing herg-tox21-chemberta ...")
    chemberta = capture_chemberta(panel)
    print(f"  loaded={chemberta['loaded']} "
          f"p50={chemberta['latency_ms']['p50']}ms p95={chemberta['latency_ms']['p95']}ms")

    golden = {
        "schema_version": 1,
        "baseline_manifest": str(MANIFEST.relative_to(ROOT)),
        "runtime": manifest["runtime"],
        "models": {"clintox-smilesgnn-v1": clintox, "herg-tox21-chemberta-v1": chemberta},
    }
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote {GOLDEN.relative_to(ROOT)}")

    servable = [m for m, d in golden["models"].items() if d["loaded"]]
    blocked = [m for m, d in golden["models"].items() if not d["loaded"]]
    print(f"\nservable: {servable}")
    if blocked:
        print(f"BLOCKED (Phase 0 stop condition): {blocked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
