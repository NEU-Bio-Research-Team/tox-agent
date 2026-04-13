#!/usr/bin/env python3
"""Evaluate new 2-head ensemble models and rank all current dual-head models.

This script:
1. Loads existing dual-head checkpoints (ChemBERTa/MoLFormer/PubChem).
2. Builds new ensemble models under 2-head logic:
   - dualhead_ensemble3_simple
   - dualhead_ensemble3_weighted
   - dualhead_ensemble5_simple
   - dualhead_ensemble6_simple
3. Evaluates tox21 (multi-task) + hERG (binary) metrics.
4. Produces a ranking table across all dual-head models currently available.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from backend.data import get_task_names, load_tox21
from backend.inference import load_pretrained_dual_head_bundle
from backend.utils import ensure_dir, set_seed
from src.datasets import get_task_config

import ensemble5_tox21 as ens5
import ensemble6_tox21 as ens6

try:
    from tdc.single_pred import Tox
except ImportError:
    Tox = None


DUAL_MODEL_DIRS: Dict[str, str] = {
    "pretrained_2head_herg_chemberta_model": "models/pretrained_2head_herg_chemberta_model",
    "pretrained_2head_herg_molformer_model": "models/pretrained_2head_herg_molformer_model",
    "pretrained_2head_herg_pubchem_model": "models/pretrained_2head_herg_pubchem_model",
    "pretrained_2head_herg_chemberta_quick": "models/pretrained_2head_herg_chemberta_quick",
    "pretrained_2head_herg_molformer_quick": "models/pretrained_2head_herg_molformer_quick",
    "pretrained_2head_herg_pubchem_quick": "models/pretrained_2head_herg_pubchem_quick",
}


def _to_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_herg_karim(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if Tox is None:
        raise ImportError("pytdc is required for hERG_Karim. Install with: pip install pytdc")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    data = Tox(name="hERG_Karim", path=str(cache_path))
    split = data.get_split(method=split_type, seed=seed)

    train_df = split["train"].rename(columns={"Drug": "smiles", "Y": "label"})
    val_df = split["valid"].rename(columns={"Drug": "smiles", "Y": "label"})
    test_df = split["test"].rename(columns={"Drug": "smiles", "Y": "label"})

    return (
        train_df[["smiles", "label"]].copy(),
        val_df[["smiles", "label"]].copy(),
        test_df[["smiles", "label"]].copy(),
    )


def _resolve_task_thresholds(bundle: Dict[str, object], task_names: Sequence[str]) -> np.ndarray:
    task_thresholds = bundle.get("tox21_thresholds") if isinstance(bundle.get("tox21_thresholds"), dict) else {}
    out = []
    for name in task_names:
        raw = task_thresholds.get(name, 0.5) if isinstance(task_thresholds, dict) else 0.5
        out.append(float(np.clip(_to_float(raw, 0.5), 0.01, 0.99)))
    return np.asarray(out, dtype=np.float32)


def _canonicalize_smiles(smiles: str) -> str:
    s = str(smiles).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    return Chem.MolToSmiles(mol)


def predict_dual_bundle_probs(
    bundle: Dict[str, object],
    smiles_list: Sequence[str],
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    max_length = int(bundle.get("max_length", 128))

    canonical_smiles = [_canonicalize_smiles(s) for s in smiles_list]
    all_herg_logits: List[np.ndarray] = []
    all_tox_logits: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(canonical_smiles), int(batch_size)):
            chunk = canonical_smiles[start : start + int(batch_size)]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            heads = model.forward_heads(input_ids=input_ids, attention_mask=attention_mask)
            all_herg_logits.append(heads["herg_logits"].detach().cpu().numpy().reshape(-1))
            all_tox_logits.append(heads["tox21_logits"].detach().cpu().numpy())

    herg_probs = _sigmoid(np.concatenate(all_herg_logits, axis=0))
    tox_probs = _sigmoid(np.concatenate(all_tox_logits, axis=0))
    return herg_probs.astype(np.float32), tox_probs.astype(np.float32)


def compute_tox21_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    task_names: Sequence[str],
) -> Dict[str, object]:
    per_auc: Dict[str, float] = {}
    per_pr: Dict[str, float] = {}

    for idx, task_name in enumerate(task_names):
        valid = np.isfinite(labels[:, idx]) & np.isfinite(probs[:, idx])
        if int(valid.sum()) < 2:
            continue

        y_true = labels[valid, idx]
        if len(np.unique(y_true)) < 2:
            continue

        y_prob = probs[valid, idx]
        per_auc[str(task_name)] = float(roc_auc_score(y_true, y_prob))
        per_pr[str(task_name)] = float(average_precision_score(y_true, y_prob))

    macro_auc = float(np.mean(list(per_auc.values()))) if per_auc else float("nan")
    macro_pr = float(np.mean(list(per_pr.values()))) if per_pr else float("nan")

    return {
        "macro_auc_roc": macro_auc,
        "macro_pr_auc": macro_pr,
        "num_valid_tasks": int(len(per_auc)),
        "per_task_auc_roc": per_auc,
        "per_task_pr_auc": per_pr,
    }


def compute_binary_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_prob = np.asarray(probs, dtype=np.float32).reshape(-1)
    y_true = np.asarray(labels, dtype=np.float32).reshape(-1)

    valid = np.isfinite(y_prob) & np.isfinite(y_true)
    y_prob = y_prob[valid]
    y_true = y_true[valid]

    if y_true.size == 0:
        return {
            "auc_roc": float("nan"),
            "pr_auc": float("nan"),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "threshold": float(threshold),
            "n_samples": 0,
        }

    if len(np.unique(y_true)) >= 2:
        auc_roc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        auc_roc = float("nan")
        pr_auc = float("nan")

    y_pred = (y_prob >= float(threshold)).astype(np.int32)
    y_true_i = y_true.astype(np.int32)

    return {
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(y_true_i, y_pred)),
        "f1": float(f1_score(y_true_i, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n_samples": int(y_true_i.size),
    }


def _joint_scores(tox_auc: float, herg_auc: float) -> Dict[str, float]:
    if not np.isfinite(tox_auc) or not np.isfinite(herg_auc):
        return {
            "joint_auc_unweighted": float("nan"),
            "joint_auc_beta3": float("nan"),
        }
    return {
        "joint_auc_unweighted": float(0.5 * tox_auc + 0.5 * herg_auc),
        "joint_auc_beta3": float((tox_auc + 3.0 * herg_auc) / 4.0),
    }


def _find_member(members: Sequence[Dict[str, object]], name: str) -> Dict[str, object]:
    for member in members:
        if str(member.get("name")) == name:
            return dict(member)
    raise KeyError(f"Member not found: {name}")


def optimize_taskwise_weights(
    val_probs_list: Sequence[np.ndarray],
    val_labels: np.ndarray,
    task_names: Sequence[str],
    n_samples: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = len(val_probs_list)
    t = len(task_names)
    weights = np.full((t, m), 1.0 / float(m), dtype=np.float32)

    candidates = rng.dirichlet(np.ones(m, dtype=np.float32), size=int(max(128, n_samples)))

    for task_idx in range(t):
        valid = np.isfinite(val_labels[:, task_idx])
        for probs in val_probs_list:
            valid &= np.isfinite(probs[:, task_idx])

        if int(valid.sum()) < 10:
            continue

        y_true = val_labels[valid, task_idx]
        if len(np.unique(y_true)) < 2:
            continue

        preds = np.stack([p[valid, task_idx] for p in val_probs_list], axis=1)

        best_auc = -np.inf
        best_w = weights[task_idx]
        for cand_w in candidates:
            ens = np.sum(preds * cand_w[None, :], axis=1)
            try:
                auc = float(roc_auc_score(y_true, ens))
            except Exception:
                continue
            if auc > best_auc:
                best_auc = auc
                best_w = cand_w.astype(np.float32)

        weights[task_idx] = best_w

    return weights


def optimize_binary_weights(
    val_probs_list: Sequence[np.ndarray],
    val_labels: np.ndarray,
    n_samples: int = 4000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = len(val_probs_list)

    valid = np.isfinite(val_labels)
    for probs in val_probs_list:
        valid &= np.isfinite(probs)

    y_true = val_labels[valid]
    if int(y_true.size) < 10 or len(np.unique(y_true)) < 2:
        return np.full((m,), 1.0 / float(m), dtype=np.float32)

    preds = np.stack([p[valid] for p in val_probs_list], axis=1)

    candidates = rng.dirichlet(np.ones(m, dtype=np.float32), size=int(max(256, n_samples)))
    best_auc = -np.inf
    best_w = np.full((m,), 1.0 / float(m), dtype=np.float32)

    for cand_w in candidates:
        ens = np.sum(preds * cand_w[None, :], axis=1)
        try:
            auc = float(roc_auc_score(y_true, ens))
        except Exception:
            continue
        if auc > best_auc:
            best_auc = auc
            best_w = cand_w.astype(np.float32)

    return best_w


def blend_taskwise(probs_list: Sequence[np.ndarray], taskwise_weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probs_list[0], dtype=np.float32)
    for idx, probs in enumerate(probs_list):
        out += probs * taskwise_weights[:, idx]
    return out


def blend_global(probs_list: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probs_list[0], dtype=np.float32)
    for idx, probs in enumerate(probs_list):
        out += probs * float(weights[idx])
    return out


def evaluate_model_bundle(
    name: str,
    tox_probs: np.ndarray,
    tox_labels: np.ndarray,
    task_names: Sequence[str],
    herg_probs: np.ndarray,
    herg_labels: np.ndarray,
    herg_threshold: float,
) -> Dict[str, object]:
    tox21_metrics = compute_tox21_metrics(tox_probs, tox_labels, task_names)
    herg_metrics = compute_binary_metrics(herg_probs, herg_labels, threshold=herg_threshold)

    joint = _joint_scores(
        tox_auc=_to_float(tox21_metrics.get("macro_auc_roc"), float("nan")),
        herg_auc=_to_float(herg_metrics.get("auc_roc"), float("nan")),
    )

    return {
        "model_name": str(name),
        "tox21": tox21_metrics,
        "herg": herg_metrics,
        "joint": joint,
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate new dual-head ensemble models")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--graph-batch-size", type=int, default=128)
    parser.add_argument("--cache-dir", type=str, default="data")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; fallback to CPU")
        device = "cpu"

    set_seed(int(args.seed))

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    print("Loading tox21 splits...")
    _, tox21_val_df, tox21_test_df = load_tox21(
        cache_dir=str(project_root / args.cache_dir),
        split_type="scaffold",
        seed=int(args.seed),
        enforce_workspace_mode=False,
    )
    task_names = get_task_names("tox21")
    task_config = get_task_config("tox21")

    tox21_val_smiles = tox21_val_df["smiles"].astype(str).tolist()
    tox21_test_smiles = tox21_test_df["smiles"].astype(str).tolist()
    tox21_val_labels = tox21_val_df[task_names].astype(np.float32).values
    tox21_test_labels = tox21_test_df[task_names].astype(np.float32).values

    print("Loading hERG_Karim splits...")
    _, herg_val_df, herg_test_df = load_herg_karim(
        cache_dir=str(project_root / args.cache_dir),
        split_type="scaffold",
        seed=int(args.seed),
    )
    herg_val_smiles = herg_val_df["smiles"].astype(str).tolist()
    herg_test_smiles = herg_test_df["smiles"].astype(str).tolist()
    herg_val_labels = pd.to_numeric(herg_val_df["label"], errors="coerce").fillna(0).astype(np.float32).values
    herg_test_labels = pd.to_numeric(herg_test_df["label"], errors="coerce").fillna(0).astype(np.float32).values

    # ---------------------------------------------------------------------
    # Load dual-head base bundles and predict probs for val/test
    # ---------------------------------------------------------------------
    base_dual_keys = [
        "pretrained_2head_herg_chemberta_model",
        "pretrained_2head_herg_molformer_model",
        "pretrained_2head_herg_pubchem_model",
    ]

    dual_bundles: Dict[str, Dict[str, object]] = {}
    dual_tox_probs_val: Dict[str, np.ndarray] = {}
    dual_tox_probs_test: Dict[str, np.ndarray] = {}
    dual_herg_probs_val: Dict[str, np.ndarray] = {}
    dual_herg_probs_test: Dict[str, np.ndarray] = {}

    for key in base_dual_keys:
        model_dir = project_root / DUAL_MODEL_DIRS[key]
        print(f"Loading dual-head bundle: {key}")
        bundle = load_pretrained_dual_head_bundle(model_dir=model_dir, device=device)
        dual_bundles[key] = bundle

        print(f"  Inference tox21 val/test: {key}")
        _, tox_val = predict_dual_bundle_probs(bundle, tox21_val_smiles, device=device, batch_size=int(args.batch_size))
        _, tox_test = predict_dual_bundle_probs(bundle, tox21_test_smiles, device=device, batch_size=int(args.batch_size))
        dual_tox_probs_val[key] = tox_val
        dual_tox_probs_test[key] = tox_test

        print(f"  Inference hERG val/test: {key}")
        herg_val, _ = predict_dual_bundle_probs(bundle, herg_val_smiles, device=device, batch_size=int(args.batch_size))
        herg_test, _ = predict_dual_bundle_probs(bundle, herg_test_smiles, device=device, batch_size=int(args.batch_size))
        dual_herg_probs_val[key] = herg_val
        dual_herg_probs_test[key] = herg_test

    # ---------------------------------------------------------------------
    # Tox21-only members used in new 2-head ensemble logic
    # ---------------------------------------------------------------------
    member_afp = _find_member(ens5.ENSEMBLE_MEMBERS, "AttentiveFP")
    member_xgb = _find_member(ens5.ENSEMBLE_MEMBERS, "XGBoost")
    member_gps = _find_member(ens5.ENSEMBLE_MEMBERS, "GPS")
    member_pregin = _find_member(ens6.ENSEMBLE_MEMBERS, "Pretrained-GIN")

    print("Inference tox21-only member: Pretrained-GIN (val/test)")
    pregin_val = ens6.infer_pretrained_gin(
        member_pregin,
        tox21_val_smiles,
        tox21_val_labels,
        task_config,
        device,
        int(args.graph_batch_size),
    )
    pregin_test = ens6.infer_pretrained_gin(
        member_pregin,
        tox21_test_smiles,
        tox21_test_labels,
        task_config,
        device,
        int(args.graph_batch_size),
    )

    print("Inference tox21-only members: AttentiveFP/GPS/XGBoost (val/test)")
    afp_val = ens5.infer_attentivefp(
        member_afp,
        tox21_val_smiles,
        tox21_val_labels,
        task_config,
        device,
        int(args.graph_batch_size),
    )
    afp_test = ens5.infer_attentivefp(
        member_afp,
        tox21_test_smiles,
        tox21_test_labels,
        task_config,
        device,
        int(args.graph_batch_size),
    )

    gps_val = ens5.infer_gps(
        member_gps,
        tox21_val_smiles,
        tox21_val_labels,
        task_config,
        device,
        int(args.graph_batch_size),
    )
    gps_test = ens5.infer_gps(
        member_gps,
        tox21_test_smiles,
        tox21_test_labels,
        task_config,
        device,
        int(args.graph_batch_size),
    )

    xgb_val = ens5.infer_xgb(member_xgb, tox21_val_smiles, tox21_val_labels, task_config)
    xgb_test = ens5.infer_xgb(member_xgb, tox21_test_smiles, tox21_test_labels, task_config)

    # ---------------------------------------------------------------------
    # Build and evaluate NEW dual-head models
    # ---------------------------------------------------------------------
    out_root = project_root / "models"
    new_results: Dict[str, Dict[str, object]] = {}

    # 1) dualhead_ensemble3_simple
    tox3_val = [
        dual_tox_probs_val["pretrained_2head_herg_chemberta_model"],
        dual_tox_probs_val["pretrained_2head_herg_molformer_model"],
        pregin_val,
    ]
    tox3_test = [
        dual_tox_probs_test["pretrained_2head_herg_chemberta_model"],
        dual_tox_probs_test["pretrained_2head_herg_molformer_model"],
        pregin_test,
    ]
    herg2_val = [
        dual_herg_probs_val["pretrained_2head_herg_chemberta_model"],
        dual_herg_probs_val["pretrained_2head_herg_molformer_model"],
    ]
    herg2_test = [
        dual_herg_probs_test["pretrained_2head_herg_chemberta_model"],
        dual_herg_probs_test["pretrained_2head_herg_molformer_model"],
    ]

    res_e3 = evaluate_model_bundle(
        name="dualhead_ensemble3_simple",
        tox_probs=np.mean(np.stack(tox3_test, axis=0), axis=0),
        tox_labels=tox21_test_labels,
        task_names=task_names,
        herg_probs=np.mean(np.stack(herg2_test, axis=0), axis=0),
        herg_labels=herg_test_labels,
        herg_threshold=0.5,
    )
    new_results["dualhead_ensemble3_simple"] = res_e3
    e3_dir = out_root / "dualhead_ensemble3"
    ensure_dir(str(e3_dir))
    save_json(e3_dir / "dualhead_metrics.json", res_e3)

    # 2) dualhead_ensemble3_weighted
    tox3_task_weights = optimize_taskwise_weights(
        val_probs_list=tox3_val,
        val_labels=tox21_val_labels,
        task_names=task_names,
        n_samples=2500,
        seed=int(args.seed),
    )
    herg2_weights = optimize_binary_weights(
        val_probs_list=herg2_val,
        val_labels=herg_val_labels,
        n_samples=3000,
        seed=int(args.seed),
    )

    tox3_weighted_test = blend_taskwise(tox3_test, tox3_task_weights)
    herg2_weighted_test = blend_global(herg2_test, herg2_weights)

    res_e3w = evaluate_model_bundle(
        name="dualhead_ensemble3_weighted",
        tox_probs=tox3_weighted_test,
        tox_labels=tox21_test_labels,
        task_names=task_names,
        herg_probs=herg2_weighted_test,
        herg_labels=herg_test_labels,
        herg_threshold=0.5,
    )
    res_e3w["weights"] = {
        "tox21_taskwise_weights": tox3_task_weights.tolist(),
        "herg_weights": herg2_weights.tolist(),
        "tox21_members": [
            "pretrained_2head_herg_chemberta_model",
            "pretrained_2head_herg_molformer_model",
            "tox21_pretrained_gin_model",
        ],
        "herg_members": [
            "pretrained_2head_herg_chemberta_model",
            "pretrained_2head_herg_molformer_model",
        ],
    }
    new_results["dualhead_ensemble3_weighted"] = res_e3w
    e3w_dir = out_root / "dualhead_weighted_ensemble3"
    ensure_dir(str(e3w_dir))
    save_json(e3w_dir / "dualhead_metrics.json", res_e3w)

    # 3) dualhead_ensemble5_simple
    tox5_test = [
        dual_tox_probs_test["pretrained_2head_herg_chemberta_model"],
        dual_tox_probs_test["pretrained_2head_herg_molformer_model"],
        afp_test,
        xgb_test,
        gps_test,
    ]
    res_e5 = evaluate_model_bundle(
        name="dualhead_ensemble5_simple",
        tox_probs=np.mean(np.stack(tox5_test, axis=0), axis=0),
        tox_labels=tox21_test_labels,
        task_names=task_names,
        herg_probs=np.mean(np.stack(herg2_test, axis=0), axis=0),
        herg_labels=herg_test_labels,
        herg_threshold=0.5,
    )
    new_results["dualhead_ensemble5_simple"] = res_e5
    e5_dir = out_root / "dualhead_ensemble5"
    ensure_dir(str(e5_dir))
    save_json(e5_dir / "dualhead_metrics.json", res_e5)

    # 4) dualhead_ensemble6_simple
    tox6_test = [
        dual_tox_probs_test["pretrained_2head_herg_chemberta_model"],
        dual_tox_probs_test["pretrained_2head_herg_molformer_model"],
        afp_test,
        xgb_test,
        gps_test,
        pregin_test,
    ]
    res_e6 = evaluate_model_bundle(
        name="dualhead_ensemble6_simple",
        tox_probs=np.mean(np.stack(tox6_test, axis=0), axis=0),
        tox_labels=tox21_test_labels,
        task_names=task_names,
        herg_probs=np.mean(np.stack(herg2_test, axis=0), axis=0),
        herg_labels=herg_test_labels,
        herg_threshold=0.5,
    )
    new_results["dualhead_ensemble6_simple"] = res_e6
    e6_dir = out_root / "dualhead_ensemble6"
    ensure_dir(str(e6_dir))
    save_json(e6_dir / "dualhead_metrics.json", res_e6)

    # ---------------------------------------------------------------------
    # Rank ALL current dual-head models (existing + new)
    # ---------------------------------------------------------------------
    ranking_rows: List[Dict[str, object]] = []

    for model_key, rel_dir in DUAL_MODEL_DIRS.items():
        metric_path = project_root / rel_dir / "pretrained_2head_herg_tox21_metrics.json"
        if not metric_path.exists():
            continue
        with open(metric_path, "r") as f:
            payload = json.load(f)

        tox_auc = _to_float(((payload.get("test") or {}).get("tox21") or {}).get("macro_auc_roc"), float("nan"))
        tox_pr = _to_float(((payload.get("test") or {}).get("tox21") or {}).get("macro_pr_auc"), float("nan"))
        herg_auc = _to_float(((payload.get("test") or {}).get("herg") or {}).get("auc_roc"), float("nan"))
        herg_pr = _to_float(((payload.get("test") or {}).get("herg") or {}).get("pr_auc"), float("nan"))

        joint = _joint_scores(tox_auc=tox_auc, herg_auc=herg_auc)
        ranking_rows.append(
            {
                "model": model_key,
                "type": "dual_head_checkpoint",
                "tox21_macro_auc_roc": tox_auc,
                "tox21_macro_pr_auc": tox_pr,
                "herg_auc_roc": herg_auc,
                "herg_pr_auc": herg_pr,
                "joint_auc_unweighted": joint["joint_auc_unweighted"],
                "joint_auc_beta3": joint["joint_auc_beta3"],
                "source": str(metric_path.relative_to(project_root)),
            }
        )

    for model_name, metrics_payload in new_results.items():
        tox_auc = _to_float(((metrics_payload.get("tox21") or {}).get("macro_auc_roc")), float("nan"))
        tox_pr = _to_float(((metrics_payload.get("tox21") or {}).get("macro_pr_auc")), float("nan"))
        herg_auc = _to_float(((metrics_payload.get("herg") or {}).get("auc_roc")), float("nan"))
        herg_pr = _to_float(((metrics_payload.get("herg") or {}).get("pr_auc")), float("nan"))
        joint = _joint_scores(tox_auc=tox_auc, herg_auc=herg_auc)

        if model_name == "dualhead_ensemble3_simple":
            src = "models/dualhead_ensemble3/dualhead_metrics.json"
        elif model_name == "dualhead_ensemble3_weighted":
            src = "models/dualhead_weighted_ensemble3/dualhead_metrics.json"
        elif model_name == "dualhead_ensemble5_simple":
            src = "models/dualhead_ensemble5/dualhead_metrics.json"
        else:
            src = "models/dualhead_ensemble6/dualhead_metrics.json"

        ranking_rows.append(
            {
                "model": model_name,
                "type": "dual_head_ensemble_new",
                "tox21_macro_auc_roc": tox_auc,
                "tox21_macro_pr_auc": tox_pr,
                "herg_auc_roc": herg_auc,
                "herg_pr_auc": herg_pr,
                "joint_auc_unweighted": joint["joint_auc_unweighted"],
                "joint_auc_beta3": joint["joint_auc_beta3"],
                "source": src,
            }
        )

    ranking_rows.sort(
        key=lambda row: (
            -1e9
            if not np.isfinite(_to_float(row.get("joint_auc_beta3"), float("nan")))
            else _to_float(row.get("joint_auc_beta3"), float("nan"))
        ),
        reverse=True,
    )

    # assign ranks after sorting desc
    ranked: List[Dict[str, object]] = []
    rank_counter = 1
    for row in ranking_rows:
        row_out = dict(row)
        row_out["rank"] = int(rank_counter)
        ranked.append(row_out)
        rank_counter += 1

    ranking_csv = out_root / "dualhead_model_ranking.csv"
    fieldnames = [
        "rank",
        "model",
        "type",
        "tox21_macro_auc_roc",
        "tox21_macro_pr_auc",
        "herg_auc_roc",
        "herg_pr_auc",
        "joint_auc_unweighted",
        "joint_auc_beta3",
        "source",
    ]
    with open(ranking_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)

    ranking_json = out_root / "dualhead_model_ranking.json"
    save_json(ranking_json, {"rows": ranked})

    print("\n=== NEW DUAL-HEAD MODEL METRICS ===")
    for model_name, payload in new_results.items():
        tox_auc = _to_float(((payload.get("tox21") or {}).get("macro_auc_roc")), float("nan"))
        herg_auc = _to_float(((payload.get("herg") or {}).get("auc_roc")), float("nan"))
        joint = _to_float(((payload.get("joint") or {}).get("joint_auc_beta3")), float("nan"))
        print(f"{model_name:30s} tox21_auc={tox_auc:.4f} herg_auc={herg_auc:.4f} joint_beta3={joint:.4f}")

    print("\n=== DUAL-HEAD RANKING (joint_auc_beta3) ===")
    for row in ranked:
        print(
            f"#{int(row['rank']):02d} {str(row['model']):35s} "
            f"joint={_to_float(row['joint_auc_beta3'], float('nan')):.4f} "
            f"tox21={_to_float(row['tox21_macro_auc_roc'], float('nan')):.4f} "
            f"herg={_to_float(row['herg_auc_roc'], float('nan')):.4f}"
        )

    print(f"\nSaved ranking CSV: {ranking_csv}")
    print(f"Saved ranking JSON: {ranking_json}")


if __name__ == "__main__":
    main()
