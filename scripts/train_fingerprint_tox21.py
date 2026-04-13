#!/usr/bin/env python3
"""
Train ECFP4 + XGBoost per-task classifiers for Tox21.

Usage:
    conda activate drug-tox-env
    python scripts/train_fingerprint_tox21.py --device cpu
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import roc_auc_score, average_precision_score

from backend.data import get_task_names, load_tox21
from backend.utils import ensure_dir, save_metrics, set_seed


def smiles_to_ecfp(smiles_list, radius=2, nbits=2048):
    """Convert SMILES to ECFP fingerprints, returning matrix and valid indices."""
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        valid_idx.append(i)
    return np.array(fps), valid_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=2048)
    args = parser.parse_args()

    set_seed(42)

    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
        import xgboost as xgb

    task_names = get_task_names("tox21")

    print("=" * 70)
    print(f"ECFP{args.radius*2} + XGBoost — Tox21")
    print("=" * 70)

    train_df, val_df, test_df = load_tox21(
        cache_dir=str(project_root / "data"),
        split_type="scaffold", seed=42,
        enforce_workspace_mode=False,
    )
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    print("Computing ECFP fingerprints...")
    X_train, train_ok = smiles_to_ecfp(list(train_df["smiles"]), args.radius, args.nbits)
    X_val, val_ok = smiles_to_ecfp(list(val_df["smiles"]), args.radius, args.nbits)
    X_test, test_ok = smiles_to_ecfp(list(test_df["smiles"]), args.radius, args.nbits)

    y_train = train_df[task_names].values[train_ok]
    y_val = val_df[task_names].values[val_ok]
    y_test = test_df[task_names].values[test_ok]
    print(f"FP matrices: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    out_dir = project_root / "models" / "tox21_fingerprint_model"
    ensure_dir(str(out_dir / "models"))

    print(f"\n{'Task':<20} {'Pos':>5} {'Neg':>5} | {'Val AUC':>8} {'Test AUC':>9}")
    print("-" * 60)

    task_metrics = {}
    for t, task in enumerate(task_names):
        tr_mask = ~np.isnan(y_train[:, t])
        v_mask = ~np.isnan(y_val[:, t])
        te_mask = ~np.isnan(y_test[:, t])

        X_tr_t, y_tr_t = X_train[tr_mask], y_train[tr_mask, t]
        X_v_t, y_v_t = X_val[v_mask], y_val[v_mask, t]
        X_te_t, y_te_t = X_test[te_mask], y_test[te_mask, t]

        if len(np.unique(y_tr_t)) < 2 or len(X_tr_t) < 20:
            print(f"  {task:<20} — skipped")
            continue

        n_pos = int(y_tr_t.sum())
        n_neg = len(y_tr_t) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=spw, eval_metric="auc",
            early_stopping_rounds=30, tree_method="hist",
            device="cuda" if args.device == "cuda" else "cpu",
            verbosity=0, random_state=42,
        )
        clf.fit(X_tr_t, y_tr_t, eval_set=[(X_v_t, y_v_t)], verbose=False)

        val_auc = roc_auc_score(y_v_t, clf.predict_proba(X_v_t)[:, 1]) if len(np.unique(y_v_t)) > 1 else 0.0
        test_auc = roc_auc_score(y_te_t, clf.predict_proba(X_te_t)[:, 1]) if len(np.unique(y_te_t)) > 1 else 0.0

        print(f"  {task:<20} {n_pos:>5} {n_neg:>5} | {val_auc:>8.4f} {test_auc:>9.4f}")

        with open(out_dir / "models" / f"{task}.pkl", "wb") as f:
            pickle.dump(clf, f)

        task_metrics[task] = {"val_auc": val_auc, "test_auc": test_auc}

    # Summary
    valid_aucs = [v["test_auc"] for v in task_metrics.values()]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    print("-" * 60)
    print(f"  {'MEAN':<20}       | {'':>8} {mean_auc:>9.4f}")

    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "test_labels.npy", y_test)

    flat = {"test_mean_auc_roc": mean_auc}
    for task, m in task_metrics.items():
        flat[f"test_auc_{task}"] = m["test_auc"]
    save_metrics(flat, str(out_dir / "metrics.txt"))

    print(f"\nSaved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
