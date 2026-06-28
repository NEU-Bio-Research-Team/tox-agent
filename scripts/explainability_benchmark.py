#!/usr/bin/env python
"""
Explainability Benchmark Pipeline — Task 1
============================================
Produces 30 toxic compounds from 3 datasets, each with:
  - SMILES, ground-truth label, prediction, atom scores
  - 3 explanation visualizations (atom heatmap, bond heatmap, bar chart)

Datasets:
  1. Tox21       — 10 toxic + 10 non-toxic  (SR-MMP task, GATv2 model)
  2. ClinTox     — 10 toxic + 10 non-toxic  (CT_TOX, RF on Morgan FP)
  3. hERG_Karim  — 10 toxic + 10 non-toxic  (hERG, RF on Morgan FP)

Output directory: benchmark/
"""

import sys
import os
import json
import csv
import io
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rdkit import Chem
from rdkit.Chem import AllChem
from PIL import Image

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
SEED = 42
DEVICE = "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════════════

def _importance_to_rgb(importance: float, cmap_name: str = "RdYlGn_r"):
    """Map normalised importance [0,1] → RGB tuple."""
    val = float(np.clip(importance, 0.0, 1.0))
    r, g, b, _ = plt.get_cmap(cmap_name)(val)
    return (r, g, b)


def _mol_to_img(mol, highlight_atoms, highlight_bonds,
                atom_colors, bond_colors, width=600, height=500):
    """Render molecule to PIL Image via rdMolDraw2DCairo."""
    from rdkit.Chem.Draw import rdMolDraw2D
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def _subtitle(smiles, pred_prob, pred_class, label, viz_name):
    smi_short = smiles[:60] + ("…" if len(smiles) > 60 else "")
    true_str = f"True: {'Toxic' if label == 1 else 'Non-toxic'}"
    pred_str = f"Pred: {'Toxic' if pred_class == 1 else 'Non-toxic'} (P={pred_prob:.3f})"
    return f"{viz_name}\n{smi_short}\n{pred_str}  |  {true_str}"


def draw_atom_heatmap(mol, atom_imp, smiles, pred_prob, pred_class,
                      label, save_path):
    """Viz 1: molecule with atoms coloured by importance."""
    atom_colors = {i: _importance_to_rgb(imp) for i, imp in enumerate(atom_imp)}
    img = _mol_to_img(
        mol,
        highlight_atoms=list(range(mol.GetNumAtoms())),
        highlight_bonds=[],
        atom_colors=atom_colors,
        bond_colors={},
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(_subtitle(smiles, pred_prob, pred_class, label,
                           "Atom Importance Heatmap"), fontsize=10)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                               norm=mcolors.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04).set_label("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_bond_heatmap(mol, atom_imp, bond_imp, smiles, pred_prob,
                      pred_class, label, save_path):
    """Viz 2: molecule with bonds coloured by importance."""
    bond_colors = {}
    for k in range(min(mol.GetNumBonds(), len(bond_imp))):
        bond_colors[k] = _importance_to_rgb(bond_imp[k])

    atom_colors = {}
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        adj = [bond_imp[b.GetIdx()]
               for b in atom.GetBonds() if b.GetIdx() < len(bond_imp)]
        atom_colors[i] = _importance_to_rgb(max(adj) if adj else 0.0)

    img = _mol_to_img(
        mol,
        highlight_atoms=list(range(mol.GetNumAtoms())),
        highlight_bonds=list(range(min(mol.GetNumBonds(), len(bond_imp)))),
        atom_colors=atom_colors,
        bond_colors=bond_colors,
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(_subtitle(smiles, pred_prob, pred_class, label,
                           "Bond Importance Heatmap"), fontsize=10)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                               norm=mcolors.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04).set_label("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_atom_bar_chart(mol, atom_imp, smiles, pred_prob, pred_class,
                        label, save_path):
    """Viz 3: bar chart of per-atom importance scores."""
    n = mol.GetNumAtoms()
    syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n)]
    x_labels = [f"{s}({i})" for i, s in enumerate(syms)]
    colors = [_importance_to_rgb(imp) for imp in atom_imp]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 5))
    bars = ax.bar(range(n), atom_imp, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Importance Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(_subtitle(smiles, pred_prob, pred_class, label,
                           "Atom Score Bar Chart"), fontsize=10)
    for bar, score in zip(bars, atom_imp):
        if score > 0.15:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02, f"{score:.2f}",
                    ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Bond importance from atom importance  (used when no model bond mask)
# ═══════════════════════════════════════════════════════════════════════════

def _atom_imp_to_bond_imp(mol, atom_imp):
    """Derive per-bond importance from atom importance."""
    bond_imp = np.array([
        0.5 * (float(atom_imp[b.GetBeginAtomIdx()])
               + float(atom_imp[b.GetEndAtomIdx()]))
        for b in mol.GetBonds()
    ], dtype=np.float32)
    if bond_imp.size > 0 and bond_imp.max() > 0:
        bond_imp = bond_imp / bond_imp.max()
    return bond_imp


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Load datasets
# ═══════════════════════════════════════════════════════════════════════════

def load_all_datasets():
    print("=" * 60)
    print("Step 1: Loading Datasets")
    print("=" * 60)

    from backend.data import load_tox21, load_clintox

    # ── Tox21 ──
    print("[1/3] Loading Tox21 …")
    t21_tr, t21_va, t21_te = load_tox21(
        cache_dir=str(PROJECT_ROOT / "data"), split_type="scaffold",
        seed=SEED, enforce_workspace_mode=False,
    )
    print(f"      Train {len(t21_tr)} | Val {len(t21_va)} | Test {len(t21_te)}")

    # ── ClinTox ──
    print("[2/3] Loading ClinTox …")
    ct_tr, ct_va, ct_te = load_clintox(
        cache_dir=str(PROJECT_ROOT / "data"), split_type="scaffold",
        seed=SEED, enforce_workspace_mode=False,
    )
    print(f"      Train {len(ct_tr)} | Val {len(ct_va)} | Test {len(ct_te)}")

    # ── hERG_Karim (PyTDC) ──
    print("[3/3] Loading hERG_Karim …")
    from tdc.single_pred import Tox
    herg = Tox(name="herg", path=str(PROJECT_ROOT / "data"))
    herg_split = herg.get_split(method="scaffold", seed=SEED)
    for key in herg_split:
        df = herg_split[key]
        rename = {}
        if "Drug" in df.columns:
            rename["Drug"] = "smiles"
        if "Y" in df.columns:
            rename["Y"] = "label"
        if rename:
            herg_split[key] = df.rename(columns=rename)
    print(f"      Train {len(herg_split['train'])} | "
          f"Val {len(herg_split['valid'])} | Test {len(herg_split['test'])}")

    return {
        "tox21":   {"train": t21_tr, "val": t21_va, "test": t21_te},
        "clintox": {"train": ct_tr,  "val": ct_va,  "test": ct_te},
        "herg":    {"train": herg_split["train"],
                    "val":   herg_split["valid"],
                    "test":  herg_split["test"]},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Sample compounds
# ═══════════════════════════════════════════════════════════════════════════

def _sample_group(df, label_col, is_toxic, n, rng):
    """Pick n valid SMILES with the desired label."""
    target = 1.0 if is_toxic else 0.0
    sub = df[df[label_col] == target].dropna(subset=["smiles", label_col])
    sub = sub[sub["smiles"].apply(
        lambda s: Chem.MolFromSmiles(str(s)) is not None
    )].reset_index(drop=True)
    idx = rng.choice(len(sub), min(n, len(sub)), replace=False)
    return sub.iloc[idx]


def sample_compounds(datasets):
    print("\n" + "=" * 60)
    print("Step 2: Sampling Compounds (seed=%d)" % SEED)
    print("=" * 60)

    rng = np.random.RandomState(SEED)
    samples: List[Dict[str, Any]] = []
    cid = 1

    def _add(rows, dataset, label, label_task):
        nonlocal cid
        for _, row in rows.iterrows():
            samples.append({
                "compound_id": f"compound_{cid:03d}",
                "dataset": dataset,
                "smiles": str(row["smiles"]),
                "label": int(label),
                "label_task": label_task,
                "seed": SEED,
            })
            cid += 1

    # Tox21 — SR-MMP
    t21 = pd.concat([datasets["tox21"]["train"],
                     datasets["tox21"]["val"],
                     datasets["tox21"]["test"]], ignore_index=True)
    tox = _sample_group(t21, "SR-MMP", True, 10, rng)
    safe = _sample_group(t21, "SR-MMP", False, 10, rng)
    _add(tox,  "Tox21", 1, "SR-MMP")
    _add(safe, "Tox21", 0, "SR-MMP")
    print(f"  Tox21:      {len(tox)} toxic + {len(safe)} non-toxic")

    # ClinTox — CT_TOX
    ct = pd.concat([datasets["clintox"]["train"],
                    datasets["clintox"]["val"],
                    datasets["clintox"]["test"]], ignore_index=True)
    tox = _sample_group(ct, "CT_TOX", True, 10, rng)
    safe = _sample_group(ct, "CT_TOX", False, 10, rng)
    _add(tox,  "ClinTox", 1, "CT_TOX")
    _add(safe, "ClinTox", 0, "CT_TOX")
    print(f"  ClinTox:    {len(tox)} toxic + {len(safe)} non-toxic")

    # hERG_Karim — label
    hg = pd.concat([datasets["herg"]["train"],
                    datasets["herg"]["val"],
                    datasets["herg"]["test"]], ignore_index=True)
    tox = _sample_group(hg, "label", True, 10, rng)
    safe = _sample_group(hg, "label", False, 10, rng)
    _add(tox,  "hERG_Karim", 1, "hERG")
    _add(safe, "hERG_Karim", 0, "hERG")
    print(f"  hERG_Karim: {len(tox)} toxic + {len(safe)} non-toxic")

    total = len(samples)
    n_tox = sum(1 for s in samples if s["label"] == 1)
    print(f"\n  Total: {total} compounds ({n_tox} toxic, {total - n_tox} non-toxic)")
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 & 4 — Prediction + atom importance extraction
# ═══════════════════════════════════════════════════════════════════════════

def _explain_tox21_gatv2(samples):
    """Use trained GATv2 model + gradient saliency for Tox21 compounds."""
    from backend.inference import load_tox21_gatv2_model
    from backend.gnn_explainer import explain_tox21_task_gradient

    model, task_names = load_tox21_gatv2_model(
        model_dir=PROJECT_ROOT / "models" / "tox21_gatv2_model",
        config_path=PROJECT_ROOT / "config" / "tox21_gatv2_config.yaml",
        device=DEVICE,
    )
    print(f"  Loaded tox21_gatv2_model ({len(task_names)} tasks)")

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['smiles'][:50]}…")
        try:
            res = explain_tox21_task_gradient(
                smiles=s["smiles"], model=model,
                task_names=task_names, target_task="SR-MMP",
                device=DEVICE, threshold=0.5,
            )
            s["prediction_prob"]  = res["prediction_prob"]
            s["predicted_class"]  = res["predicted_class"]
            s["atom_importance"]  = res["atom_importance"]
            s["bond_importance"]  = res["bond_importance"]
            s["task_scores"]      = res.get("task_scores", {})
            s["method"]           = "gradient_saliency"
            s["model"]            = "tox21_gatv2_model"
        except Exception as exc:
            print(f"        ⚠ gradient failed ({exc}), using FP fallback")
            _fp_fallback(s)


def _explain_with_rf(samples, dataset_splits, label_col, model_label):
    """Train RF on Morgan FP, predict, derive atom importance."""
    from sklearn.ensemble import RandomForestClassifier
    from backend.featurization import featurize_fingerprint
    from backend.viz import map_fingerprint_to_atoms

    train_df = dataset_splits["train"]
    smiles_tr = train_df["smiles"].astype(str).tolist()
    labels_tr = train_df[label_col].astype(float).values
    mask = ~np.isnan(labels_tr)
    smiles_tr = [s for s, m in zip(smiles_tr, mask) if m]
    labels_tr = labels_tr[mask]

    print(f"  Training RF on {len(smiles_tr)} compounds …")
    fps_tr = np.stack([featurize_fingerprint(s) for s in smiles_tr])
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                n_jobs=-1)
    rf.fit(fps_tr, labels_tr)
    fi = rf.feature_importances_          # shape (2048,)

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['smiles'][:50]}…")
        try:
            fp = featurize_fingerprint(s["smiles"])
            prob = float(rf.predict_proba(fp.reshape(1, -1))[0, 1])
            atom_imp = map_fingerprint_to_atoms(s["smiles"], fi)
            mol = Chem.MolFromSmiles(s["smiles"])
            bond_imp = (_atom_imp_to_bond_imp(mol, atom_imp)
                        if mol else np.array([], dtype=np.float32))

            s["prediction_prob"]  = prob
            s["predicted_class"]  = int(prob >= 0.5)
            s["atom_importance"]  = atom_imp
            s["bond_importance"]  = bond_imp
            s["method"]           = "rf_fingerprint"
            s["model"]            = model_label
        except Exception as exc:
            print(f"        ⚠ RF failed ({exc}), using FP fallback")
            _fp_fallback(s)


def _fp_fallback(s):
    """Last-resort fallback: raw FP bits → atom importance."""
    from backend.featurization import featurize_fingerprint
    from backend.viz import map_fingerprint_to_atoms

    fp = featurize_fingerprint(s["smiles"])
    atom_imp = map_fingerprint_to_atoms(s["smiles"], fp)
    mol = Chem.MolFromSmiles(s["smiles"])
    bond_imp = (_atom_imp_to_bond_imp(mol, atom_imp)
                if mol else np.array([], dtype=np.float32))

    s["prediction_prob"]  = float(s["label"])
    s["predicted_class"]  = int(s["label"])
    s["atom_importance"]  = atom_imp
    s["bond_importance"]  = bond_imp
    s["method"]           = "fingerprint_fallback"
    s["model"]            = "ECFP4_structure"


def predict_and_explain(samples, datasets):
    print("\n" + "=" * 60)
    print("Steps 3–4: Prediction & Atom Importance Extraction")
    print("=" * 60)

    tox21_s  = [s for s in samples if s["dataset"] == "Tox21"]
    ct_s     = [s for s in samples if s["dataset"] == "ClinTox"]
    herg_s   = [s for s in samples if s["dataset"] == "hERG_Karim"]

    # ── Tox21: GATv2 + gradient saliency ──
    print("\n─── Tox21 (GATv2 + gradient saliency) ───")
    try:
        _explain_tox21_gatv2(tox21_s)
    except Exception as exc:
        print(f"  ⚠ GATv2 unavailable ({exc}); using RF fallback for Tox21")
        _explain_with_rf(tox21_s, datasets["tox21"], "SR-MMP",
                         "RandomForest_Tox21_ECFP4")

    # ── ClinTox: RF + fingerprint ──
    print("\n─── ClinTox (RandomForest + Morgan FP) ───")
    _explain_with_rf(ct_s, datasets["clintox"], "CT_TOX",
                     "RandomForest_ClinTox_ECFP4")

    # ── hERG: RF + fingerprint ──
    print("\n─── hERG_Karim (RandomForest + Morgan FP) ───")
    _explain_with_rf(herg_s, datasets["herg"], "label",
                     "RandomForest_hERG_ECFP4")

    return samples


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Generate 3 visualizations for 30 toxic compounds
# ═══════════════════════════════════════════════════════════════════════════

def generate_visualizations(samples):
    print("\n" + "=" * 60)
    print("Step 5: Generating Visualizations (3 × 30 = 90 images)")
    print("=" * 60)

    toxic = [s for s in samples if s["label"] == 1]
    for i, s in enumerate(toxic):
        cdir = BENCHMARK_DIR / s["compound_id"]
        cdir.mkdir(parents=True, exist_ok=True)

        mol = Chem.MolFromSmiles(s["smiles"])
        if mol is None:
            print(f"  [{i+1}/{len(toxic)}] SKIP invalid SMILES")
            continue

        ai = np.asarray(s["atom_importance"], dtype=np.float32)
        bi = np.asarray(s["bond_importance"], dtype=np.float32)
        pp, pc, lb = s["prediction_prob"], s["predicted_class"], s["label"]

        print(f"  [{i+1}/{len(toxic)}] {s['compound_id']}  "
              f"({s['dataset']})  {s['smiles'][:40]}…")

        draw_atom_heatmap(mol, ai, s["smiles"], pp, pc, lb,
                          str(cdir / "atom_heatmap.png"))
        draw_bond_heatmap(mol, ai, bi, s["smiles"], pp, pc, lb,
                          str(cdir / "bond_heatmap.png"))
        draw_atom_bar_chart(mol, ai, s["smiles"], pp, pc, lb,
                            str(cdir / "atom_score_bar_chart.png"))


# ═══════════════════════════════════════════════════════════════════════════
# Steps 6–7 — Save benchmark outputs
# ═══════════════════════════════════════════════════════════════════════════

def save_outputs(samples):
    print("\n" + "=" * 60)
    print("Steps 6–7: Saving Benchmark Outputs")
    print("=" * 60)

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    toxic = [s for s in samples if s["label"] == 1]

    # ── metadata.csv (30 toxic only) ──
    rows = []
    for s in toxic:
        cid = s["compound_id"]
        rows.append({
            "compound_id":            cid,
            "dataset":                s["dataset"],
            "smiles":                 s["smiles"],
            "ground_truth":           s["label"],
            "prediction":             s.get("predicted_class", ""),
            "prediction_probability": f"{s.get('prediction_prob', 0):.4f}",
            "model":                  s.get("model", ""),
            "method":                 s.get("method", ""),
            "atom_score_file":        f"{cid}/atom_scores.csv",
            "atom_heatmap":           f"{cid}/atom_heatmap.png",
            "bond_heatmap":           f"{cid}/bond_heatmap.png",
            "bar_chart":              f"{cid}/atom_score_bar_chart.png",
        })
    pd.DataFrame(rows).to_csv(BENCHMARK_DIR / "metadata.csv", index=False)
    print(f"  metadata.csv  → {len(rows)} toxic compounds")

    # ── per-compound folders ──
    for s in toxic:
        cdir = BENCHMARK_DIR / s["compound_id"]
        cdir.mkdir(parents=True, exist_ok=True)

        # smiles.txt
        (cdir / "smiles.txt").write_text(s["smiles"], encoding="utf-8")

        # prediction.json
        pred = {
            "compound_id":            s["compound_id"],
            "dataset":                s["dataset"],
            "smiles":                 s["smiles"],
            "ground_truth":           int(s["label"]),
            "prediction":             int(s.get("predicted_class", s["label"])),
            "prediction_probability": float(s.get("prediction_prob", 0)),
            "model":                  s.get("model", ""),
            "method":                 s.get("method", ""),
            "label_task":             s.get("label_task", ""),
        }
        if "task_scores" in s:
            pred["task_scores"] = {k: round(float(v), 4)
                                   for k, v in s["task_scores"].items()}
        (cdir / "prediction.json").write_text(
            json.dumps(pred, indent=2), encoding="utf-8")

        # atom_scores.csv
        mol = Chem.MolFromSmiles(s["smiles"])
        if mol is not None:
            ai = np.asarray(s["atom_importance"])
            with open(cdir / "atom_scores.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["atom_index", "atom_symbol", "importance_score"])
                for idx in range(mol.GetNumAtoms()):
                    sym = mol.GetAtomWithIdx(idx).GetSymbol()
                    sc = float(ai[idx]) if idx < len(ai) else 0.0
                    w.writerow([idx, sym, f"{sc:.4f}"])

    print(f"  {len(toxic)} compound folders created")

    # ── all_60_compounds.csv ──
    all_rows = []
    for s in samples:
        mol = Chem.MolFromSmiles(s["smiles"])
        all_rows.append({
            "compound_id":            s["compound_id"],
            "dataset":                s["dataset"],
            "smiles":                 s["smiles"],
            "label":                  s["label"],
            "label_task":             s.get("label_task", ""),
            "prediction":             s.get("predicted_class", ""),
            "prediction_probability": f"{s.get('prediction_prob', 0):.4f}",
            "model":                  s.get("model", ""),
            "method":                 s.get("method", ""),
            "num_atoms":              mol.GetNumAtoms() if mol else 0,
            "seed":                   SEED,
        })
    pd.DataFrame(all_rows).to_csv(
        BENCHMARK_DIR / "all_60_compounds.csv", index=False)
    print(f"  all_60_compounds.csv  → {len(all_rows)} rows")


# ═══════════════════════════════════════════════════════════════════════════
# Step 8 — Generate benchmark report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(samples):
    print("\n" + "=" * 60)
    print("Step 8: Generating Benchmark Report")
    print("=" * 60)

    toxic   = [s for s in samples if s["label"] == 1]
    safe    = [s for s in samples if s["label"] == 0]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    L = []
    L.append("# Explainability Benchmark Report — Task 1\n")
    L.append(f"**Generated**: {now_str}  ")
    L.append(f"**Random Seed**: {SEED}\n")

    # ── Summary ──
    L.append("## Summary\n")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Total Compounds | {len(samples)} |")
    L.append(f"| Toxic | {len(toxic)} |")
    L.append(f"| Non-toxic | {len(safe)} |")
    L.append(f"| Datasets | 3 (Tox21, ClinTox, hERG_Karim) |")
    L.append(f"| Visualizations | {len(toxic) * 3} |")
    L.append("")

    # ── Per-dataset ──
    L.append("## Per-Dataset Statistics\n")
    for ds in ["Tox21", "ClinTox", "hERG_Karim"]:
        ds_all = [s for s in samples if s["dataset"] == ds]
        ds_t   = [s for s in ds_all if s["label"] == 1]
        ds_s   = [s for s in ds_all if s["label"] == 0]
        L.append(f"### {ds}\n")
        L.append(f"- Toxic: {len(ds_t)}  |  Non-toxic: {len(ds_s)}")
        if ds_t:
            L.append(f"- Model: `{ds_t[0].get('model', 'N/A')}`")
            L.append(f"- Method: `{ds_t[0].get('method', 'N/A')}`")
        L.append("")

    # ── Full 60-compound table ──
    L.append("## All 60 Compounds\n")
    L.append("| # | ID | Dataset | SMILES | Label | Pred | P(toxic) | Model |")
    L.append("|---|-----|---------|--------|-------|------|----------|-------|")
    for i, s in enumerate(samples):
        smi = s["smiles"][:40] + ("…" if len(s["smiles"]) > 40 else "")
        lab = "Toxic" if s["label"] == 1 else "Safe"
        prd = "Toxic" if s.get("predicted_class", 0) == 1 else "Safe"
        pp  = f"{s.get('prediction_prob', 0):.3f}"
        L.append(f"| {i+1} | {s['compound_id']} | {s['dataset']} | "
                 f"`{smi}` | {lab} | {prd} | {pp} | "
                 f"{s.get('model', '')} |")
    L.append("")

    # ── 30 toxic benchmark ──
    L.append("## Benchmark: 30 Toxic Compounds with Visualizations\n")
    L.append("| # | ID | Dataset | SMILES | P(toxic) "
             "| Atom Heatmap | Bond Heatmap | Bar Chart |")
    L.append("|---|-----|---------|--------|----------"
             "|--------------|--------------|-----------|")
    for i, s in enumerate(toxic):
        smi = s["smiles"][:30] + ("…" if len(s["smiles"]) > 30 else "")
        pp  = f"{s.get('prediction_prob', 0):.3f}"
        cid = s["compound_id"]
        L.append(
            f"| {i+1} | {cid} | {s['dataset']} | `{smi}` | {pp} "
            f"| ![atom]({cid}/atom_heatmap.png) "
            f"| ![bond]({cid}/bond_heatmap.png) "
            f"| ![bar]({cid}/atom_score_bar_chart.png) |"
        )
    L.append("")

    # ── Sample atom scores ──
    L.append("## Sample Atom Scores (First 5 Toxic Compounds)\n")
    for s in toxic[:5]:
        mol = Chem.MolFromSmiles(s["smiles"])
        if mol is None:
            continue
        ai = np.asarray(s["atom_importance"])
        L.append(f"### {s['compound_id']} ({s['dataset']})\n")
        L.append(f"SMILES: `{s['smiles']}`\n")
        L.append("| Atom Index | Atom | Score |")
        L.append("|------------|------|-------|")
        for idx in range(min(mol.GetNumAtoms(), len(ai))):
            sym = mol.GetAtomWithIdx(idx).GetSymbol()
            L.append(f"| {idx} | {sym} | {float(ai[idx]):.4f} |")
        L.append("")

    report_path = BENCHMARK_DIR / "explainability_benchmark_report.md"
    report_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  Report → {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   EXPLAINABILITY  BENCHMARK  PIPELINE  —  TASK  1      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Time  : {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Seed  : {SEED}")
    print(f"  Output: {BENCHMARK_DIR}\n")

    datasets = load_all_datasets()
    samples  = sample_compounds(datasets)
    predict_and_explain(samples, datasets)
    generate_visualizations(samples)
    save_outputs(samples)
    generate_report(samples)

    elapsed = (datetime.now() - t0).total_seconds()
    toxic_n = sum(1 for s in samples if s["label"] == 1)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                   PIPELINE  COMPLETE                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Output        : {BENCHMARK_DIR}")
    print(f"  metadata.csv  : {toxic_n} toxic compounds")
    print(f"  Compounds     : {toxic_n} directories × 6 files each")
    print(f"  Visualizations: {toxic_n * 3} images (30 × 3)")
    print(f"  Report        : explainability_benchmark_report.md")
    print(f"  Elapsed       : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
