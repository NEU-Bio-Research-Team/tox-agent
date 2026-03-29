"""
Inference utilities for SMILESGNN toxicity predictor.

Provides:
  - load_model()     — load trained checkpoint + tokenizer
  - predict_batch()  — fast Stage A batch scoring

HybridModelWrapper routes SMILES token IDs from the PyG batch
object to model.forward(). This must match the wrapper used during
training, otherwise the SMILES encoder is bypassed (zero vectors).
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.data import get_task_names
from src.graph_data import get_feature_dims, smiles_to_pyg_data
from src.graph_models import create_gatv2_model
from src.graph_models_hybrid import create_hybrid_model
from src.workspace_mode import assert_clintox_enabled, assert_tox21_enabled


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper (must match training wrapper in scripts/train_hybrid.py)
# ─────────────────────────────────────────────────────────────────────────────

class HybridModelWrapper(nn.Module):
    """Routes SMILES token IDs from the batch object into model.forward()."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.model = m

    def forward(self, batch):
        return self.model(
            batch,
            smiles_token_ids=batch.smiles_token_ids
                if hasattr(batch, "smiles_token_ids") else None,
            smiles_attention_mask=batch.smiles_attention_masks
                if hasattr(batch, "smiles_attention_masks") else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────

class _HybridDataset:
    """Attaches SMILES token IDs to each PyG Data object for batched inference."""

    def __init__(self, pyg_dataset: list, smiles_list: List[str], tokenizer):
        self.pyg_dataset = pyg_dataset
        self.smiles_list = smiles_list
        self.tok = tokenizer

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        data = self.pyg_dataset[idx]
        ids, mask = self.tok.encode(self.smiles_list[idx])
        data.smiles_token_ids      = torch.tensor(ids,  dtype=torch.long)
        data.smiles_attention_mask = torch.tensor(mask, dtype=torch.long)
        return data


def _collate(batch):
    b = Batch.from_data_list(batch)
    if hasattr(batch[0], "smiles_token_ids"):
        b.smiles_token_ids       = torch.stack([x.smiles_token_ids      for x in batch])
        b.smiles_attention_masks = torch.stack([x.smiles_attention_mask for x in batch])
    return b


def _collate_graph(batch):
    """Simple collate for plain graph-only inference."""
    return Batch.from_data_list(batch)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    model_dir:   Path,
    config_path: Path,
    device:      str = "cpu",
    enforce_workspace_mode: bool = True,
):
    """
    Load a trained SMILESGNN checkpoint.

    IMPORTANT: tokenizer is loaded first so we can read the actual vocab size
    (the config stores 100 as an upper bound; the checkpoint was saved with
    the real vocab size of ~69 tokens — using 100 causes a shape mismatch).

    Returns
    -------
    (model, tokenizer, wrapped_model)  — all in eval mode on `device`.
    """
    if enforce_workspace_mode:
        assert_clintox_enabled("load_model (SMILESGNN/ClinTox)")

    model_dir   = Path(model_dir)
    config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config["model"]

    # Load tokenizer first — its vocab size governs the embedding layer shape
    tok_path = model_dir / "tokenizer.pkl"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tok_path}\n"
            "Run 'python scripts/train_hybrid.py' first."
        )
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    actual_vocab_size = len(tokenizer.token_to_id)

    num_node_features, num_edge_features = get_feature_dims()
    model = create_hybrid_model(
        num_node_features = num_node_features,
        num_edge_features = num_edge_features,
        hidden_dim        = int(mc["hidden_dim"]),
        num_graph_layers  = int(mc["num_graph_layers"]),
        graph_model       = mc["graph_model"],
        num_heads         = int(mc["num_heads"]),
        dropout           = float(mc["dropout"]),
        use_residual      = bool(mc.get("use_residual", True)),
        use_jk            = bool(mc.get("use_jk", True)),
        jk_mode           = mc.get("jk_mode", "cat"),
        graph_pooling     = mc.get("graph_pooling", "meanmax"),
        smiles_vocab_size = actual_vocab_size,   # real vocab, NOT config default
        smiles_d_model    = int(mc["smiles_d_model"]),
        smiles_num_layers = int(mc["smiles_num_layers"]),
        fusion_method     = mc.get("fusion_method", "attention"),
    )

    ckpt = model_dir / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Run 'python scripts/train_hybrid.py' first."
        )
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    wrapped = HybridModelWrapper(model)
    wrapped.eval()

    return model, tokenizer, wrapped


def load_tox21_gatv2_model(
    model_dir: Path,
    config_path: Path,
    device: str = "cpu",
):
    """
    Load a trained Tox21 multi-task GATv2 checkpoint.

    Returns
    -------
    (model, task_names) in eval mode on `device`.
    """
    assert_tox21_enabled("load_tox21_gatv2_model")

    model_dir = Path(model_dir)
    config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config["model"]

    ckpt = model_dir / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Run 'python scripts/train_tox21_gatv2.py' first."
        )

    checkpoint = torch.load(ckpt, map_location=device)
    if isinstance(checkpoint, dict) and checkpoint.get("task_names"):
        task_names = list(checkpoint["task_names"])
    else:
        task_names = get_task_names("tox21")

    num_node_features, num_edge_features = get_feature_dims()
    model = create_gatv2_model(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=int(mc["hidden_dim"]),
        num_layers=int(mc["num_layers"]),
        num_heads=int(mc["num_heads"]),
        dropout=float(mc["dropout"]),
        use_residual=bool(mc.get("use_residual", True)),
        use_jk=bool(mc.get("use_jk", True)),
        jk_mode=str(mc.get("jk_mode", "cat")),
        pooling=str(mc.get("pooling", "set2set")),
        output_dim=len(task_names),
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, task_names


def _resolve_tox21_thresholds(
    task_names: List[str],
    default_threshold: float = 0.5,
    task_thresholds: Optional[Union[List[float], np.ndarray, Dict[str, float]]] = None,
) -> np.ndarray:
    """Normalize task threshold settings into an array aligned with task_names."""
    num_tasks = len(task_names)
    if task_thresholds is None:
        return np.full(num_tasks, float(default_threshold), dtype=np.float32)

    if isinstance(task_thresholds, dict):
        return np.array(
            [float(task_thresholds.get(task, default_threshold)) for task in task_names],
            dtype=np.float32,
        )

    thresholds = np.asarray(task_thresholds, dtype=np.float32).reshape(-1)
    if thresholds.size != num_tasks:
        raise ValueError(
            f"Expected {num_tasks} task thresholds, got {thresholds.size}"
        )
    return thresholds


def predict_tox21_batch(
    smiles_list: List[str],
    model: nn.Module,
    task_names: List[str],
    device: str,
    names: Optional[List[str]] = None,
    threshold: float = 0.5,
    task_thresholds: Optional[Union[List[float], np.ndarray, Dict[str, float]]] = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Predict Tox21 assay probabilities for a batch of SMILES.

    Returns a table with per-assay probabilities and a mechanistic alert summary.
    """
    assert_tox21_enabled("predict_tox21_batch")

    threshold_arr = _resolve_tox21_thresholds(
        task_names=task_names,
        default_threshold=threshold,
        task_thresholds=task_thresholds,
    )

    valid = []
    invalid = []

    for i, smi in enumerate(smiles_list):
        name = names[i] if names else f"Mol-{i:03d}"
        try:
            data = smiles_to_pyg_data(smi, label=None)
        except Exception:
            data = None

        if data is None:
            invalid.append((name, smi))
        else:
            valid.append((name, smi, data))

    if not valid:
        rows = []
        for name, smi in invalid:
            row = {
                "Name": name,
                "SMILES": smi,
                "MechanisticAlert": "Parse error",
                "AssayHits": -1,
                "HitTasks": "Parse error",
                "MaxAssay": "—",
                "MaxAssayProb": -1.0,
            }
            for task in task_names:
                row[f"P({task})"] = None
            rows.append(row)
        return pd.DataFrame(rows)

    _, _, pyg_list = zip(*valid)
    loader = DataLoader(
        list(pyg_list),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_graph,
    )

    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    probs_matrix = np.vstack(all_probs) if all_probs else np.empty((0, len(task_names)))

    rows = []
    for (name, smi, _), prob_vec in zip(valid, probs_matrix):
        hit_idx = np.where(prob_vec >= threshold_arr)[0]
        hit_tasks = [task_names[idx] for idx in hit_idx]

        top_idx = int(np.argmax(prob_vec))
        row = {
            "Name": name,
            "SMILES": smi,
            "MechanisticAlert": bool(len(hit_tasks) > 0),
            "AssayHits": int(len(hit_tasks)),
            "HitTasks": "; ".join(hit_tasks) if hit_tasks else "None",
            "MaxAssay": task_names[top_idx],
            "MaxAssayProb": round(float(prob_vec[top_idx]), 4),
        }

        for task_idx, task in enumerate(task_names):
            row[f"P({task})"] = round(float(prob_vec[task_idx]), 4)

        rows.append(row)

    for name, smi in invalid:
        row = {
            "Name": name,
            "SMILES": smi,
            "MechanisticAlert": "Parse error",
            "AssayHits": -1,
            "HitTasks": "Parse error",
            "MaxAssay": "—",
            "MaxAssayProb": -1.0,
        }
        for task in task_names:
            row[f"P({task})"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["AssayHits", "MaxAssayProb"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(
    smiles_list:   List[str],
    tokenizer,
    wrapped_model: nn.Module,
    device:        str,
    names:         Optional[List[str]] = None,
    true_labels:   Optional[List[int]] = None,
    threshold:     float = 0.5,
    batch_size:    int   = 32,
    enforce_workspace_mode: bool = True,
) -> pd.DataFrame:
    """
    Stage A — fast batch toxicity prediction.

    Compounds that RDKit cannot featurise (e.g. organometallics, exotic
    coordination chemistry) are reported as 'Parse error' and excluded
    from model scoring.

    Parameters
    ----------
    smiles_list   : list of SMILES strings
    tokenizer     : fitted SMILESTokenizer from training run
    wrapped_model : HybridModelWrapper (eval mode)
    device        : 'cpu' or 'cuda'
    names         : optional compound names (auto-generated if None)
    true_labels   : optional ground-truth labels (0/1)
    threshold     : decision boundary (default 0.5)
    batch_size    : GPU mini-batch size

    Returns
    -------
    pd.DataFrame sorted by P(toxic) descending.
    Parse errors appear at the bottom with P(toxic) = None.
    """
    if enforce_workspace_mode:
        assert_clintox_enabled("predict_batch (SMILESGNN/ClinTox)")

    valid, invalid = [], []

    for i, smi in enumerate(smiles_list):
        name = names[i] if names else f"Mol-{i:03d}"
        lbl  = true_labels[i] if true_labels is not None else None
        try:
            d = smiles_to_pyg_data(smi, label=lbl if lbl is not None else 0)
        except Exception:
            d = None
        if d is None:
            invalid.append({
                "Name":       name,
                "SMILES":     smi,
                "P(toxic)":   None,
                "Predicted":  "Parse error",
                "True label": ("Toxic" if lbl == 1 else "Non-toxic") if lbl is not None else "—",
                "Correct":    "—",
            })
        else:
            valid.append((i, name, smi, lbl, d))

    if not valid:
        return pd.DataFrame(invalid)

    _, vnames, vsmiles, vlabels, pyg_list = zip(*valid)
    dataset = _HybridDataset(list(pyg_list), list(vsmiles), tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    all_probs: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = wrapped_model(batch).squeeze(-1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])

    rows = []
    for name, smi, lbl, prob in zip(vnames, vsmiles, vlabels, all_probs):
        pred    = 1 if prob >= threshold else 0
        correct = (pred == lbl) if lbl is not None else None
        rows.append({
            "Name":       name,
            "SMILES":     smi,
            "P(toxic)":   round(prob, 4),
            "Predicted":  "Toxic" if pred == 1 else "Non-toxic",
            "True label": ("Toxic" if lbl == 1 else "Non-toxic") if lbl is not None else "—",
            "Correct":    ("✓" if correct else "✗") if correct is not None else "—",
        })

    rows.extend(invalid)
    df = pd.DataFrame(rows).sort_values("P(toxic)", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def predict_clinical_toxicity(
    smiles: str,
    tokenizer,
    wrapped_model: nn.Module,
    device: str,
    threshold: float = 0.5,
    name: str = "Mol-000",
    enforce_workspace_mode: bool = True,
) -> Dict[str, Union[str, float, bool]]:
    """Predict binary toxicity verdict from XSmiles/SMILESGNN for one SMILES."""
    df = predict_batch(
        smiles_list=[smiles],
        tokenizer=tokenizer,
        wrapped_model=wrapped_model,
        device=device,
        names=[name],
        threshold=threshold,
        batch_size=1,
        enforce_workspace_mode=enforce_workspace_mode,
    )

    if df.empty:
        return {
            "label": "PARSE_ERROR",
            "is_toxic": False,
            "confidence": 0.0,
            "p_toxic": 0.0,
            "threshold_used": float(threshold),
        }

    row = df.iloc[0]
    predicted = str(row.get("Predicted", "Parse error"))
    if predicted == "Parse error":
        return {
            "label": "PARSE_ERROR",
            "is_toxic": False,
            "confidence": 0.0,
            "p_toxic": 0.0,
            "threshold_used": float(threshold),
        }

    p_toxic = float(row.get("P(toxic)", 0.0))
    is_toxic = bool(p_toxic >= threshold)
    confidence = abs(p_toxic - threshold) / max(float(threshold), 1.0 - float(threshold))

    return {
        "label": "TOXIC" if is_toxic else "NON_TOXIC",
        "is_toxic": is_toxic,
        "confidence": float(min(confidence, 1.0)),
        "p_toxic": float(p_toxic),
        "threshold_used": float(threshold),
    }


def predict_toxicity_mechanism(
    smiles: str,
    model: nn.Module,
    task_names: List[str],
    device: str,
    threshold: float = 0.5,
    task_thresholds: Optional[Union[List[float], np.ndarray, Dict[str, float]]] = None,
    batch_size: int = 64,
) -> Dict[str, Union[float, int, str, bool, List[str], Dict[str, float]]]:
    """Predict Tox21 mechanism-level outputs for one SMILES."""
    threshold_arr = _resolve_tox21_thresholds(
        task_names=task_names,
        default_threshold=threshold,
        task_thresholds=task_thresholds,
    )
    task_threshold_map = {
        task_name: float(threshold_arr[idx])
        for idx, task_name in enumerate(task_names)
    }

    df = predict_tox21_batch(
        smiles_list=[smiles],
        model=model,
        task_names=task_names,
        device=device,
        names=["Mol-000"],
        threshold=threshold,
        task_thresholds=task_thresholds,
        batch_size=batch_size,
    )

    if df.empty:
        return {
            "task_scores": {},
            "active_tasks": [],
            "highest_risk_task": "—",
            "highest_risk_score": -1.0,
            "assay_hits": 0,
            "mechanistic_alert": False,
            "threshold_used": float(threshold),
            "task_thresholds": task_threshold_map,
        }

    row = df.iloc[0]
    task_scores: Dict[str, float] = {}
    for task in task_names:
        val = row.get(f"P({task})", np.nan)
        task_scores[task] = float(val) if pd.notna(val) else np.nan

    active_tasks = [
        task for task, score in task_scores.items()
        if np.isfinite(score) and score >= task_threshold_map[task]
    ]

    highest_risk_task = "—"
    highest_risk_score = -1.0
    if task_scores:
        finite_items = [(t, s) for t, s in task_scores.items() if np.isfinite(s)]
        if finite_items:
            highest_risk_task, highest_risk_score = max(finite_items, key=lambda x: x[1])

    return {
        "task_scores": task_scores,
        "active_tasks": active_tasks,
        "highest_risk_task": str(highest_risk_task),
        "highest_risk_score": float(highest_risk_score),
        "assay_hits": int(len(active_tasks)),
        "mechanistic_alert": bool(len(active_tasks) > 0),
        "threshold_used": float(threshold),
        "task_thresholds": task_threshold_map,
    }


def aggregate_toxicity_verdict(clinical_is_toxic: bool, assay_hits: int) -> str:
    """Aggregate clinical + mechanism outputs into one production verdict."""
    if clinical_is_toxic and assay_hits > 0:
        return "CONFIRMED_TOXIC"
    if (not clinical_is_toxic) and assay_hits > 0:
        return "MECHANISTIC_ALERT"
    if clinical_is_toxic and assay_hits == 0:
        return "CLINICAL_CONCERN"
    return "LIKELY_SAFE"
