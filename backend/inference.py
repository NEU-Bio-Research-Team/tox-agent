"""
Inference utilities for SMILESGNN toxicity predictor.

Provides:
  - load_model()     — load trained checkpoint + tokenizer
  - predict_batch()  — fast Stage A batch scoring
    - predict_xsmiles_toxicity_profile() — per-task toxicity profiling from XSmiles Tox21 head

HybridModelWrapper routes SMILES token IDs from the PyG batch
object to model.forward(). This must match the wrapper used during
training, otherwise the SMILES encoder is bypassed (zero vectors).
"""

import json
import pickle
import os
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
from backend.clinical_head import create_clinical_head, scores_dict_to_feature_vector
from backend.featurization import featurize_fingerprint


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


DEFAULT_CLINICAL_THRESHOLD = _env_float("CLINICAL_THRESHOLD", 0.35)

# Clinical proxy derived from mechanistic Tox21 tasks.
# Weights are normalized at runtime to support easy overrides.
DEFAULT_CLINICAL_PROXY_TASK_WEIGHTS: Dict[str, float] = {
    "SR-p53": 0.30,
    "SR-MMP": 0.25,
    "SR-ARE": 0.20,
    "NR-AhR": 0.15,
    "SR-HSE": 0.10,
}


def _normalize_proxy_weights(
    task_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Normalize and sanitize clinical proxy weights."""
    raw = task_weights or DEFAULT_CLINICAL_PROXY_TASK_WEIGHTS

    cleaned: Dict[str, float] = {}
    total = 0.0
    for task, weight in raw.items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if (not np.isfinite(w)) or w <= 0:
            continue
        cleaned[str(task)] = w
        total += w

    if total <= 0.0:
        return dict(DEFAULT_CLINICAL_PROXY_TASK_WEIGHTS)

    if abs(total - 1.0) < 1e-8:
        return cleaned

    return {task: (weight / total) for task, weight in cleaned.items()}


def clinical_score_from_tox21(
    task_scores: Dict[str, float],
    task_weights: Optional[Dict[str, float]] = None,
    renormalize_missing: bool = False,
    missing_task_value: float = 0.5,
) -> Dict[str, Union[float, List[str]]]:
    """Aggregate selected Tox21 task probabilities into a clinical proxy score."""
    weights = _normalize_proxy_weights(task_weights)
    total_weight = float(np.sum(list(weights.values())))

    weighted_sum = 0.0
    used_weight_sum = 0.0
    tasks_used: List[str] = []
    tasks_missing: List[str] = []

    try:
        fallback_value = float(missing_task_value)
    except (TypeError, ValueError):
        fallback_value = 0.5
    fallback_value = float(np.clip(fallback_value, 0.0, 1.0))

    for task, weight in weights.items():
        raw_score = task_scores.get(task, np.nan)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = np.nan

        if not np.isfinite(score):
            tasks_missing.append(task)
            if not renormalize_missing:
                weighted_sum += weight * fallback_value
            continue

        clipped_score = float(np.clip(float(score), 0.0, 1.0))
        weighted_sum += weight * clipped_score
        used_weight_sum += weight
        tasks_used.append(task)

    if total_weight <= 0.0:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "tasks_used": [],
            "tasks_missing": list(weights.keys()),
        }

    if renormalize_missing and used_weight_sum <= 0.0:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "tasks_used": [],
            "tasks_missing": list(weights.keys()),
        }

    denom = used_weight_sum if renormalize_missing else total_weight

    return {
        "score": float(weighted_sum / denom),
        "coverage": float(used_weight_sum),
        "tasks_used": tasks_used,
        "tasks_missing": tasks_missing,
    }


def predict_clinical_proxy_from_tox21(
    mechanism_result: Dict[str, Union[float, int, str, bool, List[str], Dict[str, float]]],
    threshold: float = DEFAULT_CLINICAL_THRESHOLD,
    task_weights: Optional[Dict[str, float]] = None,
    renormalize_missing: bool = False,
    missing_task_value: float = 0.5,
) -> Dict[str, Union[str, float, bool, List[str]]]:
    """Return a clinical-style toxicity output derived from Tox21 mechanism scores."""
    proxy = clinical_score_from_tox21(
        task_scores=dict(mechanism_result.get("task_scores") or {}),
        task_weights=task_weights,
        renormalize_missing=renormalize_missing,
        missing_task_value=missing_task_value,
    )

    p_toxic = float(proxy.get("score", 0.0))
    is_toxic = bool(p_toxic >= threshold)
    confidence = abs(p_toxic - threshold) / max(float(threshold), 1.0 - float(threshold))

    return {
        "label": "TOXIC" if is_toxic else "NON_TOXIC",
        "is_toxic": is_toxic,
        "confidence": float(min(confidence, 1.0)),
        "p_toxic": p_toxic,
        "threshold_used": float(threshold),
        "source": "tox21_proxy",
        "proxy_tasks_used": list(proxy.get("tasks_used") or []),
        "proxy_tasks_missing": list(proxy.get("tasks_missing") or []),
        "proxy_coverage": float(proxy.get("coverage", 0.0)),
    }


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
        output_dim        = int(mc.get("clinical_output_dim", 1)),
        tox21_output_dim  = int(mc.get("tox21_output_dim", 0)),
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


def load_clinical_head_model(
    model_dir: Path,
    device: str = "cpu",
) -> tuple[
    nn.Module,
    Dict[str, Union[float, int, bool, List[str], str, Dict[str, Union[float, int, bool, str]]]],
]:
    """Load a lightweight clinical head trained on Tox21 task features."""
    model_dir = Path(model_dir)

    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Clinical head checkpoint not found: {ckpt_path}\n"
            "Run 'python scripts/train_clinical_head.py' first."
        )

    config_path = model_dir / "clinical_head_config.json"
    config_payload: Dict[str, Union[float, int, bool, List[str], str, Dict[str, Union[float, int, bool, str]]]] = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            raw_cfg = json.load(f)
        if isinstance(raw_cfg, dict):
            config_payload = raw_cfg

    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        input_dim = int(checkpoint.get("input_dim", config_payload.get("input_dim", 12)))
        hidden_dim = int(checkpoint.get("hidden_dim", config_payload.get("hidden_dim", 32)))
        dropout = float(checkpoint.get("dropout", config_payload.get("dropout", 0.1)))
        threshold = float(checkpoint.get("threshold", config_payload.get("threshold", DEFAULT_CLINICAL_THRESHOLD)))
        task_names = checkpoint.get("task_names") or config_payload.get("task_names") or get_task_names("tox21")
        feature_source = str(checkpoint.get("feature_source", config_payload.get("feature_source", "tox21_task_probabilities")))
        feature_spec = checkpoint.get("feature_spec", config_payload.get("feature_spec", {}))
    else:
        state_dict = checkpoint
        input_dim = int(config_payload.get("input_dim", 12))
        hidden_dim = int(config_payload.get("hidden_dim", 32))
        dropout = float(config_payload.get("dropout", 0.1))
        threshold = float(config_payload.get("threshold", DEFAULT_CLINICAL_THRESHOLD))
        task_names = config_payload.get("task_names") or get_task_names("tox21")
        feature_source = str(config_payload.get("feature_source", "tox21_task_probabilities"))
        feature_spec = config_payload.get("feature_spec", {})

    if not isinstance(feature_spec, dict):
        feature_spec = {}

    feature_spec = {
        "use_tox21_probabilities": bool(feature_spec.get("use_tox21_probabilities", True)),
        "include_ecfp4": bool(feature_spec.get("include_ecfp4", False)),
        "ecfp_radius": int(feature_spec.get("ecfp_radius", 2)),
        "ecfp_bits": int(feature_spec.get("ecfp_bits", 1024)),
        "tox21_impute_value": float(feature_spec.get("tox21_impute_value", 0.5)),
    }

    model = create_clinical_head(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata: Dict[str, Union[float, int, bool, List[str], str, Dict[str, Union[float, int, bool, str]]]] = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "threshold": threshold,
        "task_names": list(task_names),
        "feature_source": feature_source,
        "feature_spec": feature_spec,
        "model_dir": str(model_dir),
    }
    return model, metadata


def _build_clinical_head_feature_vector(
    task_scores: Dict[str, float],
    task_names: List[str],
    smiles: Optional[str],
    feature_spec: Optional[Dict[str, Union[float, int, bool, str]]] = None,
) -> np.ndarray:
    spec = feature_spec or {}

    use_tox21_probabilities = bool(spec.get("use_tox21_probabilities", True))
    include_ecfp4 = bool(spec.get("include_ecfp4", False))
    ecfp_radius = int(spec.get("ecfp_radius", 2))
    ecfp_bits = int(spec.get("ecfp_bits", 1024))
    tox21_impute_value = float(spec.get("tox21_impute_value", 0.5))

    parts: List[np.ndarray] = []

    if use_tox21_probabilities:
        tox21_features = scores_dict_to_feature_vector(
            task_scores=task_scores,
            task_names=task_names,
            missing_value=tox21_impute_value,
        )
        parts.append(tox21_features)

    if include_ecfp4:
        if smiles:
            ecfp_features = featurize_fingerprint(
                smiles=str(smiles),
                radius=int(ecfp_radius),
                n_bits=int(ecfp_bits),
            )
        else:
            ecfp_features = np.zeros(int(ecfp_bits), dtype=np.float32)
        parts.append(ecfp_features.astype(np.float32))

    if not parts:
        return np.zeros(len(task_names), dtype=np.float32)

    if len(parts) == 1:
        return parts[0].astype(np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def predict_clinical_head_from_tox21_task_scores(
    task_scores: Dict[str, float],
    task_names: List[str],
    clinical_head_model: nn.Module,
    threshold: float = DEFAULT_CLINICAL_THRESHOLD,
    device: str = "cpu",
    smiles: Optional[str] = None,
    feature_spec: Optional[Dict[str, Union[float, int, bool, str]]] = None,
) -> Dict[str, Union[str, float, bool]]:
    """Predict clinical toxicity from Tox21 task score vector using trained head."""
    features = _build_clinical_head_feature_vector(
        task_scores=task_scores,
        task_names=task_names,
        smiles=smiles,
        feature_spec=feature_spec,
    )
    x = torch.tensor(features, dtype=torch.float32, device=device)

    with torch.no_grad():
        logit = clinical_head_model(x)
        prob = torch.sigmoid(logit).detach().cpu().item()

    is_toxic = bool(prob >= threshold)
    confidence = abs(prob - threshold) / max(float(threshold), 1.0 - float(threshold))

    return {
        "label": "TOXIC" if is_toxic else "NON_TOXIC",
        "is_toxic": is_toxic,
        "confidence": float(min(confidence, 1.0)),
        "p_toxic": float(prob),
        "threshold_used": float(threshold),
        "source": "clinical_head",
    }


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
    threshold:     float = DEFAULT_CLINICAL_THRESHOLD,
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
    threshold     : decision boundary (default 0.35)
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
    threshold: float = DEFAULT_CLINICAL_THRESHOLD,
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


def predict_xsmiles_toxicity_profile(
    smiles: str,
    tokenizer,
    model: nn.Module,
    device: str,
    task_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    task_thresholds: Optional[Union[List[float], np.ndarray, Dict[str, float]]] = None,
) -> Dict[str, Union[float, int, str, bool, List[str], Dict[str, float]]]:
    """
    Predict Tox21-like mechanism profile directly from XSmiles multi-task head.

    This requires a checkpoint created with tox21_output_dim > 0.
    """
    if task_names is None:
        task_names = get_task_names("tox21")

    if not hasattr(model, "forward_tox21") or getattr(model, "tox21_head", None) is None:
        raise RuntimeError(
            "Loaded XSmiles checkpoint does not include Tox21 head. "
            "Train with tox21_output_dim > 0 to use toxicity profiling mode."
        )

    graph = smiles_to_pyg_data(smiles, label=None)
    if graph is None:
        return {
            "task_scores": {},
            "active_tasks": [],
            "highest_risk_task": "Parse error",
            "highest_risk_score": -1.0,
            "assay_hits": -1,
            "mechanistic_alert": False,
            "threshold_used": float(threshold),
            "task_thresholds": {},
            "source": "xsmiles_multitask_head",
        }

    dataset = _HybridDataset([graph], [str(smiles)], tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate)

    with torch.no_grad():
        batch = next(iter(loader))
        batch = batch.to(device)
        logits = model.forward_tox21(
            batch,
            smiles_token_ids=batch.smiles_token_ids if hasattr(batch, "smiles_token_ids") else None,
            smiles_attention_mask=batch.smiles_attention_masks if hasattr(batch, "smiles_attention_masks") else None,
        )
        prob_vec = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

    if len(task_names) != prob_vec.shape[0]:
        task_names = [f"task_{i}" for i in range(prob_vec.shape[0])]

    threshold_arr = _resolve_tox21_thresholds(
        task_names=task_names,
        default_threshold=threshold,
        task_thresholds=task_thresholds,
    )
    task_threshold_map = {
        task_name: float(threshold_arr[idx])
        for idx, task_name in enumerate(task_names)
    }

    task_scores = {
        task_name: float(prob_vec[idx])
        for idx, task_name in enumerate(task_names)
    }
    active_tasks = [
        task_name
        for idx, task_name in enumerate(task_names)
        if float(prob_vec[idx]) >= float(threshold_arr[idx])
    ]

    top_idx = int(np.argmax(prob_vec))
    return {
        "task_scores": task_scores,
        "active_tasks": active_tasks,
        "highest_risk_task": str(task_names[top_idx]),
        "highest_risk_score": float(prob_vec[top_idx]),
        "assay_hits": int(len(active_tasks)),
        "mechanistic_alert": bool(len(active_tasks) > 0),
        "threshold_used": float(threshold),
        "task_thresholds": task_threshold_map,
        "source": "xsmiles_multitask_head",
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
