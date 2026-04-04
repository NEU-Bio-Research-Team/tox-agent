#!/usr/bin/env python3
"""Train a lightweight clinical head on top of frozen Tox21 task probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Batch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.clinical_head import (
    calibrate_threshold_youden,
    calibrate_threshold_youden_cv,
    compute_binary_metrics,
    create_clinical_head,
)
from backend.data import load_clintox
from backend.featurization import featurize_batch
from backend.graph_data import smiles_to_pyg_data
from backend.inference import load_tox21_gatv2_model
from backend.utils import set_seed


def _collate_graph(batch):
    return Batch.from_data_list(batch)


def _predict_tox21_prob_matrix(
    smiles_list,
    model,
    task_names,
    device: str,
    batch_size: int,
) -> np.ndarray:
    num_tasks = len(task_names)
    probs = np.full((len(smiles_list), num_tasks), np.nan, dtype=np.float32)

    valid_indices = []
    valid_graphs = []
    for idx, smiles in enumerate(smiles_list):
        try:
            data = smiles_to_pyg_data(smiles, label=None)
        except Exception:
            data = None

        if data is None:
            continue

        valid_indices.append(idx)
        valid_graphs.append(data)

    if not valid_graphs:
        return probs

    loader = DataLoader(
        valid_graphs,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=_collate_graph,
        num_workers=0,
    )

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(batch_probs)

    pred_matrix = np.vstack(all_probs)
    for row_idx, sample_idx in enumerate(valid_indices):
        probs[sample_idx, :] = pred_matrix[row_idx, :]

    return probs


def _build_feature_names(
    task_names: List[str],
    use_ecfp4_features: bool,
    ecfp_bits: int,
) -> List[str]:
    names = [f"tox21::{task}" for task in task_names]
    if use_ecfp4_features:
        names.extend([f"ecfp4::{idx}" for idx in range(int(ecfp_bits))])
    return names


def _extract_features(
    df: pd.DataFrame,
    tox21_model,
    task_names: List[str],
    device: str,
    batch_size: int,
    tox21_missing_impute: float,
    use_ecfp4_features: bool,
    ecfp_radius: int,
    ecfp_bits: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    smiles = df["smiles"].astype(str).tolist()
    labels_raw = pd.to_numeric(df["CT_TOX"], errors="coerce").to_numpy(dtype=np.float32)

    tox21_raw = _predict_tox21_prob_matrix(
        smiles_list=smiles,
        model=tox21_model,
        task_names=task_names,
        device=device,
        batch_size=batch_size,
    )

    tox21_finite = np.isfinite(tox21_raw)
    tox21_coverage = tox21_finite.mean(axis=1).astype(np.float32)

    tox21_features = np.where(
        tox21_finite,
        tox21_raw,
        float(tox21_missing_impute),
    ).astype(np.float32)
    tox21_features = np.clip(tox21_features, 0.0, 1.0)

    parts: List[np.ndarray] = [tox21_features]
    if use_ecfp4_features:
        fp_matrix = featurize_batch(
            smiles,
            mode="fingerprint",
            radius=int(ecfp_radius),
            n_bits=int(ecfp_bits),
        ).astype(np.float32)
        parts.append(fp_matrix)

    x_all = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    label_valid = np.isfinite(labels_raw)
    meta = pd.DataFrame(
        {
            "smiles": smiles,
            "label": labels_raw,
            "label_valid": label_valid.astype(int),
            "tox21_coverage": tox21_coverage,
            "tox21_parse_error": (tox21_coverage <= 0.0).astype(int),
        }
    )

    valid_idx = np.where(label_valid)[0]
    return (
        x_all[valid_idx].astype(np.float32),
        labels_raw[valid_idx].astype(np.float32),
        meta.iloc[valid_idx].reset_index(drop=True),
    )


def _predict_head_probs(
    head_model: torch.nn.Module,
    x: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.float32)

    head_model.eval()
    tensor_x = torch.tensor(x, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_x), batch_size=int(batch_size), shuffle=False)

    out = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = head_model(batch_x)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            out.append(probs)

    return np.concatenate(out).astype(np.float32)


def _summarize_feature_signal(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int,
) -> Dict[str, object]:
    if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
        return {
            "top_features": [],
            "mean_auc_distance": float("nan"),
        }

    y_int = y.astype(int)
    rows = []
    auc_distances = []
    std_y = float(np.std(y_int))

    for idx, feat_name in enumerate(feature_names):
        values = x[:, idx].astype(float)
        std_x = float(np.std(values))

        auc_roc = float("nan")
        if std_x > 1e-12 and len(np.unique(y_int)) > 1:
            try:
                auc_roc = float(roc_auc_score(y_int, values))
            except ValueError:
                auc_roc = float("nan")

        corr = float("nan")
        if std_x > 1e-12 and std_y > 1e-12:
            corr = float(np.corrcoef(values, y_int)[0, 1])

        auc_delta = abs(auc_roc - 0.5) if np.isfinite(auc_roc) else 0.0
        auc_distances.append(float(auc_delta))
        rows.append(
            {
                "feature": feat_name,
                "auc_roc": float(auc_roc),
                "auc_distance_from_random": float(auc_delta),
                "pearson_corr": float(corr),
            }
        )

    rows.sort(key=lambda item: item["auc_distance_from_random"], reverse=True)
    top_features = rows[: max(1, int(top_k))]

    tox21_auc_dist = [
        row["auc_distance_from_random"]
        for row in rows
        if row["feature"].startswith("tox21::")
    ]
    ecfp_auc_dist = [
        row["auc_distance_from_random"]
        for row in rows
        if row["feature"].startswith("ecfp4::")
    ]

    return {
        "top_features": top_features,
        "mean_auc_distance": float(np.mean(auc_distances)) if auc_distances else float("nan"),
        "mean_auc_distance_tox21": float(np.mean(tox21_auc_dist)) if tox21_auc_dist else float("nan"),
        "mean_auc_distance_ecfp4": float(np.mean(ecfp_auc_dist)) if ecfp_auc_dist else float("nan"),
    }


def _add_coverage_metrics(metrics: Dict[str, float], meta: pd.DataFrame) -> Dict[str, float]:
    out = dict(metrics)
    cov = pd.to_numeric(meta.get("tox21_coverage"), errors="coerce").to_numpy(dtype=np.float32)
    parse_err = pd.to_numeric(meta.get("tox21_parse_error"), errors="coerce").fillna(0).to_numpy(dtype=np.int32)

    out["n_parse_error"] = int(np.sum(parse_err == 1))
    out["mean_tox21_coverage"] = float(np.nanmean(cov)) if cov.size else float("nan")
    out["p_low_coverage_lt_0_8"] = float(np.nanmean(cov < 0.8)) if cov.size else float("nan")
    out["p_low_coverage_lt_0_5"] = float(np.nanmean(cov < 0.5)) if cov.size else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train lightweight clinical head on frozen Tox21 outputs"
    )
    parser.add_argument("--tox21-model-dir", type=str, default="models/tox21_gatv2_model")
    parser.add_argument("--tox21-config", type=str, default="config/tox21_gatv2_config.yaml")
    parser.add_argument("--cache-dir", type=str, default="data")
    parser.add_argument("--split-type", type=str, default="scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tox21-batch-size", type=int, default=64)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--clinical-threshold", type=float, default=None)
    parser.add_argument("--no-calibrate-threshold", action="store_true")
    parser.add_argument(
        "--threshold-calibration",
        choices=["val", "cv"],
        default="cv",
        help="Threshold calibration strategy when threshold is not provided.",
    )
    parser.add_argument("--threshold-cv-folds", type=int, default=5)
    parser.add_argument("--tox21-missing-impute", type=float, default=0.5)
    parser.add_argument("--use-ecfp4-features", action="store_true")
    parser.add_argument("--ecfp-radius", type=int, default=2)
    parser.add_argument("--ecfp-bits", type=int, default=1024)
    parser.add_argument("--feature-signal-top-k", type=int, default=20)
    parser.add_argument("--enforce-workspace-mode", action="store_true")
    parser.add_argument("--output-dir", type=str, default="models/clinical_head_model")
    args = parser.parse_args()

    set_seed(int(args.seed))

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; fallback to CPU")
        device = "cpu"

    tox21_model_dir = project_root / args.tox21_model_dir
    tox21_config_path = project_root / args.tox21_config
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tox21_model, task_names = load_tox21_gatv2_model(
        model_dir=tox21_model_dir,
        config_path=tox21_config_path,
        device=device,
    )
    tox21_model.eval()

    train_df, val_df, test_df = load_clintox(
        cache_dir=str(project_root / args.cache_dir),
        split_type=str(args.split_type),
        seed=int(args.seed),
        enforce_workspace_mode=bool(args.enforce_workspace_mode),
    )

    print("Extracting frozen Tox21 features...")
    x_train, y_train, meta_train = _extract_features(
        train_df,
        tox21_model=tox21_model,
        task_names=task_names,
        device=device,
        batch_size=int(args.tox21_batch_size),
        tox21_missing_impute=float(args.tox21_missing_impute),
        use_ecfp4_features=bool(args.use_ecfp4_features),
        ecfp_radius=int(args.ecfp_radius),
        ecfp_bits=int(args.ecfp_bits),
    )
    x_val, y_val, meta_val = _extract_features(
        val_df,
        tox21_model=tox21_model,
        task_names=task_names,
        device=device,
        batch_size=int(args.tox21_batch_size),
        tox21_missing_impute=float(args.tox21_missing_impute),
        use_ecfp4_features=bool(args.use_ecfp4_features),
        ecfp_radius=int(args.ecfp_radius),
        ecfp_bits=int(args.ecfp_bits),
    )
    x_test, y_test, meta_test = _extract_features(
        test_df,
        tox21_model=tox21_model,
        task_names=task_names,
        device=device,
        batch_size=int(args.tox21_batch_size),
        tox21_missing_impute=float(args.tox21_missing_impute),
        use_ecfp4_features=bool(args.use_ecfp4_features),
        ecfp_radius=int(args.ecfp_radius),
        ecfp_bits=int(args.ecfp_bits),
    )

    feature_names = _build_feature_names(
        task_names=task_names,
        use_ecfp4_features=bool(args.use_ecfp4_features),
        ecfp_bits=int(args.ecfp_bits),
    )
    if x_train.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Feature dimension mismatch: x_train has {x_train.shape[1]}, "
            f"but feature name list has {len(feature_names)}"
        )

    print(
        "Feature rows - "
        f"train={x_train.shape[0]}, val={x_val.shape[0]}, test={x_test.shape[0]} | "
        f"input_dim={x_train.shape[1]}"
    )
    print(
        "Mean Tox21 coverage - "
        f"train={meta_train['tox21_coverage'].mean():.4f}, "
        f"val={meta_val['tox21_coverage'].mean():.4f}, "
        f"test={meta_test['tox21_coverage'].mean():.4f}"
    )

    if x_train.shape[0] == 0 or x_val.shape[0] == 0:
        raise RuntimeError("Insufficient valid features for training clinical head")

    feature_signal = _summarize_feature_signal(
        x=x_train,
        y=y_train,
        feature_names=feature_names,
        top_k=int(args.feature_signal_top_k),
    )
    print(
        "Feature signal check - "
        f"mean |AUC-0.5|={feature_signal.get('mean_auc_distance', float('nan')):.4f}, "
        f"tox21={feature_signal.get('mean_auc_distance_tox21', float('nan')):.4f}, "
        f"ecfp4={feature_signal.get('mean_auc_distance_ecfp4', float('nan')):.4f}"
    )

    model = create_clinical_head(
        input_dim=x_train.shape[1],
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)

    pos_count = float(np.sum(y_train == 1))
    neg_count = float(np.sum(y_train == 0))
    if pos_count > 0 and neg_count > 0:
        pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor(1.0, dtype=torch.float32, device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=int(args.head_batch_size),
        shuffle=True,
        num_workers=0,
    )

    best_state = None
    best_epoch = 0
    best_metric = -1.0
    patience = 0
    history = []

    for epoch in range(int(args.num_epochs)):
        model.train()
        losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        train_probs_epoch = _predict_head_probs(
            head_model=model,
            x=x_train,
            device=device,
            batch_size=int(args.head_batch_size),
        )
        val_probs_epoch = _predict_head_probs(
            head_model=model,
            x=x_val,
            device=device,
            batch_size=int(args.head_batch_size),
        )

        train_metrics_epoch = compute_binary_metrics(y_train, train_probs_epoch, threshold=0.5)
        val_metrics_epoch = compute_binary_metrics(y_val, val_probs_epoch, threshold=0.5)

        metric_for_early_stop = val_metrics_epoch.get("auc_roc")
        if not np.isfinite(metric_for_early_stop):
            metric_for_early_stop = val_metrics_epoch.get("f1", 0.0)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "train_auc_roc": float(train_metrics_epoch.get("auc_roc", float("nan"))),
                "train_f1": float(train_metrics_epoch.get("f1", float("nan"))),
                "val_auc_roc": float(val_metrics_epoch.get("auc_roc", float("nan"))),
                "val_f1": float(val_metrics_epoch.get("f1", float("nan"))),
            }
        )

        if metric_for_early_stop > best_metric:
            best_metric = float(metric_for_early_stop)
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"train_loss={history[-1]['train_loss']:.4f} | "
                f"val_auc={history[-1]['val_auc_roc']:.4f} | "
                f"val_f1={history[-1]['val_f1']:.4f}"
            )

        if patience >= int(args.early_stopping_patience):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is None:
        raise RuntimeError("Clinical head training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    model.eval()

    train_probs = _predict_head_probs(model, x_train, device=device, batch_size=int(args.head_batch_size))
    val_probs = _predict_head_probs(model, x_val, device=device, batch_size=int(args.head_batch_size))
    test_probs = _predict_head_probs(model, x_test, device=device, batch_size=int(args.head_batch_size))

    if args.clinical_threshold is not None:
        threshold_info = {
            "threshold": float(args.clinical_threshold),
            "youden_j": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "reason": "provided_by_user",
        }
    elif args.no_calibrate_threshold:
        threshold_info = {
            "threshold": 0.35,
            "youden_j": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "reason": "default_0_35",
        }
    elif str(args.threshold_calibration) == "cv":
        threshold_info = calibrate_threshold_youden_cv(
            y_true=y_val,
            y_prob=val_probs,
            n_splits=int(args.threshold_cv_folds),
            seed=int(args.seed),
            default_threshold=0.35,
        )
    else:
        threshold_info = calibrate_threshold_youden(
            y_true=y_val,
            y_prob=val_probs,
            default_threshold=0.35,
        )

    threshold = float(threshold_info["threshold"])

    train_metrics = _add_coverage_metrics(
        compute_binary_metrics(y_train, train_probs, threshold=threshold),
        meta_train,
    )
    val_metrics = _add_coverage_metrics(
        compute_binary_metrics(y_val, val_probs, threshold=threshold),
        meta_val,
    )
    test_metrics = _add_coverage_metrics(
        compute_binary_metrics(y_test, test_probs, threshold=threshold),
        meta_test,
    )

    feature_source = (
        "tox21_task_probabilities_plus_ecfp4"
        if bool(args.use_ecfp4_features)
        else "tox21_task_probabilities"
    )
    feature_spec = {
        "use_tox21_probabilities": True,
        "include_ecfp4": bool(args.use_ecfp4_features),
        "ecfp_radius": int(args.ecfp_radius),
        "ecfp_bits": int(args.ecfp_bits) if bool(args.use_ecfp4_features) else 0,
        "tox21_impute_value": float(args.tox21_missing_impute),
    }

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "task_names": list(task_names),
            "threshold": float(threshold),
            "best_epoch": int(best_epoch),
            "best_val_metric": float(best_metric),
            "dataset": "clintox",
            "feature_source": feature_source,
            "feature_spec": feature_spec,
            "split_type": str(args.split_type),
            "seed": int(args.seed),
        },
        checkpoint_path,
    )

    config_payload = {
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "threshold": float(threshold),
        "task_names": list(task_names),
        "checkpoint": str(checkpoint_path),
        "feature_source": feature_source,
        "feature_spec": feature_spec,
        "dataset": "clintox",
    }
    config_path = output_dir / "clinical_head_config.json"
    with open(config_path, "w") as f:
        json.dump(config_payload, f, indent=2)

    metrics_payload = {
        "direction": "direction_3_clinical_head",
        "threshold": threshold_info,
        "best_epoch": int(best_epoch),
        "best_val_metric": float(best_metric),
        "feature_signal": feature_signal,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "config": {
            "tox21_model_dir": str(tox21_model_dir),
            "tox21_config_path": str(tox21_config_path),
            "split_type": str(args.split_type),
            "seed": int(args.seed),
            "device": device,
            "tox21_batch_size": int(args.tox21_batch_size),
            "head_batch_size": int(args.head_batch_size),
            "num_epochs": int(args.num_epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "threshold_calibration": str(args.threshold_calibration),
            "threshold_cv_folds": int(args.threshold_cv_folds),
            "feature_source": feature_source,
            "feature_spec": feature_spec,
        },
        "history": history,
    }
    metrics_path = output_dir / "clinical_head_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    train_pred = pd.DataFrame(
        {
            "smiles": meta_train["smiles"].values,
            "label": y_train.astype(int),
            "p_clinical_head": train_probs,
            "tox21_coverage": meta_train["tox21_coverage"].to_numpy(dtype=np.float32),
            "pred_label": (train_probs >= threshold).astype(int),
        }
    )
    val_pred = pd.DataFrame(
        {
            "smiles": meta_val["smiles"].values,
            "label": y_val.astype(int),
            "p_clinical_head": val_probs,
            "tox21_coverage": meta_val["tox21_coverage"].to_numpy(dtype=np.float32),
            "pred_label": (val_probs >= threshold).astype(int),
        }
    )
    test_pred = pd.DataFrame(
        {
            "smiles": meta_test["smiles"].values,
            "label": y_test.astype(int),
            "p_clinical_head": test_probs,
            "tox21_coverage": meta_test["tox21_coverage"].to_numpy(dtype=np.float32),
            "pred_label": (test_probs >= threshold).astype(int),
        }
    )

    train_pred.to_csv(output_dir / "clinical_head_train_predictions.csv", index=False)
    val_pred.to_csv(output_dir / "clinical_head_val_predictions.csv", index=False)
    test_pred.to_csv(output_dir / "clinical_head_test_predictions.csv", index=False)

    print("=" * 80)
    print("Direction 3 - Clinical Head Training Complete")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Threshold used: {threshold:.4f} ({threshold_info.get('reason')})")
    print(f"Train AUC/F1: {train_metrics['auc_roc']:.4f} / {train_metrics['f1']:.4f}")
    print(f"Val   AUC/F1: {val_metrics['auc_roc']:.4f} / {val_metrics['f1']:.4f}")
    print(f"Test  AUC/F1: {test_metrics['auc_roc']:.4f} / {test_metrics['f1']:.4f}")
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
