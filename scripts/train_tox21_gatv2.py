#!/usr/bin/env python3
"""
Main training script for Tox21 multi-task GATv2 molecular property prediction.

Usage:
    python scripts/train_tox21_gatv2.py [--config config/tox21_gatv2_config.yaml]
"""

import sys
from pathlib import Path
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import get_task_names, load_tox21
from src.graph_data import get_feature_dims, smiles_list_to_pyg_dataset
from src.graph_models import create_gatv2_model
from src.graph_train import (
    compute_multitask_pos_weights,
    evaluate_multitask_model,
    train_multitask_model,
)
from src.utils import save_metrics, set_seed


def collate_fn(batch):
    """Collate function for PyG DataLoader."""
    return Batch.from_data_list(batch)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def select_task_columns(df: pd.DataFrame) -> list:
    """Select Tox21 task columns from a dataframe."""
    expected_tasks = get_task_names("tox21")
    task_columns = [task for task in expected_tasks if task in df.columns]

    if task_columns:
        return task_columns

    fallback = [c for c in df.columns if c != "smiles"]
    if not fallback:
        raise ValueError("No task columns found for Tox21 dataframe")
    return fallback


def print_task_coverage(df: pd.DataFrame, task_columns: list, split_name: str):
    """Print per-task label coverage for split sanity checks."""
    print(f"\n{split_name} task coverage:")
    for task in task_columns:
        values = df[task].values
        n_valid = int(np.isfinite(values).sum())
        n_pos = int(np.nansum(values == 1))
        n_neg = int(np.nansum(values == 0))
        print(f"  {task:14s} valid={n_valid:4d}  pos={n_pos:4d}  neg={n_neg:4d}")


def main():
    parser = argparse.ArgumentParser(
        description='Train GATv2 model for Tox21 multi-task molecular property prediction'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/tox21_gatv2_config.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu or cuda)',
    )
    args = parser.parse_args()

    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(config_path)
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    output_config = config.get('output', {})

    set_seed(int(data_config.get('seed', 42)))

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print("=" * 80)
    print("Tox21 Multi-task GATv2 - Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Configuration: {config_path}")
    print()

    print("Loading Tox21 dataset...")
    train_df, val_df, test_df = load_tox21(
        cache_dir=str(project_root / data_config['cache_dir']),
        split_type=data_config['split_type'],
        seed=int(data_config['seed']),
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    task_columns = select_task_columns(train_df)
    print(f"Detected {len(task_columns)} Tox21 tasks")
    print(f"Tasks: {task_columns}")

    print_task_coverage(train_df, task_columns, "Train")
    print_task_coverage(val_df, task_columns, "Val")
    print_task_coverage(test_df, task_columns, "Test")

    train_labels = train_df[task_columns].astype(np.float32).values
    val_labels = val_df[task_columns].astype(np.float32).values
    test_labels = test_df[task_columns].astype(np.float32).values

    print("\nConverting SMILES to graph representations...")
    train_dataset = smiles_list_to_pyg_dataset(
        train_df['smiles'].tolist(),
        labels=train_labels.tolist(),
    )
    val_dataset = smiles_list_to_pyg_dataset(
        val_df['smiles'].tolist(),
        labels=val_labels.tolist(),
    )
    test_dataset = smiles_list_to_pyg_dataset(
        test_df['smiles'].tolist(),
        labels=test_labels.tolist(),
    )

    print(
        f"Train graphs: {len(train_dataset)}, "
        f"Val graphs: {len(val_dataset)}, "
        f"Test graphs: {len(test_dataset)}"
    )

    num_node_features, num_edge_features = get_feature_dims()
    print(f"Node features: {num_node_features}, Edge features: {num_edge_features}")

    batch_size = int(training_config['batch_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print("\nCreating multi-task GATv2 model...")
    model = create_gatv2_model(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=int(model_config['hidden_dim']),
        num_layers=int(model_config['num_layers']),
        num_heads=int(model_config['num_heads']),
        dropout=float(model_config['dropout']),
        use_residual=bool(model_config['use_residual']),
        use_jk=bool(model_config['use_jk']),
        jk_mode=str(model_config['jk_mode']),
        pooling=str(model_config['pooling']),
        output_dim=len(task_columns),
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    pos_weight = None
    if training_config.get('use_task_pos_weights', False):
        pos_weight = compute_multitask_pos_weights(train_labels)
        print("Using per-task positive class weights")

    print("\n" + "=" * 80)
    print("Training Model")
    print("=" * 80)

    history = train_multitask_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_names=task_columns,
        num_epochs=int(training_config['num_epochs']),
        learning_rate=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']),
        device=device,
        pos_weight=pos_weight,
        early_stopping_patience=int(training_config['early_stopping_patience']),
        early_stopping_metric=str(training_config['early_stopping_metric']),
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("Evaluating on Validation and Test")
    print("=" * 80)

    val_metrics = evaluate_multitask_model(
        model=model,
        data_loader=val_loader,
        task_names=task_columns,
        device=device,
        return_predictions=False,
    )
    test_metrics = evaluate_multitask_model(
        model=model,
        data_loader=test_loader,
        task_names=task_columns,
        device=device,
        return_predictions=False,
    )

    print("Validation metrics:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Macro AUC-ROC: {val_metrics['macro_auc_roc']:.4f}")
    print(f"  Macro PR-AUC: {val_metrics['macro_pr_auc']:.4f}")
    print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"  Micro AUC-ROC: {val_metrics['micro_auc_roc']:.4f}")
    print(f"  Micro PR-AUC: {val_metrics['micro_pr_auc']:.4f}")

    print("\nTest metrics:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Macro AUC-ROC: {test_metrics['macro_auc_roc']:.4f}")
    print(f"  Macro PR-AUC: {test_metrics['macro_pr_auc']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Micro AUC-ROC: {test_metrics['micro_auc_roc']:.4f}")
    print(f"  Micro PR-AUC: {test_metrics['micro_pr_auc']:.4f}")

    model_dir = project_root / output_config.get('model_dir', 'models/tox21_gatv2_model')
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'best_model.pt'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'num_node_features': num_node_features,
            'num_edge_features': num_edge_features,
            'task_names': task_columns,
        },
        model_path,
    )
    print(f"\nModel saved to: {model_path}")

    metrics_path = model_dir / 'tox21_gatv2_metrics.txt'
    metrics_to_save = {
        'val_loss': val_metrics['loss'],
        'val_macro_auc_roc': val_metrics['macro_auc_roc'],
        'val_macro_pr_auc': val_metrics['macro_pr_auc'],
        'val_macro_f1': val_metrics['macro_f1'],
        'val_micro_auc_roc': val_metrics['micro_auc_roc'],
        'val_micro_pr_auc': val_metrics['micro_pr_auc'],
        'test_loss': test_metrics['loss'],
        'test_macro_auc_roc': test_metrics['macro_auc_roc'],
        'test_macro_pr_auc': test_metrics['macro_pr_auc'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_micro_auc_roc': test_metrics['micro_auc_roc'],
        'test_micro_pr_auc': test_metrics['micro_pr_auc'],
    }
    save_metrics(metrics_to_save, str(metrics_path))
    print(f"Metrics saved to: {metrics_path}")

    task_metrics_df = pd.DataFrame.from_dict(test_metrics['task_metrics'], orient='index')
    task_metrics_df.index.name = 'task'
    task_metrics_df = task_metrics_df.reset_index()
    task_metrics_path = model_dir / 'tox21_task_metrics.csv'
    task_metrics_df.to_csv(task_metrics_path, index=False)
    print(f"Per-task metrics saved to: {task_metrics_path}")

    if len(history['train_loss']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = range(1, len(history['train_loss']) + 1)

        axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        if history['val_loss']:
            axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        if history['val_macro_auc_roc']:
            axes[0, 1].plot(epochs, history['val_macro_auc_roc'], label='Val Macro AUC-ROC', color='green', marker='s')
            axes[0, 1].axhline(y=test_metrics['macro_auc_roc'], color='red', linestyle='--',
                               label=f"Test ({test_metrics['macro_auc_roc']:.4f})")
        axes[0, 1].set_title('Macro AUC-ROC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC-ROC')
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        if history['val_macro_pr_auc']:
            axes[1, 0].plot(epochs, history['val_macro_pr_auc'], label='Val Macro PR-AUC', color='purple', marker='s')
            axes[1, 0].axhline(y=test_metrics['macro_pr_auc'], color='red', linestyle='--',
                               label=f"Test ({test_metrics['macro_pr_auc']:.4f})")
        axes[1, 0].set_title('Macro PR-AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PR-AUC')
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        if history['val_macro_f1']:
            axes[1, 1].plot(epochs, history['val_macro_f1'], label='Val Macro F1', color='orange', marker='s')
            axes[1, 1].axhline(y=test_metrics['macro_f1'], color='red', linestyle='--',
                               label=f"Test ({test_metrics['macro_f1']:.4f})")
        axes[1, 1].set_title('Macro F1')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1')
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        plt.tight_layout()
        curves_path = model_dir / 'training_curves.png'
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to: {curves_path}")

    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
