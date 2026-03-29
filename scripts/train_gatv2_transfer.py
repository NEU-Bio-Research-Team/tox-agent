#!/usr/bin/env python3
"""
Train a ClinTox GATv2 model initialized from a pretrained Tox21 checkpoint.

Usage:
    python scripts/train_gatv2_transfer.py [--config config/gatv2_transfer_config.yaml]
"""

import sys
from pathlib import Path
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_clintox
from src.graph_data import get_feature_dims, smiles_list_to_pyg_dataset
from src.graph_models import create_gatv2_model
from src.graph_train import create_balanced_sampler, evaluate_model, train_gatv2_model
from src.utils import save_metrics, set_seed
from src.workspace_mode import assert_clintox_enabled


def collate_fn(batch):
    """Collate function for PyG DataLoader."""
    return Batch.from_data_list(batch)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_pretrained_backbone(
    model: torch.nn.Module,
    checkpoint_path: Path,
    strict_backbone_shapes: bool = True,
) -> dict:
    """
    Load pretrained weights except prediction head.

    Returns a summary dict with loaded/skipped key statistics.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrained_state = checkpoint.get('model_state_dict', checkpoint)

    current_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in pretrained_state.items():
        if key.startswith('predictor.'):
            skipped_keys.append((key, 'predictor_head_excluded'))
            continue

        if key not in current_state:
            skipped_keys.append((key, 'missing_in_target_model'))
            continue

        if current_state[key].shape != value.shape:
            reason = (
                'shape_mismatch_strict'
                if strict_backbone_shapes
                else 'shape_mismatch_skipped'
            )
            skipped_keys.append((key, reason))
            continue

        current_state[key] = value
        loaded_keys.append(key)

    model.load_state_dict(current_state)

    return {
        'loaded_count': len(loaded_keys),
        'skipped_count': len(skipped_keys),
        'loaded_keys': loaded_keys,
        'skipped_keys': skipped_keys,
    }


def main():
    assert_clintox_enabled("scripts/train_gatv2_transfer.py")

    parser = argparse.ArgumentParser(
        description='Train ClinTox GATv2 model with Tox21 transfer initialization'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/gatv2_transfer_config.yaml',
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
    pretrained_config = config.get('pretrained', {})
    output_config = config.get('output', {})

    set_seed(int(data_config.get('seed', 42)))

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print("=" * 80)
    print("ClinTox GATv2 Transfer Training (Tox21 -> ClinTox)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Configuration: {config_path}")
    print()

    print("Loading ClinTox dataset...")
    train_df, val_df, test_df = load_clintox(
        cache_dir=str(project_root / data_config['cache_dir']),
        split_type=data_config['split_type'],
        seed=int(data_config['seed']),
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(
        "Class distribution - Train: "
        f"Toxic={train_df['CT_TOX'].sum()}, "
        f"Non-toxic={len(train_df) - train_df['CT_TOX'].sum()}"
    )

    print("\nConverting SMILES to graph representations...")
    train_dataset = smiles_list_to_pyg_dataset(
        train_df['smiles'].tolist(),
        labels=train_df['CT_TOX'].tolist(),
    )
    val_dataset = smiles_list_to_pyg_dataset(
        val_df['smiles'].tolist(),
        labels=val_df['CT_TOX'].tolist(),
    )
    test_dataset = smiles_list_to_pyg_dataset(
        test_df['smiles'].tolist(),
        labels=test_df['CT_TOX'].tolist(),
    )

    print(
        f"Train graphs: {len(train_dataset)}, "
        f"Val graphs: {len(val_dataset)}, "
        f"Test graphs: {len(test_dataset)}"
    )

    num_node_features, num_edge_features = get_feature_dims()
    print(f"Node features: {num_node_features}, Edge features: {num_edge_features}")

    print("\nCreating data loaders...")
    train_sampler = None
    if training_config.get('use_weighted_sampler', False):
        train_labels = [data.y.item() for data in train_dataset]
        train_sampler = create_balanced_sampler(train_labels)
        print("Using weighted sampler for balanced training")

    batch_size = int(training_config['batch_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
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

    pos_weight = None
    if training_config.get('loss_type') == 'weighted_bce':
        train_labels = train_df['CT_TOX'].values
        num_pos = train_labels.sum()
        num_neg = len(train_labels) - num_pos
        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        print(f"Positive class weight: {pos_weight:.4f}")

    print("\nCreating GATv2 model...")
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
        output_dim=1,
    )

    transfer_summary = {
        'loaded_count': 0,
        'skipped_count': 0,
        'loaded_keys': [],
        'skipped_keys': [],
    }

    pretrained_ckpt = pretrained_config.get('checkpoint')
    if pretrained_ckpt:
        print("\nLoading pretrained backbone from Tox21 checkpoint...")
        transfer_summary = load_pretrained_backbone(
            model=model,
            checkpoint_path=(project_root / pretrained_ckpt),
            strict_backbone_shapes=bool(pretrained_config.get('strict_backbone_shapes', True)),
        )
        print(
            f"Loaded {transfer_summary['loaded_count']} backbone tensors, "
            f"skipped {transfer_summary['skipped_count']} tensors"
        )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    print("\n" + "=" * 80)
    print("Training Model")
    print("=" * 80)

    history = train_gatv2_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(training_config['num_epochs']),
        learning_rate=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']),
        device=device,
        loss_type=str(training_config['loss_type']),
        focal_alpha=float(training_config.get('focal_alpha', 0.25)),
        focal_gamma=float(training_config.get('focal_gamma', 2.0)),
        pos_weight=pos_weight,
        early_stopping_patience=int(training_config['early_stopping_patience']),
        early_stopping_metric=str(training_config['early_stopping_metric']),
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)

    test_metrics = evaluate_model(model, test_loader, device=device, return_predictions=False)
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"PR-AUC: {test_metrics['pr_auc']:.4f}")

    model_dir = project_root / output_config.get('model_dir', 'models/gatv2_transfer_model')
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'best_model.pt'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'num_node_features': num_node_features,
            'num_edge_features': num_edge_features,
            'transfer': {
                'source_checkpoint': str(pretrained_ckpt),
                'loaded_count': transfer_summary['loaded_count'],
                'skipped_count': transfer_summary['skipped_count'],
            },
        },
        model_path,
    )
    print(f"\nModel saved to: {model_path}")

    metrics_path = model_dir / 'gatv2_transfer_metrics.txt'
    metrics_to_save = {
        'test_auc_roc': test_metrics['auc_roc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1'],
        'test_pr_auc': test_metrics['pr_auc'],
        'test_loss': test_metrics['loss'],
        'transfer_loaded_tensors': transfer_summary['loaded_count'],
        'transfer_skipped_tensors': transfer_summary['skipped_count'],
    }

    if history['val_auc_roc']:
        best_val_epoch = int(np.argmax(history['val_f1']))
        metrics_to_save['val_auc_roc'] = history['val_auc_roc'][best_val_epoch]
        metrics_to_save['val_f1'] = history['val_f1'][best_val_epoch]
        metrics_to_save['val_pr_auc'] = history['val_pr_auc'][best_val_epoch]

    save_metrics(metrics_to_save, str(metrics_path))
    print(f"Metrics saved to: {metrics_path}")

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

        if history['val_auc_roc']:
            axes[0, 1].plot(epochs, history['val_auc_roc'], label='Val AUC-ROC', color='green', marker='s')
            axes[0, 1].axhline(y=test_metrics['auc_roc'], color='red', linestyle='--',
                               label=f"Test ({test_metrics['auc_roc']:.4f})")
        axes[0, 1].set_title('AUC-ROC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC-ROC')
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        if history['val_f1']:
            axes[1, 0].plot(epochs, history['val_f1'], label='Val F1', color='orange', marker='s')
            axes[1, 0].axhline(y=test_metrics['f1'], color='red', linestyle='--',
                               label=f"Test ({test_metrics['f1']:.4f})")
        axes[1, 0].set_title('F1')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1')
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        if history['val_pr_auc']:
            axes[1, 1].plot(epochs, history['val_pr_auc'], label='Val PR-AUC', color='purple', marker='s')
            axes[1, 1].axhline(y=test_metrics['pr_auc'], color='red', linestyle='--',
                               label=f"Test ({test_metrics['pr_auc']:.4f})")
        axes[1, 1].set_title('PR-AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('PR-AUC')
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        plt.tight_layout()
        curves_path = model_dir / 'training_curves.png'
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to: {curves_path}")

    print("\n" + "=" * 80)
    print("Transfer Training Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
