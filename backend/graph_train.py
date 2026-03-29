"""
Training utilities for graph neural network models.

Provides training loops, loss functions (focal loss, weighted BCE),
and evaluation functions for molecular property prediction with class imbalance handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.data import Batch
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard examples,
    which is particularly useful for imbalanced datasets.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size,)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t (probability of true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t (weighting factor)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for class imbalance.
    """
    
    def __init__(self, pos_weight: Optional[float] = None, reduction: str = 'mean'):
        """
        Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive class (typically num_neg / num_pos)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size,)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Weighted BCE loss value
        """
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor(self.pos_weight, device=inputs.device)
            return F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=pos_weight_tensor, reduction=self.reduction
            )
        else:
            return F.binary_cross_entropy_with_logits(
                inputs, targets, reduction=self.reduction
            )


def create_balanced_sampler(
    labels: List[float],
    replacement: bool = True
) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for balanced sampling.
    
    Args:
        labels: List of labels (0 or 1)
        replacement: Whether to sample with replacement
    
    Returns:
        WeightedRandomSampler instance
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels.astype(int))
    
    # Compute weights for each sample
    # Weight = 1 / (num_samples_in_class * num_classes)
    weights = np.zeros(len(labels))
    for i in range(len(class_counts)):
        if class_counts[i] > 0:
            weights[labels == i] = 1.0 / (class_counts[i] * len(class_counts))
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    
    return WeightedRandomSampler(
        weights=weights.astype(float),
        num_samples=len(labels),
        replacement=replacement
    )


def train_gatv2_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    loss_type: str = "focal",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: Optional[float] = None,
    early_stopping_patience: int = 20,
    early_stopping_metric: str = "f1",
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train a GATv2 model for molecular property prediction.
    
    Args:
        model: GATv2MolecularPredictor model
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        device: Device to train on ('cpu' or 'cuda')
        loss_type: Loss function type ('focal', 'weighted_bce', or 'bce')
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        pos_weight: Positive class weight for weighted BCE loss
        early_stopping_patience: Number of epochs to wait before early stopping
        early_stopping_metric: Metric to use for early stopping ('f1', 'auc_roc', 'loss')
        verbose: Whether to print training progress
    
    Returns:
        Dictionary with training history containing:
        - 'train_loss': Training loss per epoch
        - 'val_loss': Validation loss per epoch
        - 'val_auc_roc': Validation AUC-ROC per epoch
        - 'val_f1': Validation F1 score per epoch
        - 'val_pr_auc': Validation PR-AUC per epoch
    """
    model = model.to(device)
    
    # Initialize loss function
    if loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == "weighted_bce":
        criterion = WeightedBCELoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler_kwargs = {
        'mode': 'max' if early_stopping_metric != 'loss' else 'min',
        'factor': 0.5,
        'patience': 10,
    }
    if 'verbose' in inspect.signature(
        torch.optim.lr_scheduler.ReduceLROnPlateau
    ).parameters:
        scheduler_kwargs['verbose'] = verbose

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **scheduler_kwargs,
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc_roc': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_pr_auc': []
    }
    
    # Early stopping
    best_metric = float('-inf') if early_stopping_metric != 'loss' else float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.y.squeeze()
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch).squeeze()
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Skip NaN/Inf losses — prevents corrupted gradients from
            # poisoning model weights (observed on CPU, early epochs)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device=device)
            
            history['val_loss'].append(val_metrics['loss'])
            history['val_auc_roc'].append(val_metrics['auc_roc'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_pr_auc'].append(val_metrics['pr_auc'])
            
            # Early stopping logic
            if early_stopping_metric == 'f1':
                current_metric = val_metrics['f1']
                is_better = current_metric > best_metric
            elif early_stopping_metric == 'auc_roc':
                current_metric = val_metrics['auc_roc']
                is_better = current_metric > best_metric
            else:  # loss
                current_metric = val_metrics['loss']
                is_better = current_metric < best_metric
            
            if is_better:
                best_metric = current_metric
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Update learning rate
            scheduler.step(current_metric)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val AUC-ROC: {val_metrics['auc_roc']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        elif verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    return_predictions: bool = False
) -> Dict:
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on
        return_predictions: Whether to return predictions and labels
    
    Returns:
        Dictionary containing:
        - 'loss': Average loss
        - 'auc_roc': AUC-ROC score
        - 'accuracy': Accuracy
        - 'f1': F1 score
        - 'pr_auc': PR-AUC score
        - 'confusion_matrix': Confusion matrix
        - 'predictions': Predictions (if return_predictions=True)
        - 'labels': Labels (if return_predictions=True)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    losses = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            labels = batch.y.squeeze()
            
            # Forward pass
            logits = model(batch).squeeze()
            
            # Compute loss
            loss = criterion(logits, labels)
            losses.append(loss.item())
            
            # Get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Guard against NaN/Inf from early-epoch numerical instability
    # (observed on Windows with certain sklearn/numpy versions)
    if not np.all(np.isfinite(all_preds)):
        all_preds = np.nan_to_num(all_preds, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Binary predictions (threshold = 0.5)
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Compute metrics
    metrics = {
        'loss': np.mean(losses),
        'auc_roc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
        'accuracy': accuracy_score(all_labels, binary_preds),
        'f1': f1_score(all_labels, binary_preds, zero_division=0.0),
        'pr_auc': average_precision_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(all_labels, binary_preds).tolist()
    }
    
    if return_predictions:
        metrics['predictions'] = all_preds
        metrics['labels'] = all_labels
        metrics['logits'] = all_logits
    
    return metrics


def compute_multitask_pos_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute per-task positive-class weights for multi-task classification.

    Missing labels must be encoded as NaN and are ignored in counts.

    Args:
        labels: Label matrix of shape (num_samples, num_tasks) with values in
            {0, 1, NaN}.

    Returns:
        Tensor of shape (num_tasks,) where each value is neg/pos for a task.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array of shape (num_samples, num_tasks)")

    weights: List[float] = []
    for task_idx in range(labels.shape[1]):
        task_labels = labels[:, task_idx]
        valid_mask = np.isfinite(task_labels)
        valid_labels = task_labels[valid_mask]

        if valid_labels.size == 0:
            weights.append(1.0)
            continue

        num_pos = np.sum(valid_labels == 1)
        num_neg = np.sum(valid_labels == 0)

        if num_pos == 0 or num_neg == 0:
            weights.append(1.0)
        else:
            weights.append(float(num_neg / num_pos))

    return torch.tensor(weights, dtype=torch.float32)


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogits loss that ignores missing labels (NaN entries).

    This is required for datasets such as Tox21 where each assay can be
    missing for a subset of molecules.
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)

        valid_mask = torch.isfinite(targets)
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(inputs.device)

        loss = F.binary_cross_entropy_with_logits(
            inputs,
            safe_targets,
            pos_weight=pos_weight,
            reduction='none'
        )

        loss = loss * valid_mask.float()
        valid_count = valid_mask.float().sum()

        if self.reduction == 'sum':
            return loss.sum()

        if self.reduction == 'mean':
            if valid_count.item() == 0:
                return inputs.sum() * 0.0
            return loss.sum() / valid_count

        return loss


class MaskedFocalLoss(nn.Module):
    """
    Focal loss variant that ignores missing labels (NaN entries).

    This is the multi-task counterpart of FocalLoss, adapted for datasets
    such as Tox21 where each task can have missing labels.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)

        valid_mask = torch.isfinite(targets)
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            safe_targets,
            reduction='none'
        )

        probs = torch.sigmoid(inputs)
        p_t = probs * safe_targets + (1 - probs) * (1 - safe_targets)
        alpha_t = self.alpha * safe_targets + (1 - self.alpha) * (1 - safe_targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        loss = focal_weight * bce_loss

        loss = loss * valid_mask.float()
        valid_count = valid_mask.float().sum()

        if self.reduction == 'sum':
            return loss.sum()

        if self.reduction == 'mean':
            if valid_count.item() == 0:
                return inputs.sum() * 0.0
            return loss.sum() / valid_count

        return loss


def _nanmean_or_zero(values: List[float]) -> float:
    """Return nanmean(values), falling back to 0.0 when all values are NaN."""
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanmean(arr))


def _resolve_thresholds(
    num_tasks: int,
    task_names: List[str],
    thresholds: Optional[Union[float, List[float], np.ndarray, Dict[str, float]]] = None
) -> np.ndarray:
    """Normalize threshold configuration to an array of shape (num_tasks,)."""
    if thresholds is None:
        return np.full(num_tasks, 0.5, dtype=np.float32)

    if isinstance(thresholds, (int, float)):
        return np.full(num_tasks, float(thresholds), dtype=np.float32)

    if isinstance(thresholds, dict):
        return np.array(
            [float(thresholds.get(task_name, 0.5)) for task_name in task_names],
            dtype=np.float32,
        )

    threshold_arr = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if threshold_arr.size != num_tasks:
        raise ValueError(
            f"Expected {num_tasks} thresholds, got {threshold_arr.size}"
        )
    return threshold_arr


def evaluate_multitask_model(
    model: nn.Module,
    data_loader: DataLoader,
    task_names: Optional[List[str]] = None,
    device: str = "cpu",
    thresholds: Optional[Union[float, List[float], np.ndarray, Dict[str, float]]] = None,
    return_predictions: bool = False,
) -> Dict:
    """
    Evaluate a multi-task binary classification model with missing labels.

    Args:
        model: Trained model returning logits of shape (batch, num_tasks).
        data_loader: Evaluation dataloader.
        task_names: Optional list of task names. If omitted, uses task_0..task_n.
        device: Evaluation device.
        thresholds: Optional classification thresholds.
        return_predictions: Whether to return raw logits/probabilities/labels.

    Returns:
        Metrics dictionary with per-task, macro, and micro summaries.
    """
    model.eval()

    all_probs_chunks: List[np.ndarray] = []
    all_labels_chunks: List[np.ndarray] = []
    all_logits_chunks: List[np.ndarray] = []
    losses: List[float] = []

    criterion = MaskedBCEWithLogitsLoss(reduction='mean')

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            labels = batch.y.float()

            logits = model(batch)
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1)
            if labels.dim() == 1 and logits.dim() == 2 and labels.numel() == logits.numel():
                labels = labels.view(logits.size(0), logits.size(1))
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            if labels.shape != logits.shape:
                raise ValueError(
                    f"Label shape {tuple(labels.shape)} does not match logits shape "
                    f"{tuple(logits.shape)} in evaluate_multitask_model"
                )

            loss = criterion(logits, labels)
            losses.append(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs_chunks.append(probs)
            all_labels_chunks.append(labels.cpu().numpy())
            all_logits_chunks.append(logits.cpu().numpy())

    if not all_probs_chunks:
        return {
            'loss': 0.0,
            'macro_auc_roc': 0.0,
            'macro_pr_auc': 0.0,
            'macro_accuracy': 0.0,
            'macro_f1': 0.0,
            'micro_auc_roc': 0.0,
            'micro_pr_auc': 0.0,
            'micro_accuracy': 0.0,
            'micro_f1': 0.0,
            'task_metrics': {}
        }

    all_probs = np.vstack(all_probs_chunks)
    all_labels = np.vstack(all_labels_chunks)
    all_logits = np.vstack(all_logits_chunks)

    if not np.all(np.isfinite(all_probs)):
        all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0, neginf=0.0)

    num_tasks = all_probs.shape[1]
    if task_names is None:
        task_names = [f"task_{i}" for i in range(num_tasks)]
    if len(task_names) != num_tasks:
        raise ValueError(
            f"task_names length ({len(task_names)}) does not match num_tasks ({num_tasks})"
        )

    threshold_arr = _resolve_thresholds(num_tasks, task_names, thresholds)

    task_metrics: Dict[str, Dict] = {}
    auc_values: List[float] = []
    pr_values: List[float] = []
    acc_values: List[float] = []
    f1_values: List[float] = []

    for task_idx, task_name in enumerate(task_names):
        y_true = all_labels[:, task_idx]
        y_prob = all_probs[:, task_idx]
        valid_mask = np.isfinite(y_true)

        if np.sum(valid_mask) == 0:
            task_metrics[task_name] = {
                'n_valid': 0,
                'positive_rate': np.nan,
                'auc_roc': np.nan,
                'pr_auc': np.nan,
                'accuracy': np.nan,
                'f1': np.nan,
                'threshold': float(threshold_arr[task_idx]),
                'confusion_matrix': [[0, 0], [0, 0]],
            }
            auc_values.append(np.nan)
            pr_values.append(np.nan)
            acc_values.append(np.nan)
            f1_values.append(np.nan)
            continue

        y_true_valid = y_true[valid_mask].astype(int)
        y_prob_valid = y_prob[valid_mask]
        y_pred_valid = (y_prob_valid >= threshold_arr[task_idx]).astype(int)

        if len(np.unique(y_true_valid)) > 1:
            task_auc = roc_auc_score(y_true_valid, y_prob_valid)
            task_pr = average_precision_score(y_true_valid, y_prob_valid)
        else:
            task_auc = np.nan
            task_pr = np.nan

        task_acc = accuracy_score(y_true_valid, y_pred_valid)
        task_f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0.0)
        task_cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1]).tolist()

        task_metrics[task_name] = {
            'n_valid': int(y_true_valid.shape[0]),
            'positive_rate': float(np.mean(y_true_valid == 1)),
            'auc_roc': float(task_auc) if np.isfinite(task_auc) else np.nan,
            'pr_auc': float(task_pr) if np.isfinite(task_pr) else np.nan,
            'accuracy': float(task_acc),
            'f1': float(task_f1),
            'threshold': float(threshold_arr[task_idx]),
            'confusion_matrix': task_cm,
        }

        auc_values.append(task_auc)
        pr_values.append(task_pr)
        acc_values.append(task_acc)
        f1_values.append(task_f1)

    threshold_matrix = np.broadcast_to(threshold_arr.reshape(1, -1), all_probs.shape)
    valid_mask_all = np.isfinite(all_labels)
    flat_true = all_labels[valid_mask_all].astype(int)
    flat_prob = all_probs[valid_mask_all]
    flat_pred = (all_probs >= threshold_matrix)[valid_mask_all].astype(int)

    if flat_true.size > 0 and len(np.unique(flat_true)) > 1:
        micro_auc = float(roc_auc_score(flat_true, flat_prob))
        micro_pr_auc = float(average_precision_score(flat_true, flat_prob))
    else:
        micro_auc = 0.0
        micro_pr_auc = 0.0

    if flat_true.size > 0:
        micro_acc = float(accuracy_score(flat_true, flat_pred))
        micro_f1 = float(f1_score(flat_true, flat_pred, zero_division=0.0))
    else:
        micro_acc = 0.0
        micro_f1 = 0.0

    metrics = {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'macro_auc_roc': _nanmean_or_zero(auc_values),
        'macro_pr_auc': _nanmean_or_zero(pr_values),
        'macro_accuracy': _nanmean_or_zero(acc_values),
        'macro_f1': _nanmean_or_zero(f1_values),
        'micro_auc_roc': micro_auc,
        'micro_pr_auc': micro_pr_auc,
        'micro_accuracy': micro_acc,
        'micro_f1': micro_f1,
        'task_metrics': task_metrics,
    }

    if return_predictions:
        metrics['probabilities'] = all_probs
        metrics['labels'] = all_labels
        metrics['logits'] = all_logits

    return metrics


def train_multitask_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    task_names: Optional[List[str]] = None,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    loss_type: str = "bce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: Optional[torch.Tensor] = None,
    early_stopping_patience: int = 20,
    early_stopping_metric: str = "macro_auc_roc",
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train a graph model for multi-task binary classification with missing labels.

    Args:
        model: Model producing logits of shape (batch, num_tasks).
        train_loader: Training dataloader.
        val_loader: Optional validation dataloader.
        task_names: Optional list of task names.
        num_epochs: Number of epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization factor.
        device: Device string.
        loss_type: One of 'bce', 'weighted_bce', or 'focal'.
        focal_alpha: Alpha factor for focal loss.
        focal_gamma: Gamma factor for focal loss.
        pos_weight: Optional per-task positive class weights.
        early_stopping_patience: Early stopping patience in epochs.
        early_stopping_metric: One of:
            'macro_auc_roc', 'macro_pr_auc', 'macro_f1',
            'micro_auc_roc', 'micro_pr_auc', 'micro_f1', 'loss'.
        verbose: Whether to log progress.

    Returns:
        History dictionary containing training and validation trajectories.
    """
    model = model.to(device)

    if loss_type == "focal":
        criterion = MaskedFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean'
        )
    elif loss_type == "weighted_bce":
        criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    else:
        criterion = MaskedBCEWithLogitsLoss(reduction='mean')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    optimize_for_loss = early_stopping_metric == 'loss'
    scheduler_kwargs = {
        'mode': 'min' if optimize_for_loss else 'max',
        'factor': 0.5,
        'patience': 10,
    }
    if 'verbose' in inspect.signature(
        torch.optim.lr_scheduler.ReduceLROnPlateau
    ).parameters:
        scheduler_kwargs['verbose'] = verbose

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **scheduler_kwargs,
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_macro_auc_roc': [],
        'val_macro_pr_auc': [],
        'val_macro_f1': [],
        'val_micro_auc_roc': [],
        'val_micro_pr_auc': [],
        'val_micro_f1': [],
    }

    best_metric = float('inf') if optimize_for_loss else float('-inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses: List[float] = []

        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.y.float()

            optimizer.zero_grad()

            logits = model(batch)
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1)
            if labels.dim() == 1 and logits.dim() == 2 and labels.numel() == logits.numel():
                labels = labels.view(logits.size(0), logits.size(1))
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            if labels.shape != logits.shape:
                raise ValueError(
                    f"Label shape {tuple(labels.shape)} does not match logits shape "
                    f"{tuple(logits.shape)} in train_multitask_model"
                )

            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        history['train_loss'].append(avg_train_loss)

        if val_loader is not None:
            val_metrics = evaluate_multitask_model(
                model=model,
                data_loader=val_loader,
                task_names=task_names,
                device=device,
            )

            history['val_loss'].append(val_metrics['loss'])
            history['val_macro_auc_roc'].append(val_metrics['macro_auc_roc'])
            history['val_macro_pr_auc'].append(val_metrics['macro_pr_auc'])
            history['val_macro_f1'].append(val_metrics['macro_f1'])
            history['val_micro_auc_roc'].append(val_metrics['micro_auc_roc'])
            history['val_micro_pr_auc'].append(val_metrics['micro_pr_auc'])
            history['val_micro_f1'].append(val_metrics['micro_f1'])

            current_metric = val_metrics.get(early_stopping_metric)
            if current_metric is None:
                raise ValueError(
                    f"Unknown early_stopping_metric '{early_stopping_metric}'"
                )

            if np.isnan(current_metric):
                current_metric = float('inf') if optimize_for_loss else float('-inf')

            is_better = (
                current_metric < best_metric
                if optimize_for_loss
                else current_metric > best_metric
            )

            if is_better:
                best_metric = current_metric
                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            scheduler_metric = current_metric
            if not np.isfinite(scheduler_metric):
                scheduler_metric = 0.0
            scheduler.step(scheduler_metric)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Macro AUC: {val_metrics['macro_auc_roc']:.4f}, "
                    f"Val Macro PR-AUC: {val_metrics['macro_pr_auc']:.4f}, "
                    f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
                )

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        elif verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history

