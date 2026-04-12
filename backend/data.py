"""
Data loading and preprocessing for toxicity prediction datasets.

Provides functions to load ClinTox and Tox21 datasets with proper
train/validation/test splits using scaffold-based splitting.
"""

from typing import Tuple, Optional, List
from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.workspace_mode import assert_clintox_enabled, assert_tox21_enabled


def _splitter_name(split_type: str) -> str:
    mode = str(split_type).strip().lower()
    if mode == "scaffold":
        return "ScaffoldSplitter"
    if mode == "stratified":
        return "RandomStratifiedSplitter"
    return "RandomSplitter"


def _load_dc_cached_split(split_dir: Path, task_names: List[str]) -> Optional[pd.DataFrame]:
    shard_id_files = sorted(split_dir.glob("shard-*-ids.npy"))
    if not shard_id_files:
        return None

    smiles_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    for ids_path in shard_id_files:
        stem_parts = ids_path.stem.split("-")
        if len(stem_parts) < 2:
            continue

        shard_idx = stem_parts[1]
        y_path = split_dir / f"shard-{shard_idx}-y.npy"
        w_path = split_dir / f"shard-{shard_idx}-w.npy"
        if not y_path.exists():
            continue

        ids_arr = np.load(ids_path, allow_pickle=True)
        y_arr = np.load(y_path, allow_pickle=True)
        w_arr = np.load(w_path, allow_pickle=True) if w_path.exists() else None

        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        if y_arr.shape[1] != len(task_names):
            continue

        # DeepChem stores missing labels via task weights (w == 0).
        # Align with teacher pipeline: convert these entries to NaN.
        if w_arr is not None:
            if w_arr.ndim == 1:
                w_arr = w_arr.reshape(-1, 1)
            if w_arr.shape == y_arr.shape:
                y_arr = y_arr.astype(np.float32, copy=False)
                y_arr[w_arr == 0] = np.nan

        smiles_chunks.append(np.asarray(ids_arr))
        label_chunks.append(np.asarray(y_arr, dtype=np.float32))

    if not smiles_chunks or not label_chunks:
        return None

    smiles = np.concatenate(smiles_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)

    data = {"smiles": [str(x) for x in smiles.tolist()]}
    for idx, task_name in enumerate(task_names):
        data[task_name] = labels[:, idx]

    return pd.DataFrame(data).replace(-1, np.nan)


def _load_tox21_from_cached_rawfeaturizer(
    cache_dir: str,
    split_type: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    cache_path = Path(cache_dir)
    root = (
        cache_path
        / "tox21-featurized"
        / "RawFeaturizer"
        / _splitter_name(split_type)
        / "BalancingTransformer"
    )

    train_dir = root / "train_dir"
    valid_dir = root / "valid_dir"
    test_dir = root / "test_dir"
    if not (train_dir.exists() and valid_dir.exists() and test_dir.exists()):
        return None

    tasks_path = train_dir / "tasks.json"
    if not tasks_path.exists():
        return None

    try:
        with open(tasks_path, "r") as f:
            task_names = json.load(f)
    except Exception:
        return None

    if not isinstance(task_names, list) or not task_names:
        return None

    train_df = _load_dc_cached_split(train_dir, task_names)
    val_df = _load_dc_cached_split(valid_dir, task_names)
    test_df = _load_dc_cached_split(test_dir, task_names)

    if train_df is None or val_df is None or test_df is None:
        return None

    return train_df, val_df, test_df


def load_clintox(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42,
    enforce_workspace_mode: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load ClinTox dataset with train/val/test splits.
    
    Args:
        cache_dir: Directory to cache downloaded datasets
        split_type: Type of split ('scaffold', 'random', 'stratified')
        seed: Random seed for reproducibility
        enforce_workspace_mode: Whether to enforce workspace mode guards.
    
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames with columns:
        - 'smiles': SMILES strings
        - 'CT_TOX': Binary toxicity label (1 = toxic, 0 = non-toxic)
    
    Example:
        >>> train, val, test = load_clintox()
        >>> print(f"Train size: {len(train)}")
    """
    if enforce_workspace_mode:
        assert_clintox_enabled("load_clintox")

    import os
    from pathlib import Path
    
    # Try DeepChem first
    try:
        import deepchem as dc
        from deepchem.molnet import load_clintox as dc_load_clintox
        
        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load dataset using DeepChem
        # Map split_type to DeepChem splitter
        if split_type == "scaffold":
            splitter = dc.splits.ScaffoldSplitter()
        elif split_type == "random":
            splitter = dc.splits.RandomSplitter()
        elif split_type == "stratified":
            splitter = dc.splits.RandomStratifiedSplitter()
        else:
            splitter = dc.splits.RandomSplitter()  # Default fallback
        
        tasks, datasets, transformers = dc_load_clintox(
            data_dir=str(cache_path),
            save_dir=str(cache_path),
            featurizer=dc.feat.RawFeaturizer(),  # Just return SMILES
            splitter=splitter,
            seed=seed
        )
        
        train_dataset, val_dataset, test_dataset = datasets
        
        # DeepChem ClinTox has 2 tasks: ['FDA_APPROVED', 'CT_TOX']
        # We want the CT_TOX task (index 1, or second column)
        # Also handle cases where some molecules might have failed featurization
        # by filtering based on valid weight mask if available
        
        # Extract SMILES from ids (ids contains SMILES strings)
        # Extract labels - ClinTox has 2 tasks, we want CT_TOX (index 1)
        def extract_data(dataset):
            smiles = dataset.ids
            # If y is 2D, take the CT_TOX column (index 1), else flatten
            if len(dataset.y.shape) == 2 and dataset.y.shape[1] == 2:
                # ClinTox has [FDA_APPROVED, CT_TOX] - we want CT_TOX (column 1)
                labels = dataset.y[:, 1]
            elif len(dataset.y.shape) == 2:
                # If it's 2D but not 2 columns, take first column
                labels = dataset.y[:, 0]
            else:
                # 1D array
                labels = dataset.y.flatten()
            
            # Ensure same length (filter out any mismatches)
            min_len = min(len(smiles), len(labels))
            return pd.DataFrame({
                'smiles': smiles[:min_len],
                'CT_TOX': labels[:min_len]
            })
        
        train_df = extract_data(train_dataset)
        val_df = extract_data(val_dataset)
        test_df = extract_data(test_dataset)
        
        return train_df, val_df, test_df
    
    except ImportError:
        # Fallback to PyTDC
        try:
            from tdc.single_pred import Tox
            import os
            
            # Create cache directory
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Load dataset
            data = Tox(name='ClinTox', path=str(cache_path))
            df = data.get_data()
            
            # Rename columns to match expected format
            df = df.rename(columns={'Drug': 'smiles', 'Y': 'CT_TOX'})
            
            # Simple train/val/test split (80/10/10)
            # For scaffold split, would need additional processing
            np.random.seed(seed)
            shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            
            n = len(shuffled)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            
            train_df = shuffled[:n_train]
            val_df = shuffled[n_train:n_train + n_val]
            test_df = shuffled[n_train + n_val:]
            
            return train_df, val_df, test_df
        
        except ImportError:
            raise ImportError(
                "Neither DeepChem nor PyTDC is installed. "
                "Install one with: pip install deepchem or pip install pytdc"
            )


def load_tox21(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42,
    enforce_workspace_mode: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Tox21 multi-task toxicity dataset.
    
    Args:
        cache_dir: Directory to cache downloaded datasets
        split_type: Type of split ('scaffold', 'random', 'stratified')
        seed: Random seed for reproducibility
        enforce_workspace_mode: Whether to enforce workspace mode guards.
    
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames with columns:
        - 'smiles': SMILES strings
        - Multiple binary task columns (NR-AR, NR-AR-LBD, etc.)
        - Missing labels encoded as NaN
    
    Example:
        >>> train, val, test = load_tox21()
        >>> print(f"Number of tasks: {len([c for c in train.columns if c != 'smiles'])}")
    """
    if enforce_workspace_mode:
        assert_tox21_enabled("load_tox21")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cached = _load_tox21_from_cached_rawfeaturizer(
        cache_dir=str(cache_path),
        split_type=split_type,
    )
    if cached is not None:
        print(
            "[load_tox21] Using cached RawFeaturizer split from "
            f"{cache_path / 'tox21-featurized'}"
        )
        return cached
    
    # Try DeepChem first
    try:
        import deepchem as dc
        from deepchem.molnet import load_tox21 as dc_load_tox21

        print("[load_tox21] Cache miss for precomputed split. Falling back to DeepChem loader...")
        
        # Load dataset using DeepChem
        # Map split_type to DeepChem splitter
        if split_type == "scaffold":
            splitter = dc.splits.ScaffoldSplitter()
        elif split_type == "random":
            splitter = dc.splits.RandomSplitter()
        elif split_type == "stratified":
            splitter = dc.splits.RandomStratifiedSplitter()
        else:
            splitter = dc.splits.RandomSplitter()  # Default fallback
        
        tasks, datasets, transformers = dc_load_tox21(
            data_dir=str(cache_path),
            save_dir=str(cache_path),
            featurizer=dc.feat.RawFeaturizer(),  # Just return SMILES
            splitter=splitter,
            seed=seed
        )
        
        train_dataset, val_dataset, test_dataset = datasets
        
        # Convert to DataFrame with all tasks.
        # Prefer DeepChem weight masks (w == 0) for missing labels,
        # then retain -1 -> NaN fallback for compatibility.
        train_data = {'smiles': train_dataset.ids}
        val_data = {'smiles': val_dataset.ids}
        test_data = {'smiles': test_dataset.ids}

        for i, task in enumerate(tasks):
            train_col = np.asarray(train_dataset.y[:, i], dtype=np.float32).copy()
            val_col = np.asarray(val_dataset.y[:, i], dtype=np.float32).copy()
            test_col = np.asarray(test_dataset.y[:, i], dtype=np.float32).copy()

            train_w = getattr(train_dataset, "w", None)
            val_w = getattr(val_dataset, "w", None)
            test_w = getattr(test_dataset, "w", None)

            if train_w is not None and np.asarray(train_w).ndim == 2 and train_w.shape[1] > i:
                train_col[np.asarray(train_w[:, i]) == 0] = np.nan
            if val_w is not None and np.asarray(val_w).ndim == 2 and val_w.shape[1] > i:
                val_col[np.asarray(val_w[:, i]) == 0] = np.nan
            if test_w is not None and np.asarray(test_w).ndim == 2 and test_w.shape[1] > i:
                test_col[np.asarray(test_w[:, i]) == 0] = np.nan

            train_data[task] = train_col
            val_data[task] = val_col
            test_data[task] = test_col
        
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        test_df = pd.DataFrame(test_data)
        
        # Replace -1 with NaN for missing labels (DeepChem convention)
        train_df = train_df.replace(-1, np.nan)
        val_df = val_df.replace(-1, np.nan)
        test_df = test_df.replace(-1, np.nan)
        
        return train_df, val_df, test_df
    
    except ImportError:
        # Fallback to PyTDC
        try:
            from tdc.single_pred import Tox
            
            # Create cache directory
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Load dataset
            data = Tox(name='Tox21', path=str(cache_path))
            df = data.get_data()
            
            # Rename columns to match expected format
            df = df.rename(columns={'Drug': 'smiles'})
            
            # Simple train/val/test split (80/10/10)
            np.random.seed(seed)
            shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            
            n = len(shuffled)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            
            train_df = shuffled[:n_train]
            val_df = shuffled[n_train:n_train + n_val]
            test_df = shuffled[n_train + n_val:]
            
            return train_df, val_df, test_df
        
        except ImportError:
            raise ImportError(
                "Neither DeepChem nor PyTDC is installed. "
                "Install one with: pip install deepchem or pip install pytdc"
            )


def get_task_names(dataset_name: str = "tox21") -> List[str]:
    """
    Get list of task/column names for a dataset.
    
    Args:
        dataset_name: Name of dataset ('clintox' or 'tox21')
    
    Returns:
        List of task column names
    """
    if dataset_name == "clintox":
        return ["CT_TOX"]
    elif dataset_name == "tox21":
        return [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
            "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

