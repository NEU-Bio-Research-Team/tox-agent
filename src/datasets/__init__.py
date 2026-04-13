"""Unified dataset interface used by training and ensemble scripts."""

from typing import Tuple

import pandas as pd

from src.datasets.base import (
    CLINTOX_CONFIG,
    CLINTOX_TASKS,
    TOX21_CONFIG,
    TOX21_TASKS,
    TaskConfig,
    get_task_config,
)
from src.datasets.clintox import load_clintox
from src.datasets.tox21 import load_tox21


def load_dataset(
    dataset_name: str,
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a supported dataset by name."""

    dataset_key = dataset_name.strip().lower()
    loaders = {"clintox": load_clintox, "tox21": load_tox21}
    if dataset_key not in loaders:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Available: {sorted(loaders.keys())}"
        )
    return loaders[dataset_key](cache_dir=cache_dir, split_type=split_type, seed=seed)


__all__ = [
    "TaskConfig",
    "TOX21_TASKS",
    "CLINTOX_TASKS",
    "CLINTOX_CONFIG",
    "TOX21_CONFIG",
    "get_task_config",
    "load_dataset",
    "load_clintox",
    "load_tox21",
]
