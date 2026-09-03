"""Compatibility loader for ClinTox."""

from typing import Tuple

import pandas as pd

from backend.data import load_clintox as _load_clintox


def load_clintox(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ClinTox using the backend loader."""

    return _load_clintox(
        cache_dir=cache_dir,
        split_type=split_type,
        seed=seed,
        enforce_workspace_mode=False,
    )
