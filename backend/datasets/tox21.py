"""Compatibility loader for Tox21."""

from typing import Tuple

import pandas as pd

from backend.data import load_tox21 as _load_tox21


def load_tox21(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Tox21 using the backend loader."""

    return _load_tox21(
        cache_dir=cache_dir,
        split_type=split_type,
        seed=seed,
        enforce_workspace_mode=False,
    )
