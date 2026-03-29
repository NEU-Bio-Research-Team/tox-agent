"""Python compatibility aliases for legacy ``src.*`` imports.

The ML backend moved to the ``backend`` package, but some runtime modules
still import via ``src.*``. Keep this alias list intentionally small so
server startup does not eagerly import optional analysis/training deps.
"""

from importlib import import_module
import sys

# Map only runtime-critical modules to avoid importing optional packages
# (for example, seaborn used by analysis-only workflows) at startup.
_ALIAS_MODULES = [
    "workspace_mode",
    "featurization",
    "data",
    "graph_data",
    "graph_models",
    "graph_models_hybrid",
    "smiles_tokenizer",
]

for _name in _ALIAS_MODULES:
    sys.modules[f"src.{_name}"] = import_module(f"backend.{_name}")
